"use client";

/*
 * 02 · Nueva sesion (escritorio) y 08 · Subir desde el celular (04 §02).
 *
 * Formulario en pasos dentro de la misma pagina, cada uno plegable cuando esta completo. La
 * 08 del diseno es esta misma ruta en la base; la 02 es su layout de escritorio, con el
 * formulario a la izquierda y checklist, consentimiento y resumen a la derecha.
 *
 * La subida va con XHR (progreso real) y se puede cancelar. Mientras sube, salir de la
 * pagina pide confirmacion (beforeunload).
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Encabezado } from "@/components/boxtwin/Encabezado";
import { Casilla, Paso, SelectNativo } from "@/components/boxtwin/Controles";
import { Esquina } from "@/components/boxtwin/Esquina";
import { ErrorConReintento, Esqueleto } from "@/components/boxtwin/Estados";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { Progress } from "@/components/animate-ui/components/radix/progress";
import { ToggleGroup, ToggleGroupItem } from "@/components/animate-ui/components/radix/toggle-group";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription,
  AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from "@/components/animate-ui/components/radix/alert-dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { HojaPeleador } from "@/components/pantallas/peleadores/HojaPeleador";
import { ErrorApi, subirConProgreso, type Subida } from "@/lib/api/cliente";
import { asignarBoxeadores, claves, useBoxeadores, useProducto } from "@/lib/api/consultas";
import { CONTEXTOS, PRACTICAS, type Contexto } from "@/lib/api/esquemas";
import { estadoDe } from "@/lib/banderas";
import { megabytes, mmss, nombreCorto, porcentaje } from "@/lib/formato";
import {
  CONDICIONES_FILMACION, avisoDeCalidad, estimacion, leerMetadatos, mmssASegundos,
  validarArchivo, validarDuracion, type Metadatos,
} from "@/lib/subida";
import { DESCRIPCION_CONTEXTO, NOMBRE_CONTEXTO, NOMBRE_PRACTICA, textos } from "@/lib/textos";

const SIN_PERFIL = "__sin_perfil__";
const AGREGAR = "__agregar__";
const DE_DOS: Contexto[] = ["sparring", "combate"];

type EstadoSubida =
  | { fase: "quieta" }
  | { fase: "subiendo"; enviados: number; total: number }
  | { fase: "error"; mensaje: string };

export function PantallaNueva() {
  const router = useRouter();
  const qc = useQueryClient();
  const producto = useProducto();
  const boxeadores = useBoxeadores();

  // 1. Video
  const [archivo, setArchivo] = useState<File | null>(null);
  const [meta, setMeta] = useState<Metadatos | null>(null);
  const [errorVideo, setErrorVideo] = useState<string | null>(null);
  // 2. Contexto
  const [contexto, setContexto] = useState<Contexto>("sparring");
  const [practica, setPractica] = useState<string>("");
  // 3. Peleadores
  const [pelA, setPelA] = useState<string>(SIN_PERFIL);
  const [pelB, setPelB] = useState<string>(SIN_PERFIL);
  const [hojaPara, setHojaPara] = useState<"A" | "B" | null>(null);
  // 4. Rounds
  const [sinRounds, setSinRounds] = useState(false);
  const [cantidad, setCantidad] = useState("3");
  const [duracionRound, setDuracionRound] = useState("3:00");
  const [descanso, setDescanso] = useState("1:00");
  // 5 y 6
  const [condiciones, setCondiciones] = useState<Record<string, boolean>>({});
  const [consentimiento, setConsentimiento] = useState(false);
  // Pasos abiertos a mano despues de completos
  const [abiertos, setAbiertos] = useState<Record<number, boolean>>({});
  // Subida
  const [subida, setSubida] = useState<EstadoSubida>({ fase: "quieta" });
  const [confirmarCancelar, setConfirmarCancelar] = useState(false);
  const enCurso = useRef<Subida | null>(null);

  const p = producto.data;
  const individuales = estadoDe("02.individuales", p) === "medido";
  const deDos = DE_DOS.includes(contexto);

  // beforeunload mientras sube: no se abandona sin confirmar (04 §02).
  useEffect(() => {
    if (subida.fase !== "subiendo") return;
    const f = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", f);
    return () => window.removeEventListener("beforeunload", f);
  }, [subida.fase]);

  async function elegirArchivo(f: File | null) {
    setArchivo(null);
    setMeta(null);
    setErrorVideo(null);
    if (!f || !p) return;
    const motivo = validarArchivo(f, p);
    if (motivo) {
      setErrorVideo(motivo);
      return;
    }
    setArchivo(f);
    try {
      const m = await leerMetadatos(f);
      const largo = validarDuracion(m.duracion_s, p.max_minutos);
      if (largo) {
        setArchivo(null);
        setErrorVideo(largo);
        return;
      }
      setMeta(m);
    } catch {
      // Sin metadatos se puede subir igual: la duracion la informa el servidor despues.
      setMeta(null);
    }
  }

  const rondasValidas = useMemo(() => {
    if (sinRounds) return true;
    const n = Number(cantidad);
    return Number.isInteger(n) && n >= 1 && mmssASegundos(duracionRound) != null && mmssASegundos(descanso) != null;
  }, [sinRounds, cantidad, duracionRound, descanso]);

  const completos = {
    1: !!archivo,
    2: contexto !== "sombra" || !!practica,
    3: !deDos || pelA !== pelB || pelA === SIN_PERFIL,
    4: rondasValidas,
  };
  const mismoPeleador = deDos && pelA !== SIN_PERFIL && pelA === pelB;
  const faltanCondiciones = CONDICIONES_FILMACION.some((c) => !condiciones[c.id]);
  const listo = completos[1] && completos[2] && completos[3] && completos[4] && consentimiento && !mismoPeleador;

  function porQueNo(): string | null {
    if (!archivo) return "Falta elegir el video.";
    if (!completos[2]) return "Falta decir qué se practicó.";
    if (mismoPeleador) return "El mismo peleador no puede estar en los dos rincones.";
    if (!rondasValidas) return "Revisá los rounds: cantidad, duración y descanso en mm:ss.";
    if (!consentimiento) return "Falta confirmar el consentimiento de las personas filmadas.";
    return null;
  }

  const est = meta && p ? estimacion(meta.duracion_s, p.factor_procesamiento) : null;
  const lista = boxeadores.data ?? [];
  const nombreDe = (id: string) => lista.find((b) => b.id === id)?.nombre ?? null;

  function tituloDeSesion() {
    const a = nombreDe(pelA);
    const b = deDos ? nombreDe(pelB) : null;
    const quien = a && b ? `${nombreCorto(a)} vs ${nombreCorto(b)}` : a ? nombreCorto(a) : b ? nombreCorto(b) : null;
    const base = NOMBRE_CONTEXTO[contexto];
    return quien ? `${base} · ${quien}` : archivo ? archivo.name.replace(/\.[^.]+$/, "") : base;
  }

  async function enviar() {
    if (!listo || !archivo) return;
    const datos = new FormData();
    datos.append("archivo", archivo);
    datos.append("nombre", tituloDeSesion());
    datos.append("contexto", contexto);
    if (contexto === "sombra" && practica) datos.append("practica", practica);
    datos.append("consentimiento", "true");
    if (!sinRounds) {
      datos.append("rounds", String(Number(cantidad)));
      datos.append("round_s", String(mmssASegundos(duracionRound)));
      datos.append("descanso_s", String(mmssASegundos(descanso)));
    }
    setSubida({ fase: "subiendo", enviados: 0, total: archivo.size });
    const s = subirConProgreso(datos, (enviados, total) => setSubida({ fase: "subiendo", enviados, total }));
    enCurso.current = s;
    try {
      const r = await s.promesa;
      enCurso.current = null;
      const a = pelA !== SIN_PERFIL ? pelA : null;
      const b = deDos && pelB !== SIN_PERFIL ? pelB : null;
      if (a || b) {
        try {
          await asignarBoxeadores(r.job_id, a, b);
        } catch {
          toast.warning("El video se subió, pero no se pudieron asignar los peleadores. Asignalos desde la sesión.");
        }
      }
      setSubida({ fase: "quieta" });
      qc.invalidateQueries({ queryKey: claves.sesiones });
      qc.invalidateQueries({ queryKey: claves.panel });
      toast.success("Subido. El análisis sigue aunque cierres la página.");
      router.push(`/sesion/?id=${r.job_id}`);
    } catch (err) {
      enCurso.current = null;
      if (err instanceof ErrorApi && err.estado === -1) {
        setSubida({ fase: "quieta" });
        toast("Subida cancelada. No quedó ninguna sesión creada.");
        return;
      }
      setSubida({ fase: "error", mensaje: err instanceof Error ? err.message : "no se pudo subir" });
    }
  }

  if (producto.isPending) return <Esqueleto filas={4} />;
  if (producto.isError) return <ErrorConReintento error={producto.error} alReintentar={() => producto.refetch()} />;

  const subiendo = subida.fase === "subiendo";
  const fraccion = subiendo && subida.total ? subida.enviados / subida.total : 0;

  const columnaDerecha = (
    <div className="flex flex-col gap-4">
      <section aria-labelledby="paso-5" className="rounded-md border border-borde bg-superficie p-4">
        <h2 id="paso-5" className="font-display text-xl font-semibold lg:text-2xl">Para que el análisis salga bien</h2>
        <div className="mt-3 flex flex-col">
          {CONDICIONES_FILMACION.map((c) => (
            <Casilla key={c.id} checked={!!condiciones[c.id]} onChange={(v) => setCondiciones((x) => ({ ...x, [c.id]: v }))}>
              {c.texto}
            </Casilla>
          ))}
        </div>
      </section>

      <section aria-labelledby="paso-6" className="rounded-md border border-borde bg-superficie p-4">
        <h2 id="paso-6" className="font-display text-xl font-semibold lg:text-2xl">Consentimiento</h2>
        <Casilla className="mt-2" checked={consentimiento} onChange={setConsentimiento}>
          {textos.consentimiento}
        </Casilla>
      </section>

      <section aria-labelledby="paso-7" className="rounded-md border border-borde bg-superficie p-4">
        <h2 id="paso-7" className="font-display text-xl font-semibold lg:text-2xl">Resumen</h2>
        <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
          <dt className="text-tenue">Sesión</dt>
          <dd>{tituloDeSesion()}</dd>
          <dt className="text-tenue">Rounds</dt>
          <dd>{sinRounds ? "sin rounds" : `${cantidad} de ${duracionRound}, descanso ${descanso}`}</dd>
          {est && (
            <>
              <dt className="text-tenue">Tiempo</dt>
              <dd>
                {est.texto}
                <span className="block text-xs text-texto-3">
                  estimación con lo medido en la estación; avisamos cuando esté listo
                </span>
              </dd>
            </>
          )}
        </dl>
        {faltanCondiciones && (
          <p className="mt-3 text-sm text-estimado">Puede que el análisis pierda tramos.</p>
        )}
      </section>
    </div>
  );

  return (
    <>
      <Encabezado titulo="Nueva sesión" rotulo="Subir" />

      <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr] lg:gap-6">
        <div className="flex flex-col gap-4">
          {/* 1. Video */}
          <Paso
            numero={1}
            titulo="Video"
            completo={completos[1]}
            abierto={!!abiertos[1]}
            onAbrir={() => setAbiertos((a) => ({ ...a, 1: true }))}
            resumen={archivo && `${archivo.name} · ${megabytes(archivo.size)}${meta ? ` · ${mmss(meta.duracion_s)}` : ""}`}
          >
            <ElegirVideo onElegir={(f) => { setAbiertos((a) => ({ ...a, 1: false })); void elegirArchivo(f); }} formatos={p!.formatos} />
            {errorVideo && <p role="alert" className="mt-3 rounded-md border border-error-borde bg-error-fondo px-3 py-2 text-sm text-error">{errorVideo}</p>}
            {archivo && (
              <p className="mt-3 dato text-texto-2">
                {archivo.name} · {megabytes(archivo.size)}
                {meta && ` · ${mmss(meta.duracion_s)} · ${meta.ancho}×${meta.alto}`}
              </p>
            )}
            {meta && avisoDeCalidad(meta) && <p className="mt-2 text-sm text-estimado">{avisoDeCalidad(meta)}</p>}
          </Paso>

          {/* 2. Que se filmo */}
          <Paso
            numero={2}
            titulo="Qué se filmó"
            completo={completos[2] && !!archivo}
            abierto={!!abiertos[2] || !archivo}
            onAbrir={() => setAbiertos((a) => ({ ...a, 2: true }))}
            resumen={`${NOMBRE_CONTEXTO[contexto]}${practica ? ` · ${NOMBRE_PRACTICA[practica]}` : ""}`}
          >
            <ToggleGroup
              type="single"
              value={contexto}
              onValueChange={(v) => v && setContexto(v as Contexto)}
              className="grid w-full grid-cols-2 gap-2 md:grid-cols-4"
              aria-label="Qué se filmó"
            >
              {CONTEXTOS.map((c) => {
                const proximamente = !individuales && !DE_DOS.includes(c);
                return (
                  <ToggleGroupItem
                    key={c}
                    value={c}
                    disabled={proximamente}
                    className="flex h-auto min-h-16 flex-col items-start gap-0.5 border border-borde px-3 py-2 text-left whitespace-normal"
                  >
                    <span className="font-medium">{NOMBRE_CONTEXTO[c]}</span>
                    <span className="text-xs text-texto-3">{proximamente ? "Próximamente" : DESCRIPCION_CONTEXTO[c]}</span>
                  </ToggleGroupItem>
                );
              })}
            </ToggleGroup>
            {contexto === "sombra" && (
              <fieldset className="mt-4">
                <legend className="mb-2 text-sm text-texto-2">Qué se practicó</legend>
                <ToggleGroup type="single" value={practica} onValueChange={setPractica} className="flex w-full flex-wrap gap-2">
                  {PRACTICAS.map((x) => (
                    <ToggleGroupItem key={x} value={x} className="min-h-11 border border-borde px-3">
                      {NOMBRE_PRACTICA[x]}
                    </ToggleGroupItem>
                  ))}
                </ToggleGroup>
              </fieldset>
            )}
          </Paso>

          {/* 3. Peleadores */}
          <Paso
            numero={3}
            titulo="Peleadores"
            completo={completos[3] && !mismoPeleador}
            abierto
            onAbrir={() => {}}
          >
            {boxeadores.isError && <ErrorConReintento error={boxeadores.error} alReintentar={() => boxeadores.refetch()} />}
            <div className={`grid gap-3 ${deDos ? "md:grid-cols-2" : ""}`}>
              {(deDos ? (["A", "B"] as const) : (["A"] as const)).map((l) => {
                const valor = l === "A" ? pelA : pelB;
                const otro = l === "A" ? pelB : pelA;
                return (
                  <div key={l} className="flex flex-col gap-1.5">
                    <Label htmlFor={`pel-${l}`} className="flex items-center gap-2">
                      <Esquina lado={l} /> {l === "A" ? "Rincón rojo" : "Rincón azul"}
                    </Label>
                    <SelectNativo
                      id={`pel-${l}`}
                      value={valor}
                      onChange={(e) => {
                        if (e.target.value === AGREGAR) {
                          setHojaPara(l);
                          return;
                        }
                        (l === "A" ? setPelA : setPelB)(e.target.value);
                      }}
                    >
                      <option value={SIN_PERFIL}>Rival sin perfil (no se guarda)</option>
                      {lista.map((b) => (
                        <option key={b.id} value={b.id} disabled={deDos && b.id === otro}>
                          {b.nombre}{b.categoria ? ` · ${b.categoria}` : ""}
                        </option>
                      ))}
                      <option value={AGREGAR}>+ Agregar peleador</option>
                    </SelectNativo>
                  </div>
                );
              })}
            </div>
            {mismoPeleador && <p role="alert" className="mt-2 text-sm text-error">El mismo peleador no puede estar en los dos rincones.</p>}
          </Paso>

          {/* 4. Rounds */}
          <Paso
            numero={4}
            titulo="Rounds"
            completo={completos[4]}
            abierto
            onAbrir={() => {}}
          >
            <Casilla checked={sinRounds} onChange={setSinRounds}>Sin rounds (video continuo)</Casilla>
            {!sinRounds && (
              <div className="mt-2 grid grid-cols-3 gap-3">
                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="r-cantidad">Cantidad</Label>
                  <Input id="r-cantidad" inputMode="numeric" value={cantidad} onChange={(e) => setCantidad(e.target.value)} className="h-11 text-base" />
                </div>
                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="r-duracion">Duración</Label>
                  <Input id="r-duracion" inputMode="numeric" placeholder="3:00" value={duracionRound} onChange={(e) => setDuracionRound(e.target.value)} className="h-11 text-base" />
                </div>
                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="r-descanso">Descanso</Label>
                  <Input id="r-descanso" inputMode="numeric" placeholder="1:00" value={descanso} onChange={(e) => setDescanso(e.target.value)} className="h-11 text-base" />
                </div>
              </div>
            )}
            {!rondasValidas && <p className="mt-2 text-sm text-error">Cantidad entera y tiempos en mm:ss, por ejemplo 3:00.</p>}
          </Paso>
        </div>

        {/* Una sola vez: en el celular la grilla es de una columna y esto cae despues de los
            pasos, que es el orden de 08; en escritorio es la columna derecha de 02. */}
        <div>{columnaDerecha}</div>
      </div>

      {/* Envio: fijo abajo en el celular, encima de la barra de navegacion (08). */}
      <div
        className="fixed inset-x-0 z-30 border-t border-borde-suave bg-fondo/95 px-4 py-3 backdrop-blur lg:static lg:mt-6 lg:border-0 lg:bg-transparent lg:p-0"
        style={{ bottom: "calc(env(safe-area-inset-bottom) + 64px)" }}
      >
        <div className="mx-auto max-w-[1200px] lg:max-w-none">
          {subiendo ? (
            <div role="status" aria-live="polite" className="flex flex-col gap-2">
              <div className="flex items-center justify-between text-sm">
                <span>Subiendo · {porcentaje(fraccion)}</span>
                <span className="dato text-texto-2">{megabytes(subida.enviados)} de {megabytes(subida.total)}</span>
              </div>
              <Progress value={fraccion * 100} aria-label="Progreso de la subida" />
              <Button variant="outline" className="min-h-11" onClick={() => setConfirmarCancelar(true)}>
                Cancelar
              </Button>
            </div>
          ) : (
            <>
              {subida.fase === "error" && (
                <p role="alert" className="mb-2 text-sm text-error">{subida.mensaje}. Un corte obliga a subir de nuevo.</p>
              )}
              <Button className="min-h-12 w-full text-base lg:w-auto lg:px-10" disabled={!listo} onClick={enviar}>
                Subir y procesar
              </Button>
              {!listo && <p className="mt-1 text-xs text-texto-3">{porQueNo()}</p>}
            </>
          )}
        </div>
      </div>
      <div aria-hidden="true" className="h-40 lg:hidden" />

      <AlertDialog open={confirmarCancelar} onOpenChange={setConfirmarCancelar}>
        <AlertDialogContent className="bg-superficie">
          <AlertDialogHeader>
            <AlertDialogTitle>¿Cancelar la subida?</AlertDialogTitle>
            <AlertDialogDescription className="text-texto-2">
              No queda ninguna sesión creada. Para analizar el video hay que subirlo de nuevo.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="min-h-11">Seguir subiendo</AlertDialogCancel>
            <AlertDialogAction className="min-h-11" onClick={() => enCurso.current?.cancelar()}>
              Cancelar subida
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {hojaPara && (
        <HojaPeleador
          abierta
          corta
          onCerrar={() => setHojaPara(null)}
          onGuardado={(b) => (hojaPara === "A" ? setPelA : setPelB)(b.id)}
        />
      )}
    </>
  );
}

/*
 * En el celular, dos botones grandes: "Filmar ahora" abre la camara trasera (capture) y
 * "Elegir video" la galeria. En escritorio, zona para soltar y boton.
 */
function ElegirVideo({ onElegir, formatos }: { onElegir: (f: File | null) => void; formatos: string[] }) {
  const camara = useRef<HTMLInputElement>(null);
  const galeria = useRef<HTMLInputElement>(null);
  const [encima, setEncima] = useState(false);
  const accept = ["video/*", ...formatos].join(",");
  return (
    <>
      <input ref={camara} type="file" accept="video/*" capture="environment" className="sr-only" tabIndex={-1}
        aria-hidden="true" onChange={(e) => onElegir(e.target.files?.[0] ?? null)} data-testid="input-camara" />
      <input ref={galeria} type="file" accept={accept} className="sr-only" tabIndex={-1}
        aria-hidden="true" onChange={(e) => onElegir(e.target.files?.[0] ?? null)} data-testid="input-galeria" />
      <div className="grid grid-cols-2 gap-3 lg:hidden">
        <Button type="button" className="min-h-16 text-base" onClick={() => camara.current?.click()}>Filmar ahora</Button>
        <Button type="button" variant="outline" className="min-h-16 text-base" onClick={() => galeria.current?.click()}>Elegir video</Button>
      </div>
      <div
        className={`hidden min-h-36 flex-col items-center justify-center gap-3 rounded-md border border-dashed p-6 text-center lg:flex ${encima ? "border-estimado-borde bg-estimado-fondo/30" : "border-borde-fuerte"}`}
        onDragOver={(e) => { e.preventDefault(); setEncima(true); }}
        onDragLeave={() => setEncima(false)}
        onDrop={(e) => { e.preventDefault(); setEncima(false); onElegir(e.dataTransfer.files?.[0] ?? null); }}
      >
        <p className="text-texto-2">Soltá el video acá</p>
        <Button type="button" variant="outline" className="min-h-11" onClick={() => galeria.current?.click()}>Elegir archivo</Button>
        <p className="text-xs text-texto-3">{formatos.join(", ")}</p>
      </div>
    </>
  );
}
