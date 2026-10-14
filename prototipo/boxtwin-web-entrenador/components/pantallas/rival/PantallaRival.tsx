"use client";

/*
 * 07 · Rival Fantasma (04 §07). Ruta /rival/?a=&b=.
 *
 * Lectura cualitativa entre dos perfiles ya construidos: no es un pronostico ni una
 * probabilidad de victoria. Solo metricas de sparring y combate de cada perfil.
 *
 * Cambios del 14-10 (specs/cambios-14-10/02): las dimensiones se dibujan con dos maniquies 3D
 * ilustrativos, con los brazos que se estiran y se recogen (W25, DC-9), y "oportunidades y
 * amenazas" y el cruce de zonas son un solo bloque, "Ataque contra exposicion" (W26, DC-8),
 * cuyas reglas viven en la API (GET /rival). Las reglas de metricas de W12 ya no se muestran.
 */
import { Desplazable } from "@/components/boxtwin/Desplazable";
import { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Encabezado } from "@/components/boxtwin/Encabezado";
import { Esquina } from "@/components/boxtwin/Esquina";
import { EstadoBloque, Estimado } from "@/components/boxtwin/EstadoBloque";
import { ErrorConReintento, Esqueleto, Vacio } from "@/components/boxtwin/Estados";
import { Rotulo } from "@/components/boxtwin/Rotulo";
import { SelectNativo } from "@/components/boxtwin/Controles";
import { Avatar3D } from "@/components/boxtwin/avatar/Avatar3D";
import { NOMBRE_ZONA_ATAQUE, SiluetaZonas } from "@/components/boxtwin/graficos/SiluetaZonas";
import { NOMBRE_DISTANCIA, NOMBRE_ESTILO } from "@/components/pantallas/peleador/BloquesTacticos";
import { ToggleGroup, ToggleGroupItem } from "@/components/animate-ui/components/radix/toggle-group";
import { useBoxeadores, usePerfil, useProducto, useRival } from "@/lib/api/consultas";
import { ZONAS, type Perfil, type Producto, type Rival, type Zona } from "@/lib/api/esquemas";
import { estadoDe } from "@/lib/banderas";
import { decimal, porcentajeAR } from "@/lib/formato";
import { alcance, medidasDe } from "@/lib/maniqui";
import { textos } from "@/lib/textos";

export interface FilaRival {
  nombre: string;
  altura_m: number | null;
  envergadura_m: number | null;
  guardia: string | null;
  distancia: string | null;
  estilo: string | null;
  golpes_por_minuto: number | null;
  jab_fraccion: number | null;
  guardia_opuesta_arriba: number | null;
  caida_ultimo_round: number | null;
}

const DE_PELEA = ["sparring", "combate"];

/* Las metricas de sparring y combate de un perfil, promediadas por cantidad de sesiones. */
export function filaDe(p: Perfil): FilaRival {
  const filas = (p.por_contexto ?? []).filter((f) => DE_PELEA.includes(f.contexto));
  const peso = (f: (typeof filas)[number]) => f.sesiones ?? 1;
  const media = (k: "golpes_por_minuto" | "jab_fraccion" | "guardia_opuesta_arriba" | "caida_ultimo_round") => {
    const con = filas.filter((f) => f[k] != null);
    const w = con.reduce((a, f) => a + peso(f), 0);
    return w ? con.reduce((a, f) => a + (f[k] as number) * peso(f), 0) / w : null;
  };
  return {
    nombre: p.boxeador.nombre,
    altura_m: p.boxeador.altura_m ?? null,
    envergadura_m: p.boxeador.envergadura_m ?? null,
    guardia: p.boxeador.guardia,
    distancia: p.boxeador.distancia ?? null,
    estilo: p.boxeador.estilo ?? null,
    golpes_por_minuto: media("golpes_por_minuto"),
    jab_fraccion: media("jab_fraccion"),
    guardia_opuesta_arriba: media("guardia_opuesta_arriba"),
    caida_ultimo_round: media("caida_ultimo_round"),
  };
}

export function PantallaRival() {
  const router = useRouter();
  const q = useSearchParams();
  const a = q.get("a") ?? "";
  const b = q.get("b") ?? "";
  const boxeadores = useBoxeadores();
  const producto = useProducto();
  const pa = usePerfil(a || null);
  const pb = usePerfil(b || null);
  const [superpuestas, setSuperpuestas] = useState("lado");

  function elegir(lado: "a" | "b", id: string) {
    const u = new URLSearchParams(q.toString());
    if (id) u.set(lado, id);
    else u.delete(lado);
    router.replace(`/rival/?${u.toString()}`);
  }

  if (boxeadores.isPending) return <Esqueleto />;
  if (boxeadores.isError) return <ErrorConReintento error={boxeadores.error} alReintentar={() => boxeadores.refetch()} />;
  const lista = boxeadores.data;

  return (
    <>
      <Encabezado titulo="Rival Fantasma" rotulo="Comparar dos perfiles">
        <p className="text-texto-2">{textos.rivalAviso}</p>
      </Encabezado>

      {lista.length < 2 ? (
        <Vacio titulo="Hacen falta dos peleadores con perfil">Cargá peleadores y asignalos a sus sesiones.</Vacio>
      ) : (
        <div className="flex flex-col gap-4">
          <div className="grid gap-3 md:grid-cols-2">
            {(["a", "b"] as const).map((l) => (
              <div key={l} className="flex flex-col gap-1.5">
                <label htmlFor={`rival-${l}`} className="flex items-center gap-2 text-sm"><Esquina lado={l === "a" ? "A" : "B"} /> Perfil {l.toUpperCase()}</label>
                <SelectNativo id={`rival-${l}`} value={l === "a" ? a : b} onChange={(e) => elegir(l, e.target.value)}>
                  <option value="">Elegí un peleador</option>
                  {lista.map((x) => (
                    // CA-07.2: el mismo peleador no puede estar en los dos lados.
                    <option key={x.id} value={x.id} disabled={x.id === (l === "a" ? b : a)}>
                      {x.nombre} · {x.sesiones ?? 0} {x.sesiones === 1 ? "sesión" : "sesiones"}
                    </option>
                  ))}
                </SelectNativo>
              </div>
            ))}
          </div>

          {a && b && a === b && <p role="alert" className="text-sm text-error">Elegí dos peleadores distintos.</p>}

          {a && b && a !== b && (pa.isPending || pb.isPending) && <Esqueleto filas={3} />}
          {(pa.isError || pb.isError) && <ErrorConReintento error={pa.error ?? pb.error} alReintentar={() => { pa.refetch(); pb.refetch(); }} />}
          {a && b && a !== b && pa.data && pb.data && <Comparacion pa={pa.data} pb={pb.data} producto={producto.data} superpuestas={superpuestas} onSuperpuestas={setSuperpuestas} />}
        </div>
      )}
    </>
  );
}

function Comparacion({ pa, pb, producto, superpuestas, onSuperpuestas }: { pa: Perfil; pb: Perfil; producto?: Producto; superpuestas: string; onSuperpuestas: (v: string) => void }) {
  const fa = filaDe(pa);
  const fb = filaDe(pb);
  const banderas = { comparacion_rounds: estadoDe("07.frente_r3", producto) === "medido", guardia: estadoDe("07.frente_guardia", producto) === "medido" };
  const conRounds = banderas.comparacion_rounds;
  const promedioRounds = (p: Perfil) => {
    const r = p.rounds_promedio ?? [];
    return r.length ? r.reduce((s, x) => s + x.golpes_por_minuto, 0) / r.length : null;
  };

  type Fila = { nombre: string; a: React.ReactNode; b: React.ReactNode; estado?: "apagado" };
  const filas: Fila[] = [
    { nombre: "Altura", a: fa.altura_m ? `${decimal(fa.altura_m, 2)} m` : "—", b: fb.altura_m ? `${decimal(fb.altura_m, 2)} m` : "—" },
    { nombre: "Envergadura", a: fa.envergadura_m ? `${decimal(fa.envergadura_m, 2)} m` : "—", b: fb.envergadura_m ? `${decimal(fb.envergadura_m, 2)} m` : "—" },
    { nombre: "Guardia", a: fa.guardia ?? "—", b: fb.guardia ?? "—" },
    // B17: cargados a mano, no medidos.
    { nombre: "Distancia de trabajo (cargada)", a: fa.distancia ? NOMBRE_DISTANCIA[fa.distancia] : "—", b: fb.distancia ? NOMBRE_DISTANCIA[fb.distancia] : "—" },
    { nombre: "Estilo (cargado)", a: fa.estilo ? NOMBRE_ESTILO[fa.estilo] : "—", b: fb.estilo ? NOMBRE_ESTILO[fb.estilo] : "—" },
    conRounds
      ? { nombre: "Golpes por minuto por round (promedio)", a: decimal(promedioRounds(pa), 1), b: decimal(promedioRounds(pb), 1) }
      : { nombre: "Golpes por minuto", a: decimal(fa.golpes_por_minuto, 1), b: decimal(fb.golpes_por_minuto, 1) },
    { nombre: "Jab sobre el total", a: fa.jab_fraccion == null ? "—" : <Estimado>{porcentajeAR(fa.jab_fraccion)}</Estimado>, b: fb.jab_fraccion == null ? "—" : <Estimado>{porcentajeAR(fb.jab_fraccion)}</Estimado> },
    banderas.guardia
      ? { nombre: "Guardia opuesta arriba", a: porcentajeAR(fa.guardia_opuesta_arriba), b: porcentajeAR(fb.guardia_opuesta_arriba) }
      : { nombre: "Guardia opuesta arriba", a: null, b: null, estado: "apagado" },
    conRounds
      ? { nombre: "Caída en el último round", a: porcentajeAR(fa.caida_ultimo_round), b: porcentajeAR(fb.caida_ultimo_round) }
      : { nombre: "Caída en el último round", a: null, b: null, estado: "apagado" },
  ];

  return (
    <>
      <EstadoBloque
        estado={estadoDe("07.dimensiones", producto)}
        titulo="Dimensiones reales"
        accion={
          <ToggleGroup type="single" value={superpuestas} onValueChange={(v) => v && onSuperpuestas(v)} aria-label="Cómo dibujar las siluetas" className="gap-1">
            <ToggleGroupItem value="lado" className="min-h-11 border border-borde px-3">Lado a lado</ToggleGroupItem>
            <ToggleGroupItem value="encima" className="min-h-11 border border-borde px-3">Superpuestas</ToggleGroupItem>
          </ToggleGroup>
        }
      >
        <Dimensiones pa={pa} pb={pb} superpuestas={superpuestas === "encima"} />
      </EstadoBloque>

      <EstadoBloque estado={estadoDe("07.frente", producto)} titulo="Frente a frente" nota="Solo sesiones de sparring y combate de cada perfil.">
        <Desplazable nombre="Tabla frente a frente">
          <table className="w-full min-w-[420px] text-sm">
            <caption className="sr-only">Frente a frente</caption>
            <thead>
              <tr>
                <th scope="col" className="rotulo py-2 text-left font-normal">Métrica</th>
                <th scope="col" className="py-2 text-right font-normal"><span className="inline-flex items-center gap-1"><Esquina lado="A" tamano={16} />{fa.nombre}</span></th>
                <th scope="col" className="py-2 text-right font-normal"><span className="inline-flex items-center gap-1"><Esquina lado="B" tamano={16} />{fb.nombre}</span></th>
              </tr>
            </thead>
            <tbody>
              {filas.map((f) => (
                <tr key={f.nombre} className="border-t border-borde-suave">
                  <th scope="row" className="py-2 pr-3 text-left font-normal">{f.nombre}</th>
                  {f.estado === "apagado" ? (
                    <td colSpan={2} className="py-2 text-right"><Rotulo>en validación</Rotulo></td>
                  ) : (
                    <>
                      <td className="dato py-2 text-right">{f.a}</td>
                      <td className="dato py-2 text-right">{f.b}</td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </Desplazable>
      </EstadoBloque>

      <AtaqueContraExposicion a={pa.boxeador.id} b={pb.boxeador.id} nombres={{ A: fa.nombre, B: fb.nombre }} producto={producto} />
    </>
  );
}

// -- W25 · Dimensiones con avatares 3D (DC-9) --------------------------------------------

type Extension = { adelantada: number; atrasada: number };

function Dimensiones({ pa, pb, superpuestas }: { pa: Perfil; pb: Perfil; superpuestas: boolean }) {
  const [ext, setExt] = useState<Record<"A" | "B", Extension>>({
    A: { adelantada: 0, atrasada: 0 },
    B: { adelantada: 0, atrasada: 0 },
  });
  const perfiles = { A: pa, B: pb };
  const medidas = {
    A: medidasDe(pa.boxeador.altura_m, pa.boxeador.envergadura_m),
    B: medidasDe(pb.boxeador.altura_m, pb.boxeador.envergadura_m),
  };
  const guardia = (p: Perfil) => (p.boxeador.guardia === "zurda" || p.boxeador.guardia === "ortodoxa" ? p.boxeador.guardia : null);
  // Distancia de presentacion: la suma de los dos alcances extendidos (07 §1). No es medida.
  const separacion = alcance(medidas.A, 1) + alcance(medidas.B, 1);
  // Lo que falta se dice con nombre: una medida inventada es un numero inventado (04 §07).
  const descripcion = (l: "A" | "B") => `${perfiles[l].boxeador.nombre}: ${faltante(perfiles[l])}`;
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <Rotulo className="text-estimado">{textos.avatarIlustrativo}</Rotulo>
        {!superpuestas && <Rotulo>{textos.distanciaPresentacion}</Rotulo>}
      </div>
      <Avatar3D
        figuras={[
          { medidas: medidas.A, guardia: guardia(pa), lado: "A", extension: ext.A, x: 0, mira: 1, opacidad: superpuestas ? 0.6 : 1 },
          { medidas: medidas.B, guardia: guardia(pb), lado: "B", extension: ext.B, x: superpuestas ? 0 : separacion, mira: superpuestas ? 1 : -1, opacidad: superpuestas ? 0.6 : 1 },
        ]}
        etiqueta={`Dos maniquíes en guardia, ${superpuestas ? "superpuestos" : "enfrentados"}. ${descripcion("A")}. ${descripcion("B")}.`}
        alternativa={`${descripcion("A")}. ${descripcion("B")}.`}
        alto={300}
      />
      <div className="grid gap-4 md:grid-cols-2">
        {(["A", "B"] as const).map((l) => (
          <fieldset key={l} className="flex flex-col gap-3 rounded-md border border-borde p-3">
            <legend className="flex items-center gap-2 px-1 text-sm"><Esquina lado={l} tamano={16} />{perfiles[l].boxeador.nombre}</legend>
            {!medidas[l].aEscala && <p className="text-xs text-texto-3">{faltante(perfiles[l])}.</p>}
            {(["adelantada", "atrasada"] as const).map((brazo) => {
              const id = `brazo-${l}-${brazo}`;
              const valor = ext[l][brazo];
              return (
                <div key={brazo} className="flex flex-col gap-1">
                  <label htmlFor={id} className="flex items-baseline justify-between text-sm">
                    <span>Brazo {brazo === "adelantada" ? "adelantado" : "atrasado"}</span>
                    <span className="dato text-texto-2" data-alcance={`${l}-${brazo}`}>alcance {decimal(alcance(medidas[l], valor), 2)} m</span>
                  </label>
                  <input
                    id={id}
                    type="range"
                    min={0}
                    max={100}
                    step={5}
                    value={Math.round(valor * 100)}
                    onChange={(e) => setExt((x) => ({ ...x, [l]: { ...x[l], [brazo]: Number(e.target.value) / 100 } }))}
                    aria-valuetext={`${Math.round(valor * 100)}% extendido, alcance ${decimal(alcance(medidas[l], valor), 2)} metros`}
                    className="h-11 w-full accent-[var(--color-texto)]"
                  />
                </div>
              );
            })}
          </fieldset>
        ))}
      </div>
    </div>
  );
}

function faltante(p: Perfil): string {
  const { altura_m: h, envergadura_m: e } = p.boxeador;
  if (h != null && e != null) return `${decimal(h, 2)} m de altura y ${decimal(e, 2)} m de envergadura`;
  if (h != null) return `${decimal(h, 2)} m de altura; falta la envergadura, los brazos van en proporción estándar`;
  if (e != null) return `${decimal(e, 2)} m de envergadura; falta la altura, el cuerpo va en proporción estándar`;
  return "sin medidas cargadas, en proporciones estándar";
}

// -- W26 · Ataque contra exposicion (DC-8) ------------------------------------------------

function AtaqueContraExposicion({ a, b, nombres, producto }: { a: string; b: string; nombres: Record<"A" | "B", string>; producto?: Producto }) {
  const estado = estadoDe("07.cruces", producto);
  const rival = useRival(a, b, estado === "estimacion");
  const [atacante, setAtacante] = useState<"A" | "B">("A");
  if (estado !== "estimacion") {
    return <EstadoBloque estado={estado} titulo="Ataque contra exposición" motivo={textos.nodispExposicion} />;
  }
  return (
    <EstadoBloque
      estado="estimacion"
      titulo="Ataque contra exposición"
      nota={rival.data?.sesgo_detector.corregido ? textos.exposicionCorregida(rival.data.sesgo_detector.recall_cuerpo, rival.data.sesgo_detector.recall_cabeza) : undefined}
    >
      <p className="mb-3 text-sm text-texto-2">
        Dónde tira cada uno, contra dónde suelen tirarle al otro. &quot;Le tiran&quot; son golpes dirigidos: no dice si tocaron. {textos.cruceSinElOtro}
      </p>
      {rival.isPending ? (
        <Esqueleto filas={2} />
      ) : rival.isError ? (
        <ErrorConReintento error={rival.error} alReintentar={() => rival.refetch()} />
      ) : soloEntreEllos(rival.data).length > 0 ? (
        // DC-10: sin exposicion de otros rivales no hay cruce posible, y se dice por que.
        <ul className="space-y-2 text-sm text-estimado">
          {soloEntreEllos(rival.data).map((l) => (
            <li key={l}>{textos.cruceSoloEntreEllos(nombres[l], nombres[l === "A" ? "B" : "A"])}</li>
          ))}
        </ul>
      ) : rival.data.cruces.length === 0 ? (
        <p className="text-sm text-texto-3">{rival.data.faltan ? textos.cruceFaltan(rival.data.base.A, rival.data.base.B, rival.data.faltan.minimo) : "Todavía no hay golpes con altura en sparring o combate para cruzar."}</p>
      ) : (
        <ContenidoCruce r={rival.data} nombres={nombres} atacante={atacante} onAtacante={setAtacante} />
      )}
    </EstadoBloque>
  );
}

/* Los lados a los que no les queda exposicion porque todas sus sesiones eran contra el otro. */
function soloEntreEllos(r: Rival): ("A" | "B")[] {
  return (["A", "B"] as const).filter((l) => r.base[l].exposicion === 0 && (r.base[l].excluidas ?? 0) > 0);
}

function ContenidoCruce({ r, nombres, atacante, onAtacante }: {
  r: Rival; nombres: Record<"A" | "B", string>; atacante: "A" | "B"; onAtacante: (l: "A" | "B") => void;
}) {
  const receptor = atacante === "A" ? "B" : "A";
  const cruces = r.cruces.filter((c) => c.atacante === atacante);
  const at = Object.fromEntries(ZONAS.map((z) => [z, cruces.find((c) => c.zona === z)?.ataque_fraccion ?? null])) as Record<Zona, number | null>;
  const ex = Object.fromEntries(ZONAS.map((z) => [z, cruces.find((c) => c.zona === z)?.exposicion_fraccion ?? null])) as Record<Zona, number | null>;
  const colorear = !r.faltan;
  return (
    <div className="flex flex-col gap-4">
      <ToggleGroup type="single" value={atacante} onValueChange={(v) => v && onAtacante(v as "A" | "B")} aria-label="Quién ataca" className="gap-1">
        {(["A", "B"] as const).map((l) => (
          <ToggleGroupItem key={l} value={l} className="min-h-11 border border-borde px-3">
            <Esquina lado={l} tamano={14} /> {nombres[l]} ataca
          </ToggleGroupItem>
        ))}
      </ToggleGroup>

      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <p className="mb-1 text-center text-sm">Hacia dónde tira {nombres[atacante]}</p>
          <SiluetaZonas fracciones={at} colorear={colorear} ancho={160} etiqueta={`Hacia dónde tira ${nombres[atacante]}: ${ZONAS.map((z) => `${NOMBRE_ZONA_ATAQUE[z].toLowerCase()} ${porcentajeAR(at[z])}`).join(", ")}`} />
        </div>
        <div>
          <p className="mb-1 text-center text-sm">Hacia dónde le tiran a {nombres[receptor]}</p>
          <SiluetaZonas fracciones={ex} colorear={colorear} ancho={160} etiqueta={`Hacia dónde le tiran a ${nombres[receptor]}: ${ZONAS.map((z) => `${NOMBRE_ZONA_ATAQUE[z].toLowerCase()} ${porcentajeAR(ex[z])}`).join(", ")}`} />
        </div>
      </div>

      <Desplazable nombre="Tabla de ataque contra exposición">
        <table className="w-full min-w-[420px] text-sm">
          <caption className="sr-only">Ataque contra exposición, por zona y en los dos sentidos</caption>
          <thead>
            <tr>
              <th scope="col" className="rotulo py-1 text-left font-normal">Quién tira</th>
              <th scope="col" className="rotulo py-1 text-left font-normal">Zona</th>
              <th scope="col" className="rotulo py-1 text-right font-normal">Ataque</th>
              <th scope="col" className="rotulo py-1 text-right font-normal">Al otro le tiran</th>
            </tr>
          </thead>
          <tbody>
            {r.cruces.map((c) => (
              <tr key={`${c.atacante}-${c.zona}`} className="border-t border-borde-suave">
                <td className="py-1.5">{nombres[c.atacante]}</td>
                <th scope="row" className="py-1.5 text-left font-normal">{NOMBRE_ZONA_ATAQUE[c.zona]}</th>
                <td className="py-1.5 text-right"><Estimado className="dato">{porcentajeAR(c.ataque_fraccion)}</Estimado></td>
                <td className="py-1.5 text-right"><Estimado className="dato">{porcentajeAR(c.exposicion_fraccion)}</Estimado></td>
              </tr>
            ))}
          </tbody>
        </table>
      </Desplazable>

      {r.faltan ? (
        <p className="text-sm text-estimado">{textos.cruceFaltan(r.base.A, r.base.B, r.faltan.minimo)}</p>
      ) : r.lecturas.length === 0 ? (
        <p className="text-sm text-texto-2">{textos.cruceSinLecturas}</p>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {(["A", "B"] as const).map((l) => {
            const suyas = r.lecturas.filter((x) => x.para === l);
            return (
              <div key={l}>
                <p className="flex items-center gap-2 font-medium"><Esquina lado={l} />{nombres[l]}</p>
                <ul className="mt-2 space-y-2 text-sm">
                  {suyas.filter((x) => x.tipo === "oportunidad").map((x) => (
                    <li key={`${x.regla}-${x.zona}-${x.atacante}`} className="rounded-sm border-l-2 border-ok-borde pl-2"><Rotulo className="text-ok">oportunidad</Rotulo><span className="block">{x.texto}</span></li>
                  ))}
                  {suyas.filter((x) => x.tipo === "amenaza").map((x) => (
                    <li key={`${x.regla}-${x.zona}-${x.atacante}`} className="rounded-sm border-l-2 border-error-borde pl-2"><Rotulo className="text-error">amenaza</Rotulo><span className="block">{x.texto}</span></li>
                  ))}
                  {suyas.length === 0 && <li className="text-texto-3">Sin lecturas para este lado.</li>}
                </ul>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
