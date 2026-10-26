"use client";

/*
 * 05 · Fight-Card y 09 · Fight-Card en el celular (04 §05). El mismo contenido y los mismos
 * bloques condicionados en las dos (D5).
 *
 * Todo sale del documento ya cargado: cambiar de round no pide nada a la API (CA-05.2). El
 * texto del margen sale del documento (CA-05.3). Ninguna version muestra conectados, puntos
 * ni ganador (CA-05.4): el bloque no disponible de contacto se titula sin esa palabra.
 */
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/animate-ui/components/radix/accordion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/animate-ui/components/radix/tabs";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/animate-ui/components/radix/dropdown-menu";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { Download } from "@/components/animate-ui/icons/download";
import { Esquina } from "@/components/boxtwin/Esquina";
import { EstadoBloque, Estimado } from "@/components/boxtwin/EstadoBloque";
import { Margen } from "@/components/boxtwin/Margen";
import { Rotulo } from "@/components/boxtwin/Rotulo";
import { BarrasPorRound } from "@/components/boxtwin/graficos/BarrasPorRound";
import { mensajeDeError } from "@/components/boxtwin/Estados";
import { nombresDe } from "@/components/pantallas/sesion/PantallaSesion";
import { esIndividual, useCuenta } from "@/lib/cuenta";
import { pedir } from "@/lib/api/cliente";
import { leerToken } from "@/lib/sesion-usuario";
import { useProducto } from "@/lib/api/consultas";
import { TIPOS_GOLPE, type Sesion } from "@/lib/api/esquemas";
import { estadoDe } from "@/lib/banderas";
import {
  alturaDe, combinacionesDe, lados as ladosDe, ladosConGuardiaSupuesta, lecturaRounds, prefijoRound, recallDeCombinaciones, resumen,
  type Altura, type FightCard, type Lado, type Resumen,
} from "@/lib/fightcard";
import { decimal, entero, nombreDeTipo, porcentajeAR } from "@/lib/formato";
import { textos } from "@/lib/textos";
import { cn } from "@/lib/utils";

/* Fraccion de golpes con la guardia opuesta arriba, sobre los medibles. */
export function guardiaArriba(fc: FightCard, lado: Lado, round: number | null): number | null {
  const p = fc.peleadores[lado];
  if (!p) return null;
  const r = round != null ? p.por_round.find((x) => x.round === round) : null;
  const ev = (p.guardia ?? []).filter((e) => e.medible !== false && e.mano_opuesta_caida != null && (!r || (e.t >= r.inicio_s && e.t < r.fin_s)));
  if (!ev.length) return null;
  return ev.filter((e) => e.mano_opuesta_caida === false).length / ev.length;
}

export function VistaFightCard({ s, fc }: { s: Sesion; fc: FightCard }) {
  const router = useRouter();
  const q = useSearchParams();
  const producto = useProducto();
  const p = producto.data;
  const lados = ladosDe(fc);
  const cuenta = useCuenta().data;
  const nombres = nombresDe(s, fc, cuenta);
  const nombreDe: Record<Lado, string> = { A: nombres.A ?? "Peleador A", B: nombres.B ?? "Peleador B" };
  const conRounds = estadoDe("05.rounds", p) === "medido";
  const rondas = fc.video.rounds;
  const pedido = Number(q.get("round"));
  const round = conRounds && rondas.some((r) => r.round === pedido) ? pedido : null;

  const res = Object.fromEntries(lados.map((l) => [l, resumen(fc, l, round)])) as Record<Lado, Resumen | null>;
  const estGuardia = estadoDe("05.guardia", p);
  const detector = fc.detector;
  const clasif = fc.clasificador;

  function irRound(v: string) {
    const u = new URLSearchParams(q.toString());
    if (v === "sesion") u.delete("round");
    else u.set("round", v);
    router.replace(`/sesion/?${u.toString()}`);
  }

  const descartados = lados.reduce((a, l) => a + (res[l]?.descartados ?? 0), 0);
  const sinAsignar = fc.identidad.sin_asignar ?? 0;

  const cuerpo = (
    <>
      {/* 3. Cabecera A y B */}
      <div className={cn("grid gap-3", lados.length > 1 ? "grid-cols-2" : "grid-cols-1")}>
        {lados.map((l) => {
          const b = fc.sesion?.boxeadores?.[l];
          return (
            <div key={l} className="rounded-md border border-borde bg-superficie p-3">
              <p className="flex items-center gap-2 font-display text-lg font-semibold lg:text-xl"><Esquina lado={l} /> <span className="truncate">{nombreDe[l]}</span></p>
              <p className="mt-0.5 text-xs text-texto-3">
                {esIndividual(cuenta) && l === "B"
                  ? textos.individualCompañero
                  : [b?.categoria, b?.guardia].filter(Boolean).join(" · ") || "sin perfil asignado"}
              </p>
            </div>
          );
        })}
      </div>

      {/* 4. Filas medidas */}
      <EstadoBloque estado="medido" titulo={round == null ? "Volumen de la sesión" : `Volumen del round ${round}`}>
        <div className="flex flex-col gap-4">
          <FilaAB titulo="Golpes detectados" grande lados={lados} valores={(l) => res[l]?.detectados ?? null} />
          <FilaAB titulo="Izquierda" lados={lados} valores={(l) => res[l]?.izq ?? null} />
          <FilaAB titulo="Derecha" lados={lados} valores={(l) => res[l]?.der ?? null} />
          <FilaAB titulo="Golpes por minuto" lados={lados} valores={(l) => res[l]?.porMinuto ?? null} decimales={1} />
        </div>
      </EstadoBloque>

      {/* 4b. Combinaciones (W13, regla del 14-10): medido, por tiempo y brazo. Por round solo con C2, como el resto. */}
      <Combinaciones fc={fc} lados={lados} round={round} estado={estadoDe("05.combinaciones", p)} />

      {/* 5. Filas estimadas */}
      <EstadoBloque
        estado={estadoDe("05.distribucion", p)}
        titulo="Tipo de golpe"
        nota={[textos.estimacionTipo(clasif.checkpoint, clasif.exactitud_familia_fuente_no_vista),
          textos.modoAcademico(clasif.metodo, clasif.licencia)].filter(Boolean).join(" ")}
      >
        <div className="flex flex-col gap-4">
          {ladosConGuardiaSupuesta(fc).map((l) => (
            <p key={l} role="note" data-aviso-guardia={l} className="rounded-md border border-estimado-borde bg-superficie-2 p-2 text-xs text-texto-2">
              {textos.avisoGuardiaSupuesta(nombres[l] ?? l)}
            </p>
          ))}
          {TIPOS_GOLPE.map((t) => (
            <FilaAB
              key={t}
              titulo={nombreDeTipo(t, fc.nomenclatura)!}
              aclaracion={t === "hook" ? "tipo menos confiable" : undefined}
              lados={lados}
              estimado
              valores={(l) => res[l]?.tipos[t] ?? null}
            />
          ))}
          {lados.some((l) => (res[l]?.sinClasificar ?? 0) > 0) && (
            <p className="text-xs text-texto-3">
              Sin clasificar: {lados.map((l) => `${nombreDe[l]} ${res[l]?.sinClasificar ?? 0}`).join(" · ")}
            </p>
          )}
        </div>
      </EstadoBloque>

      {/* 5b. A la cabeza y al cuerpo (W14). Estimacion por decision de producto (14-10), con
          lo medido en C8 en la nota. Apagada, va al final como no disponible. */}
      {estadoDe("05.altura", p) === "estimacion" && (
        <AlturaAB fc={fc} lados={lados} round={round} nombreDe={nombreDe} />
      )}

      {/* 6. Guardia opuesta arriba */}
      <EstadoBloque estado={estGuardia} titulo="Guardia opuesta arriba" motivo={textos.condicionadoC3}>
        <FilaAB titulo="Golpes con la otra mano arriba" lados={lados} valores={(l) => guardiaArriba(fc, l, round)} porcentaje />
      </EstadoBloque>

      {/* 7. Golpes por round */}
      {rondas.length > 1 && (
        <EstadoBloque
          estado={estadoDe("05.rounds", p)}
          titulo="Golpes por round"
          motivo={textos.condicionadoC2}
          accion={p?.criterios.C2.estado !== "aprobado" ? <Rotulo className="text-estimado">{textos.roundsSinValidar}</Rotulo> : undefined}
        >
          {fc.video.rounds_origen === "segmentos_180s" && <p className="mb-2 text-xs text-texto-3">{textos.roundsSegmentos}</p>}
          <BarrasPorRound fc={fc} nombres={nombreDe} />
          <p className="mt-2 text-sm text-texto-2">
            {lados.map((l) => lecturaRounds(fc.peleadores[l]!.por_round, nombreDe[l])).filter(Boolean).join(". ")}
          </p>
        </EstadoBloque>
      )}

      {/* 8. Descartados y sin identidad */}
      {(descartados > 0 || sinAsignar > 0) && (
        <section className="rounded-md border border-borde bg-superficie p-4">
          {descartados > 0 && (
            <div data-fila className="flex items-center justify-between gap-3">
              <span>Descartados por el entrenador</span>
              <span className="flex gap-4 dato">
                {lados.map((l) => <span key={l} className="inline-flex items-center gap-1"><Esquina lado={l} tamano={16} />{res[l]?.descartados ?? 0}</span>)}
              </span>
            </div>
          )}
          {sinAsignar > 0 && (
            <p className={cn("text-sm text-texto-2", descartados > 0 && "mt-3")}>
              {textos.sinAsignar(sinAsignar)}{" "}
              {fc.sinRevisionDeTramos ? (
                <span className="block mt-1 text-texto-3">{textos.tramosVersionVieja}</span>
              ) : (
                <Link href={`/sesion/identificar/?id=${s.id}`} className="underline underline-offset-4">Revisar tramos</Link>
              )}
            </p>
          )}
        </section>
      )}

    </>
  );

  return (
    <div className="flex flex-col gap-4">
      {/* 1. Margen arriba de todo */}
      <div>
        <Margen precision={detector.precision_medida} recall={detector.recall_medido} medidoEn={detector.medido_en} contexto={fc.sesion?.contexto} />
        <Accordion type="single" collapsible className="mt-1">
          <AccordionItem value="leer" className="border-0">
            <AccordionTrigger className="min-h-11 py-2 text-sm text-texto-2">Cómo leer esta tarjeta</AccordionTrigger>
            <AccordionContent className="space-y-2 text-sm text-texto-2">
              {detector.medido_en && fc.sesion?.contexto && detector.medido_en !== fc.sesion.contexto ? (
                <p>Los números son golpes que el sistema detectó, no todos los que se tiraron. Cuántos se le escapan se midió en {detector.medido_en}, no en este contexto.</p>
              ) : (
                <p>Los números son golpes que el sistema detectó, no todos los que se tiraron. El detector se midió sobre material que no había visto: de cada 10 golpes que marca, acierta {Math.round(detector.precision_medida * 10)}; de cada 10 golpes que hubo, encuentra {Math.round(detector.recall_medido * 10)}.</p>
              )}
              <p>Por eso lo que más vale es comparar adentro de la sesión: un peleador contra el otro, un round contra otro.</p>
              <p>El tipo de golpe es una estimación del modelo y se muestra con ≈ y borde punteado. No hay puntaje de rounds ni resultado del combate.</p>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>

      {/* 2. Selector de round. Con C2 los rounds son pestanas y lo que sigue es su panel. */}
      {conRounds && rondas.length > 1 ? (
        <Tabs value={round == null ? "sesion" : String(round)} onValueChange={irRound}>
          <div className="-mx-4 mb-4 overflow-x-auto px-4">
            <TabsList className="w-max">
              <TabsTrigger value="sesion" className="min-h-11 px-4">Sesión</TabsTrigger>
              {rondas.map((r) => (
                <TabsTrigger key={r.round} value={String(r.round)} className="min-h-11 px-4">{prefijoRound(fc)}{r.round}</TabsTrigger>
              ))}
            </TabsList>
          </div>
          <TabsContent value={round == null ? "sesion" : String(round)} transition={{ duration: 0.15, ease: "easeOut" }} className="flex flex-col gap-4">
            {cuerpo}
          </TabsContent>
        </Tabs>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-3">
            <span className="inline-flex min-h-11 items-center rounded-md border border-borde bg-elevado px-4 text-sm">Sesión</span>
            {rondas.length > 1 && <Rotulo>comparación entre rounds en validación</Rotulo>}
          </div>
          {cuerpo}
        </>
      )}

      {/* 9. No disponible en esta version */}
      <EstadoBloque estado={estadoDe("05.contacto", p)} titulo="Si el golpe tocó, fue bloqueado o esquivado" motivo={textos.nodispContacto} />
      {estadoDe("05.altura", p) !== "estimacion" && (
        <EstadoBloque estado={estadoDe("05.altura", p)} titulo="A la cabeza y al cuerpo" motivo={textos.nodispAltura} />
      )}

      {/* 10. Exportar */}
      <Exportar s={s} />
    </div>
  );
}

function Combinaciones({ fc, lados, round, estado }: { fc: FightCard; lados: Lado[]; round: number | null; estado: ReturnType<typeof estadoDe> }) {
  const valores = Object.fromEntries(lados.map((l) => [l, combinacionesDe(fc, l, round)])) as Record<Lado, number | null>;
  const sinDato = lados.every((l) => valores[l] == null);
  const ninguna = !sinDato && lados.every((l) => valores[l] === 0);
  return (
    <EstadoBloque estado={estado} titulo="Combinaciones">
      {sinDato ? (
        <p className="text-sm text-texto-3">{textos.combinacionesVersionVieja}</p>
      ) : (
        <div className="flex flex-col gap-3">
          {ninguna ? (
            <p className="text-sm text-texto-2">{textos.combinacionesVacio}</p>
          ) : (
            <FilaAB titulo={textos.combinacionesRegla} lados={lados} valores={(l) => valores[l]} />
          )}
          <p className="rounded-md border-l-2 border-estimado-borde bg-estimado-fondo px-3 py-2 text-sm text-texto-2">
            {textos.margenCombinaciones(recallDeCombinaciones(fc))}
          </p>
        </div>
      )}
    </EstadoBloque>
  );
}

function AlturaAB({ fc, lados, round, nombreDe }: { fc: FightCard; lados: Lado[]; round: number | null; nombreDe: Record<Lado, string> }) {
  const valores = Object.fromEntries(lados.map((l) => [l, alturaDe(fc, l, round)])) as Record<Lado, Altura | null>;
  const nota = textos.estimacionAltura(fc.metodos?.objetivo?.medido);
  if (lados.every((l) => valores[l] == null)) {
    return (
      <EstadoBloque estado="estimacion" titulo="A la cabeza y al cuerpo" nota={nota}>
        <p className="text-sm text-texto-3">{textos.alturaVersionVieja}</p>
      </EstadoBloque>
    );
  }
  const sinDato = lados.reduce((a, l) => a + (valores[l]?.sinDato ?? 0), 0);
  return (
    <EstadoBloque estado="estimacion" titulo="A la cabeza y al cuerpo" nota={nota}>
      <div className={cn("grid gap-3", lados.length > 1 ? "grid-cols-2" : "grid-cols-1")} data-fila>
        {lados.map((l, i) => {
          const v = valores[l];
          if (!v) return <span key={l} className="text-texto-3">—</span>;
          const conDato = v.cabeza + v.cuerpo;
          const fCabeza = conDato ? v.cabeza / conDato : 0;
          return (
            <div key={l} className={cn("flex flex-col gap-1.5", i === 1 && "items-end text-right")} aria-label={`${nombreDe[l]}: ${v.cabeza} a la cabeza, ${v.cuerpo} al cuerpo`}>
              <div className={cn("flex gap-4", i === 1 && "flex-row-reverse")}>
                <span className="flex flex-col">
                  <span className="text-xs text-texto-3">cabeza</span>
                  <Estimado><span className="font-display text-2xl font-semibold leading-none">{entero(v.cabeza)}</span></Estimado>
                  <span className="dato text-xs text-texto-3">{conDato ? porcentajeAR(fCabeza) : "—"}</span>
                </span>
                <span className="flex flex-col">
                  <span className="text-xs text-texto-3">cuerpo</span>
                  <Estimado><span className="font-display text-2xl font-semibold leading-none">{entero(v.cuerpo)}</span></Estimado>
                  <span className="dato text-xs text-texto-3">{conDato ? porcentajeAR(1 - fCabeza) : "—"}</span>
                </span>
              </div>
              {/* Barra partida en dos tonos: cabeza llena, cuerpo atenuado. */}
              <div aria-hidden="true" className={cn("flex h-1.5 w-full overflow-hidden rounded-full bg-superficie-2", i === 1 && "flex-row-reverse")}>
                <div className={cn("h-full", l === "A" ? "bg-a-trazo" : "bg-b-trazo")} style={{ width: `${Math.round(fCabeza * 100)}%` }} />
                <div className={cn("h-full opacity-40", l === "A" ? "bg-a-trazo" : "bg-b-trazo")} style={{ width: `${conDato ? Math.round((1 - fCabeza) * 100) : 0}%` }} />
              </div>
            </div>
          );
        })}
      </div>
      {sinDato > 0 && (
        <p className="mt-3 text-xs text-texto-3">
          {textos.alturaSinDato(lados.length > 1 ? lados.map((l) => `${nombreDe[l]} ${valores[l]?.sinDato ?? 0}`).join(" · ") : String(sinDato))}
        </p>
      )}
    </EstadoBloque>
  );
}

function FilaAB({
  titulo, lados, valores, grande = false, estimado = false, decimales = 0, porcentaje = false, aclaracion,
}: {
  titulo: string;
  lados: Lado[];
  valores: (l: Lado) => number | null;
  grande?: boolean;
  estimado?: boolean;
  decimales?: number;
  porcentaje?: boolean;
  aclaracion?: string;
}) {
  const vs = lados.map((l) => valores(l));
  const max = Math.max(...vs.map((v) => v ?? 0), porcentaje ? 1 : 1e-9);
  const fmt = (v: number | null) => (v == null ? "—" : porcentaje ? porcentajeAR(v) : decimales ? decimal(v, decimales) : entero(v));
  return (
    <div data-fila>
      <p className="mb-1 flex items-baseline gap-2 text-sm">
        <span>{titulo}</span>
        {aclaracion && <span className="text-xs text-estimado">{aclaracion}</span>}
      </p>
      <div className={cn("grid gap-3", lados.length > 1 ? "grid-cols-2" : "grid-cols-1")}>
        {lados.map((l, i) => {
          const v = vs[i];
          const num = (
            <span className={cn("font-display font-semibold leading-none", grande ? "text-[34px] lg:text-[42px]" : "text-2xl")}>{fmt(v)}</span>
          );
          return (
            <div key={l} className={cn("flex flex-col gap-1", i === 1 && "items-end text-right")}>
              {estimado && v != null ? <Estimado>{num}</Estimado> : num}
              <div aria-hidden="true" className={cn("h-1.5 w-full rounded-full bg-superficie-2", i === 1 && "flex justify-end")}>
                <div
                  className={cn("h-full rounded-full", l === "A" ? "bg-a-trazo" : "bg-b-trazo", estimado && "opacity-70")}
                  style={{ width: `${Math.round(((v ?? 0) / max) * 100)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/*
 * Exportar (04 §05, punto 10). El export pide el token, que un enlace no manda: se pide con
 * fetch y se entrega como blob. La version para imprimir se abre en una pestana nueva y se
 * imprime desde el navegador, como hoy.
 */
async function bajar(ruta: string): Promise<Blob> {
  const r = await fetch(ruta, { headers: { Authorization: `Bearer ${leerToken()}` } });
  if (!r.ok) throw new Error(`no se pudo exportar (error ${r.status})`);
  return r.blob();
}

function guardarArchivo(blob: Blob, nombre: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = nombre;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export function Exportar({ s }: { s: Sesion }) {
  const base = `/fightcards/${s.id}/export`;

  async function imprimir() {
    const ventana = window.open("", "_blank");
    try {
      const blob = await bajar(`${base}?formato=pdf`);
      const url = URL.createObjectURL(new Blob([blob], { type: "text/html" }));
      if (ventana) ventana.location.href = url;
      else window.location.assign(url);
    } catch (e) {
      ventana?.close();
      toast.error(mensajeDeError(e));
    }
  }

  async function csv() {
    try {
      guardarArchivo(await bajar(`${base}?formato=csv`), `${s.id}.csv`);
    } catch (e) {
      toast.error(mensajeDeError(e));
    }
  }

  async function json() {
    try {
      const datos = await pedir(`/fightcards/${s.id}`);
      guardarArchivo(new Blob([JSON.stringify(datos, null, 2)], { type: "application/json" }), `${s.id}.fightcard.json`);
    } catch (e) {
      toast.error(mensajeDeError(e));
    }
  }

  /* navigator.share con el CSV como archivo cuando se puede; si no, se descarga. */
  async function compartir() {
    try {
      const blob = await bajar(`${base}?formato=csv`);
      const archivo = new File([blob], `${s.nombre || s.id}.csv`, { type: "text/csv" });
      if (navigator.canShare?.({ files: [archivo] })) {
        await navigator.share({ files: [archivo], title: `Fight-Card · ${s.nombre}` });
      } else {
        guardarArchivo(blob, archivo.name);
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      toast.error(mensajeDeError(e));
    }
  }

  return (
    <div className="flex flex-wrap gap-3">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" className="min-h-11"><Download size={16} /> Exportar</Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="bg-superficie">
          <DropdownMenuItem className="min-h-11" onSelect={imprimir}>Para imprimir (PDF)</DropdownMenuItem>
          <DropdownMenuItem className="min-h-11" onSelect={csv}>Lista de golpes (CSV)</DropdownMenuItem>
          <DropdownMenuItem className="min-h-11" onSelect={json}>Datos completos (JSON)</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <Button asChild variant="outline" className="min-h-11 lg:hidden">
        <Link href={`/sesion/?id=${s.id}&vista=video`}>Ver video</Link>
      </Button>
      <Button variant="outline" className="min-h-11 lg:hidden" onClick={compartir}>Compartir</Button>
    </div>
  );
}

