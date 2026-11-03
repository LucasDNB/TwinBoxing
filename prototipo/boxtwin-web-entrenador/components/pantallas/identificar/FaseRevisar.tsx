"use client";

/*
 * 03 · Identificacion, fase 2: revisar tramos (04 §03; B3; W9).
 *
 * Primero el resultado, reducido a un numero (3-11): cuantos clips quedan sin asignar. Lo que
 * salio de cuadro ya no llega como pregunta: el anotador lo ignora solo. Despues la lista de
 * tramos dudosos con su motivo, su recorte y los botones con el nombre de cada boxeador (o
 * Rincon rojo / Rincon azul si la sesion no los tiene) e Ignorar.
 * Tocar el rango abre el video en ese instante: en hoja en el celular, en linea en
 * escritorio. "Aplicar" manda todas las decisiones y la sesion vuelve a completar.
 *
 * Identidad estable (docs/specs/2026-11-02-identidad-estable): la lista trae primero los
 * tramos ASIGNADOS que se sospecha que estan mal, con lo que dicen hoy; con el video abierto,
 * "Acá cambia de persona" parte el tramo en ese instante y cada mitad lleva su decision. Con
 * "Ver también lo asignado" se puede corregir cualquier tramo.
 */
import { useRef, useState } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/animate-ui/components/radix/toggle-group";
import { esIndividual as esCuentaIndividual, useCuenta } from "@/lib/cuenta";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/animate-ui/components/radix/sheet";
import { AutoHeight } from "@/components/animate-ui/primitives/effects/auto-height";
import { ErrorConReintento, Esqueleto, mensajeDeError } from "@/components/boxtwin/Estados";
import { Rotulo } from "@/components/boxtwin/Rotulo";
import { Recorte } from "@/components/pantallas/identificar/PantallaIdentificar";
import { urlDeVideo } from "@/components/pantallas/sesion/VistaVideo";
import { ErrorApi } from "@/lib/api/cliente";
import { type AsignacionTramo, useAsignarTramos, useTicket, useTramos } from "@/lib/api/consultas";
import type { Sesion, Tramos } from "@/lib/api/esquemas";
import type { FightCard } from "@/lib/fightcard";
import { decimal, mmss, nombreCorto } from "@/lib/formato";
import { useEsEscritorio } from "@/lib/medios";
import { textos } from "@/lib/textos";


export function FaseRevisar({ s, fc }: { s: Sesion; fc: FightCard }) {
  const [todos, setTodos] = useState(false);
  const tramos = useTramos(s.id, !fc.sinRevisionDeTramos, todos);
  const ident = fc.identidad;
  // Una Fight-Card de antes del 3-11 no trae el campo: se cuenta de la lista.
  const clipsSinAsignar = ident.tramos_sin_asignar
    ?? tramos.data?.tramos.filter((t) => sinAsignar(t) && !saleDeCuadro(t) && !t.asignado
      && t.revision_pendiente !== false).length
    ?? ident.tracks_sin_asignar ?? 0;

  return (
    <div className="flex flex-col gap-4">
      <section aria-labelledby="propagacion" className="rounded-md border border-borde bg-superficie p-4">
        <h2 id="propagacion" className="font-display text-xl font-semibold lg:text-2xl">Resultado</h2>
        <p className="mt-2 text-[15px]">{textos.clipsSinAsignar(clipsSinAsignar)}</p>
      </section>

      {fc.sinRevisionDeTramos ? (
        <p className="rounded-md border border-borde bg-superficie-2 p-4 text-sm text-texto-2">{textos.tramosVersionVieja}</p>
      ) : tramos.isPending ? (
        <Esqueleto filas={2} />
      ) : tramos.isError ? (
        tramos.error instanceof ErrorApi && tramos.error.estado === 409 ? (
          <p className="rounded-md border border-borde bg-superficie-2 p-4 text-sm text-texto-2">{tramos.error.message}</p>
        ) : (
          <ErrorConReintento error={tramos.error} alReintentar={() => tramos.refetch()} />
        )
      ) : (
        <Revision key={todos ? "todos" : "dudosos"} s={s} datos={tramos.data} todos={todos} setTodos={setTodos} />
      )}

      <Button asChild variant="outline" className="min-h-11 w-fit">
        <Link href={`/sesion/?id=${s.id}&vista=fightcard`}>Ver resultados</Link>
      </Button>
    </div>
  );
}

type Rol = "A" | "B" | "ninguno";
type Tramo = Tramos["tramos"][number];
/* Una decision: un rol para todo el tramo, o un corte con un rol a cada lado. */
type Decision = { rol: Rol | "" } | { corte: number; antes: Rol | ""; despues: Rol | "" };

const claveDe = (t: Tramo) => t.clave ?? String(t.track);
// Las sesiones de antes del 2-11 no traen tipo: todo lo que listaban era sin asignar.
const sinAsignar = (t: Tramo) => (t.tipo ?? "sin_asignar") === "sin_asignar";
/* 3-11: lo que salio de cuadro se ignora solo. Las sesiones nuevas ya lo traen ignorado; en las
 * de antes sigue sin asignar y aca no se pregunta. */
const saleDeCuadro = (t: Tramo) => t.motivo === "corto" && sinAsignar(t);
const esTrackEntero = (t: Tramo) => claveDe(t) === String(t.track);

function inicialDe(t: Tramo): Decision {
  const p = t.partes ?? [];
  if (p.length === 2) return { corte: p[0].t_hasta, antes: p[0].rol, despues: p[1].rol };
  return { rol: t.asignado ?? "" };
}

function completa(d: Decision): boolean {
  return "rol" in d ? d.rol !== "" : d.antes !== "" && d.despues !== "";
}

function iguales(a: Decision, b: Decision): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/* Lo que se manda por cada tramo. Un track entero sin cortar va como antes, sin rango. */
export function asignacionesDe(t: Tramo, d: Decision): AsignacionTramo[] {
  if ("rol" in d) {
    if (d.rol === "") return [];
    return esTrackEntero(t)
      ? [{ track: t.track, rol: d.rol }]
      : [{ track: t.track, rol: d.rol, desde: t.t_desde, hasta: t.t_hasta }];
  }
  if (d.antes === "" || d.despues === "") return [];
  return [
    { track: t.track, rol: d.antes, desde: t.t_desde, hasta: d.corte },
    { track: t.track, rol: d.despues, desde: d.corte, hasta: t.t_hasta },
  ];
}

function Revision({ s, datos: crudos, todos, setTodos }: {
  s: Sesion; datos: Tramos; todos: boolean; setTodos: (v: boolean) => void;
}) {
  const datos = { ...crudos, tramos: crudos.tramos.filter((t) => !saleDeCuadro(t)) };
  const aplicar = useAsignarTramos(s.id);
  const ticket = useTicket(s.id);
  const escritorio = useEsEscritorio();
  const inicial = Object.fromEntries(datos.tramos.map((t) => [claveDe(t), inicialDe(t)])) as Record<string, Decision>;
  const [decisiones, setDecisiones] = useState<Record<string, Decision>>(inicial);
  const [viendo, setViendo] = useState<Tramo | null>(null);
  const cambios = datos.tramos.filter((t) => {
    const d = decisiones[claveDe(t)];
    return d && completa(d) && !iguales(d, inicial[claveDe(t)]);
  });
  // W19: en bolsa y sombra hay una sola persona; el tramo es ella o se ignora.
  const individual = s.contexto === "bolsa" || s.contexto === "sombra";
  // Versiones 26-10 (03 §I-03): en la version Individual la revision le habla al boxeador.
  const vos = esCuentaIndividual(useCuenta().data);
  // 3-11: el nombre del boxeador asignado; si la sesion no lo tiene, el rincon.
  const nombreDe = (r: "A" | "B") => {
    const n = s.boxeadores?.[r]?.nombre;
    return n ? nombreCorto(n) : textos.rincon[r];
  };
  const nombreRol = (r: string) =>
    r === "ninguno" ? "ignorado" : vos ? (r === "A" ? "soy yo" : "mi compañero") : individual ? "es él"
      : r === "A" || r === "B" ? nombreDe(r) : r;

  function decidir(t: Tramo, d: Decision) {
    setDecisiones((x) => ({ ...x, [claveDe(t)]: d }));
  }

  function partir(t: Tramo, segundo: number) {
    // El corte cae adentro del tramo, con un margen para que ninguna mitad quede vacia.
    const corte = Math.min(Math.max(segundo, t.t_desde + 0.1), t.t_hasta - 0.1);
    const previo = decisiones[claveDe(t)];
    const rol = previo && "rol" in previo ? previo.rol : "";
    decidir(t, { corte: Math.round(corte * 100) / 100, antes: rol, despues: "" });
  }

  async function enviar() {
    try {
      await aplicar.mutateAsync(cambios.flatMap((t) => asignacionesDe(t, decisiones[claveDe(t)])));
      toast.success("Listo. El análisis se completa de nuevo con lo que marcaste.");
    } catch (e) {
      toast.error(mensajeDeError(e));
    }
  }

  const casillaTodos = (
    <label className="flex min-h-11 items-center gap-2 text-sm text-texto-2">
      <input type="checkbox" checked={todos} onChange={(e) => setTodos(e.target.checked)} className="size-4" />
      Ver también lo asignado
    </label>
  );

  if (datos.tramos.length === 0) {
    return (
      <div className="flex flex-col gap-2">
        <p className="text-sm text-texto-2">No quedaron tramos de más de un segundo para revisar.</p>
        {casillaTodos}
      </div>
    );
  }

  const botones = (valor: Rol | "", alCambiar: (v: Rol | "") => void, etiqueta: string) => (
    <ToggleGroup
      type="single"
      value={valor}
      onValueChange={(v) => alCambiar(v as Rol | "")}
      aria-label={etiqueta}
      className={`mt-2 grid w-full gap-1 ${individual ? "grid-cols-2" : "grid-cols-3"}`}
    >
      {individual ? (
        <ToggleGroupItem value="A" className="min-h-11 border border-borde">{vos ? "Soy yo" : "Es él"}</ToggleGroupItem>
      ) : vos ? (
        <>
          <ToggleGroupItem value="A" className="min-h-11 border border-borde">Soy yo</ToggleGroupItem>
          <ToggleGroupItem value="B" className="min-h-11 border border-borde">Es mi compañero</ToggleGroupItem>
        </>
      ) : (
        <>
          <ToggleGroupItem value="A" className="min-h-11 min-w-0 border border-borde px-2"><span className="truncate">{nombreDe("A")}</span></ToggleGroupItem>
          <ToggleGroupItem value="B" className="min-h-11 min-w-0 border border-borde px-2"><span className="truncate">{nombreDe("B")}</span></ToggleGroupItem>
        </>
      )}
      <ToggleGroupItem value="ninguno" className="min-h-11 border border-borde">Ignorar</ToggleGroupItem>
    </ToggleGroup>
  );

  return (
    <section aria-labelledby="revisar">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 id="revisar" className="font-display text-xl font-semibold lg:text-2xl">Revisar tramos dudosos</h2>
        {casillaTodos}
      </div>
      <ul className="flex flex-col gap-3">
        {datos.tramos.map((t) => {
          const k = claveDe(t);
          const d = decisiones[k] ?? { rol: "" };
          return (
            <li key={k} className="rounded-md border border-borde bg-superficie p-3">
              <div className="flex gap-3">
                <Recorte c={t} ticket={ticket.data?.ticket} />
                <div className="min-w-0 flex-1">
                  <button
                    type="button"
                    className="dato min-h-11 text-left underline underline-offset-4"
                    onClick={() => setViendo(viendo && claveDe(viendo) === k ? null : t)}
                    aria-expanded={!!viendo && claveDe(viendo) === k}
                  >
                    {mmss(t.t_desde)} a {mmss(t.t_hasta)}
                  </button>
                  {t.motivo && <p className="text-sm text-texto-2">{textos.motivoTramo[t.motivo]}</p>}
                  <p className="text-xs text-texto-3">{decimal(t.t_hasta - t.t_desde, 1)} s</p>
                  {t.tipo === "sospechoso" && t.rol_actual && (
                    <Rotulo className="text-estimado">{textos.rolActual(nombreRol(t.rol_actual))}</Rotulo>
                  )}
                  {t.tipo === "asignado" && t.rol_actual && (
                    <Rotulo>{textos.rolActual(nombreRol(t.rol_actual))}</Rotulo>
                  )}
                  {t.asignado && <Rotulo className="text-ok">asignado a mano: {nombreRol(t.asignado)}</Rotulo>}
                  {(t.partes?.length ?? 0) > 0 && (
                    <Rotulo className="text-ok">
                      partido a mano: {t.partes!.map((p) => `${mmss(p.t_desde)} ${nombreRol(p.rol)}`).join(", ")}
                    </Rotulo>
                  )}
                </div>
              </div>
              {escritorio && (
                <AutoHeight>
                  {viendo && claveDe(viendo) === k && ticket.data && (
                    <VideoEn s={s} ticket={ticket.data.ticket} t={t.t_desde} alPartir={individual ? undefined : (seg) => partir(t, seg)} />
                  )}
                </AutoHeight>
              )}
              {"rol" in d ? (
                botones(d.rol, (v) => decidir(t, { rol: v }), `Quién es en el tramo de ${mmss(t.t_desde)}`)
              ) : (
                <div className="mt-2 flex flex-col gap-1">
                  <p className="text-sm">Antes de {mmss(d.corte)}</p>
                  {botones(d.antes, (v) => decidir(t, { ...d, antes: v }), `Quién es antes de ${mmss(d.corte)}`)}
                  <p className="mt-2 text-sm">Desde {mmss(d.corte)}</p>
                  {botones(d.despues, (v) => decidir(t, { ...d, despues: v }), `Quién es desde ${mmss(d.corte)}`)}
                  <Button variant="outline" className="mt-2 min-h-11 w-fit" onClick={() => decidir(t, { rol: d.antes })}>
                    Deshacer el corte
                  </Button>
                </div>
              )}
            </li>
          );
        })}
      </ul>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <Button className="min-h-11" disabled={cambios.length === 0 || aplicar.isPending} onClick={enviar}>
          Aplicar{cambios.length ? ` (${cambios.length})` : ""}
        </Button>
      </div>

      {!escritorio && (
        <Sheet open={!!viendo} onOpenChange={(o) => !o && setViendo(null)}>
          <SheetContent side="bottom" className="h-auto bg-superficie pb-[env(safe-area-inset-bottom)]">
            <SheetHeader>
              <SheetTitle className="font-display text-2xl">{viendo ? `${mmss(viendo.t_desde)} a ${mmss(viendo.t_hasta)}` : ""}</SheetTitle>
              <SheetDescription className="text-texto-3">{viendo?.motivo ? textos.motivoTramo[viendo.motivo] : ""}</SheetDescription>
            </SheetHeader>
            <div className="px-4 pb-6">
              {viendo && ticket.data && (
                <VideoEn
                  s={s}
                  ticket={ticket.data.ticket}
                  t={viendo.t_desde}
                  alPartir={individual ? undefined : (seg) => { partir(viendo, seg); setViendo(null); }}
                />
              )}
            </div>
          </SheetContent>
        </Sheet>
      )}
    </section>
  );
}

function VideoEn({ s, ticket, t, alPartir }: {
  s: Sesion; ticket: string; t: number; alPartir?: (segundo: number) => void;
}) {
  const ref = useRef<HTMLVideoElement>(null);
  return (
    <div className="flex flex-col gap-2">
      <video
        ref={ref}
        src={urlDeVideo(s.id, ticket, false)}
        controls
        playsInline
        preload="metadata"
        aria-label={`Video desde ${mmss(t)}`}
        className="mt-3 aspect-video w-full rounded-md border border-borde bg-black"
        onLoadedMetadata={(e) => { e.currentTarget.currentTime = t; }}
      />
      {alPartir && (
        <Button
          variant="outline"
          className="min-h-11 w-fit"
          onClick={() => alPartir(ref.current?.currentTime ?? t)}
        >
          Acá cambia de persona
        </Button>
      )}
    </div>
  );
}
