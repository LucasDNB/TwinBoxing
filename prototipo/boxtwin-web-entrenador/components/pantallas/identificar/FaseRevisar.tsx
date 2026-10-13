"use client";

/*
 * 03 · Identificacion, fase 2: revisar tramos (04 §03; B3; W9).
 *
 * Primero el resultado de la propagacion, dicho como es (D6): cuantos tramos asigno solo,
 * la separacion de color contra su minimo, y que los golpes de los tramos sin asignar no
 * se cuentan. Despues la lista de tramos dudosos con su motivo, su recorte y A / B / Ignorar.
 * Tocar el rango abre el video en ese instante: en hoja en el celular, en linea en
 * escritorio. "Aplicar" manda todas las decisiones y la sesion vuelve a completar.
 */
import { useState } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/animate-ui/components/radix/toggle-group";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/animate-ui/components/radix/sheet";
import { AutoHeight } from "@/components/animate-ui/primitives/effects/auto-height";
import { ErrorConReintento, Esqueleto, mensajeDeError } from "@/components/boxtwin/Estados";
import { Rotulo } from "@/components/boxtwin/Rotulo";
import { Recorte } from "@/components/pantallas/identificar/PantallaIdentificar";
import { urlDeVideo } from "@/components/pantallas/sesion/VistaVideo";
import { ErrorApi } from "@/lib/api/cliente";
import { useAsignarTramos, useTicket, useTramos } from "@/lib/api/consultas";
import type { Sesion, Tramos } from "@/lib/api/esquemas";
import type { FightCard } from "@/lib/fightcard";
import { decimal, mmss } from "@/lib/formato";
import { useEsEscritorio } from "@/lib/medios";
import { textos } from "@/lib/textos";

type Rol = "A" | "B" | "ninguno";
type Tramo = Tramos["tramos"][number];

export function FaseRevisar({ s, fc }: { s: Sesion; fc: FightCard }) {
  const tramos = useTramos(s.id, !fc.sinRevisionDeTramos);
  const ident = fc.identidad;

  return (
    <div className="flex flex-col gap-4">
      <section aria-labelledby="propagacion" className="rounded-md border border-borde bg-superficie p-4">
        <h2 id="propagacion" className="font-display text-xl font-semibold lg:text-2xl">Resultado</h2>
        <ul className="mt-2 space-y-1 text-[15px]">
          {ident.tramos_total != null && ident.tramos_asignados_auto != null && (
            <li>{ident.tramos_asignados_auto} de {ident.tramos_total} tramos asignados solos
              {ident.tramos_asignados_manual ? `, ${ident.tramos_asignados_manual} a mano` : ""}</li>
          )}
          {ident.separacion_color != null && ident.piso_de_separacion != null && (
            <li className={ident.separacion_color < ident.piso_de_separacion ? "text-estimado" : undefined}>
              {textos.separacion(ident.separacion_color, ident.piso_de_separacion)}
            </li>
          )}
          {(ident.sin_asignar ?? 0) > 0 && <li>{textos.sinAsignar(ident.sin_asignar!)}</li>}
        </ul>
        {(ident.sin_asignar ?? 0) > 0 && (
          <p className="mt-2 text-sm text-texto-2">Los golpes que caen en tramos sin asignar no se cuentan. Revisalos para no perder conteo.</p>
        )}
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
        <Revision s={s} datos={tramos.data} />
      )}

      <Button asChild variant="outline" className="min-h-11 w-fit">
        <Link href={`/sesion/?id=${s.id}&vista=fightcard`}>Ver resultados</Link>
      </Button>
    </div>
  );
}

function Revision({ s, datos }: { s: Sesion; datos: Tramos }) {
  const aplicar = useAsignarTramos(s.id);
  const ticket = useTicket(s.id);
  const escritorio = useEsEscritorio();
  const inicial = Object.fromEntries(datos.tramos.map((t) => [t.track, t.asignado ?? ""])) as Record<number, Rol | "">;
  const [decisiones, setDecisiones] = useState<Record<number, Rol | "">>(inicial);
  const [viendo, setViendo] = useState<Tramo | null>(null);
  const cambios = datos.tramos.filter((t) => decisiones[t.track] && decisiones[t.track] !== (t.asignado ?? ""));

  async function enviar() {
    try {
      await aplicar.mutateAsync(cambios.map((t) => ({ track: t.track, rol: decisiones[t.track] as Rol })));
      toast.success("Listo. El análisis se completa de nuevo con lo que marcaste.");
    } catch (e) {
      toast.error(mensajeDeError(e));
    }
  }

  if (datos.tramos.length === 0) {
    return <p className="text-sm text-texto-2">No quedaron tramos de más de un segundo para revisar.</p>;
  }

  return (
    <section aria-labelledby="revisar">
      <h2 id="revisar" className="mb-3 font-display text-xl font-semibold lg:text-2xl">Revisar tramos dudosos</h2>
      <ul className="flex flex-col gap-3">
        {datos.tramos.map((t) => (
          <li key={t.track} className="rounded-md border border-borde bg-superficie p-3">
            <div className="flex gap-3">
              <Recorte c={t} ticket={ticket.data?.ticket} />
              <div className="min-w-0 flex-1">
                <button
                  type="button"
                  className="dato min-h-11 text-left underline underline-offset-4"
                  onClick={() => setViendo(viendo?.track === t.track ? null : t)}
                  aria-expanded={viendo?.track === t.track}
                >
                  {mmss(t.t_desde)} a {mmss(t.t_hasta)}
                </button>
                <p className="text-sm text-texto-2">{textos.motivoTramo[t.motivo]}</p>
                <p className="text-xs text-texto-3">{decimal(t.t_hasta - t.t_desde, 1)} s</p>
                {t.asignado && <Rotulo className="text-ok">asignado a mano: {t.asignado === "ninguno" ? "ignorado" : t.asignado}</Rotulo>}
              </div>
            </div>
            {escritorio && (
              <AutoHeight>
                {viendo?.track === t.track && ticket.data && (
                  <VideoEn s={s} ticket={ticket.data.ticket} t={t.t_desde} />
                )}
              </AutoHeight>
            )}
            <ToggleGroup
              type="single"
              value={decisiones[t.track] ?? ""}
              onValueChange={(v) => setDecisiones((d) => ({ ...d, [t.track]: v as Rol | "" }))}
              aria-label={`Quién es en el tramo de ${mmss(t.t_desde)}`}
              className="mt-3 grid w-full grid-cols-3 gap-1"
            >
              <ToggleGroupItem value="A" className="min-h-11 border border-borde">A</ToggleGroupItem>
              <ToggleGroupItem value="B" className="min-h-11 border border-borde">B</ToggleGroupItem>
              <ToggleGroupItem value="ninguno" className="min-h-11 border border-borde">Ignorar</ToggleGroupItem>
            </ToggleGroup>
          </li>
        ))}
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
              <SheetDescription className="text-texto-3">{viendo ? textos.motivoTramo[viendo.motivo] : ""}</SheetDescription>
            </SheetHeader>
            <div className="px-4 pb-6">
              {viendo && ticket.data && <VideoEn s={s} ticket={ticket.data.ticket} t={viendo.t_desde} />}
            </div>
          </SheetContent>
        </Sheet>
      )}
    </section>
  );
}

function VideoEn({ s, ticket, t }: { s: Sesion; ticket: string; t: number }) {
  return (
    <video
      src={urlDeVideo(s.id, ticket, false)}
      controls
      playsInline
      preload="metadata"
      aria-label={`Video desde ${mmss(t)}`}
      className="mt-3 aspect-video w-full rounded-md border border-borde bg-black"
      onLoadedMetadata={(e) => { e.currentTarget.currentTime = t; }}
    />
  );
}
