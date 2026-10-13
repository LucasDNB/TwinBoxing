"use client";

/*
 * La linea de tiempo de golpes detectados (04 §04, punto 2; 03 §8).
 *
 * Dos carriles (A arriba, B abajo), un tick por golpe, divisiones por round y el cursor del
 * tiempo actual. El cursor se mueve con el tiempo del video y no con una animacion (03 §7).
 * Cada combinacion izq-der-izq (W13) se marca con un corchete sobre sus tres ticks; tocarlo
 * lleva al primer golpe, o sea a t_inicio - 0,5 s como cualquier golpe.
 *
 * Es un grupo de botones navegable con flechas (izquierda y derecha: golpe anterior y
 * siguiente del mismo peleador; arriba y abajo: cambiar de peleador), con tabindex movil:
 * un solo tick entra en el orden de tabulacion. Cada tick dice que es en su aria-label.
 */
import { useMemo, useRef, type KeyboardEvent } from "react";
import type { Golpe } from "@/lib/api/esquemas";
import type { FightCard, Lado } from "@/lib/fightcard";
import { lados as ladosDe, tipoEfectivo } from "@/lib/fightcard";
import { mmss, nombreDeTipo, tiempoHablado } from "@/lib/formato";
import { cn } from "@/lib/utils";

export interface Ventana {
  inicio: number;
  fin: number;
}

export function etiquetaDeGolpe(g: Golpe, nombre: string, nomenclatura: Record<string, string>): string {
  const t = tipoEfectivo(g);
  const tipo = g.descartado
    ? "marcado como no golpe"
    : t
      ? `${nombreDeTipo(t, nomenclatura)} ${g.corregido ? "corregido" : "estimado"}`
      : "sin tipo";
  return `Golpe detectado de ${nombre}, ${tipo}, ${tiempoHablado(g.t_inicio)}`;
}

/* El golpe anterior o siguiente del mismo peleador, y el mas cercano en el otro carril. */
export function vecino(golpes: Record<Lado, Golpe[]>, lado: Lado, actual: Golpe | null, tecla: string): { lado: Lado; golpe: Golpe } | null {
  const propios = golpes[lado] ?? [];
  if (tecla === "ArrowRight" || tecla === "ArrowLeft") {
    if (!propios.length) return null;
    if (!actual) return { lado, golpe: propios[0] };
    const i = propios.findIndex((g) => g.id === actual.id);
    const j = tecla === "ArrowRight" ? Math.min(i + 1, propios.length - 1) : Math.max(i - 1, 0);
    return { lado, golpe: propios[j] };
  }
  if (tecla === "ArrowUp" || tecla === "ArrowDown") {
    const otro: Lado = lado === "A" ? "B" : "A";
    const suyos = golpes[otro] ?? [];
    if (!suyos.length) return null;
    const t = actual?.t_inicio ?? 0;
    const cerca = suyos.reduce((m, g) => (Math.abs(g.t_inicio - t) < Math.abs(m.t_inicio - t) ? g : m), suyos[0]);
    return { lado: otro, golpe: cerca };
  }
  return null;
}

export function LineaTiempo({
  fc, nombres, tiempo, seleccionado, onElegir, ventana,
}: {
  fc: FightCard;
  nombres: Record<Lado, string>;
  tiempo: number;
  seleccionado: { lado: Lado; id: string } | null;
  onElegir: (lado: Lado, g: Golpe) => void;
  ventana: Ventana;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const lados = ladosDe(fc);
  const golpes = useMemo(() => {
    const r = {} as Record<Lado, Golpe[]>;
    for (const l of lados) r[l] = [...(fc.peleadores[l]?.golpes ?? [])].sort((a, b) => a.t_inicio - b.t_inicio);
    return r;
  }, [fc, lados]);
  const largo = Math.max(ventana.fin - ventana.inicio, 1e-6);
  const pos = (t: number) => `${((t - ventana.inicio) / largo) * 100}%`;
  const dentro = (t: number) => t >= ventana.inicio && t <= ventana.fin;

  const enFoco = seleccionado ?? (golpes.A?.[0] ? { lado: "A" as Lado, id: golpes.A[0].id } : golpes.B?.[0] ? { lado: "B" as Lado, id: golpes.B[0].id } : null);

  function tecla(e: KeyboardEvent<HTMLDivElement>) {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) return;
    e.preventDefault();
    const lado = enFoco?.lado ?? "A";
    const actual = golpes[lado]?.find((g) => g.id === enFoco?.id) ?? null;
    const v = vecino(golpes, lado, actual, e.key);
    if (!v) return;
    onElegir(v.lado, v.golpe);
    requestAnimationFrame(() => ref.current?.querySelector<HTMLButtonElement>(`[data-golpe="${v.golpe.id}"]`)?.focus());
  }

  return (
    <div
      ref={ref}
      role="group"
      aria-label="Línea de tiempo de golpes detectados. Flechas izquierda y derecha: golpe anterior y siguiente; arriba y abajo: el otro peleador."
      onKeyDown={tecla}
      className="relative select-none rounded-md border border-borde bg-superficie-2 px-2 py-2"
    >
      {/* Divisiones por round */}
      <div aria-hidden="true" className="pointer-events-none absolute inset-y-0 left-2 right-2">
        {fc.video.rounds.filter((r) => dentro(r.inicio_s)).map((r) => (
          <div key={r.round} className="absolute inset-y-0 border-l border-dashed border-borde-fuerte" style={{ left: pos(r.inicio_s) }}>
            <span className="rotulo absolute -top-0.5 left-1 text-[10px]">R{r.round}</span>
          </div>
        ))}
      </div>
      {lados.map((l) => (
        <div key={l} className="relative mt-3 h-9 first:mt-4">
          <span className={cn("rotulo absolute -left-0.5 -top-3.5 text-[10px]", l === "A" ? "text-a-trazo" : "text-b-trazo")}>{l}</span>
          <div aria-hidden="true" className="absolute inset-x-0 top-1/2 h-px bg-borde" />
          {(fc.peleadores[l]?.combinaciones?.lista ?? []).filter((c) => dentro(c.t_inicio)).map((c) => {
            const primero = golpes[l].find((g) => g.id === c.golpes[0]);
            const ultimo = golpes[l].find((g) => g.id === c.golpes[2]);
            if (!primero || !ultimo) return null;
            return (
              <button
                key={c.golpes[0]}
                type="button"
                data-combinacion={c.golpes[0]}
                // Fuera del orden de tabulacion: la linea tiene una sola parada (CA-04.4) y el
                // teclado llega a la misma terna por la lista, donde lleva la insignia.
                tabIndex={-1}
                aria-label={`Combinación izquierda, derecha, izquierda de ${nombres[l]}, ${tiempoHablado(c.t_inicio)}`}
                onClick={() => onElegir(l, primero)}
                className="absolute -top-2 z-10 h-2.5 min-w-3 -translate-x-1.5 border-x-2 border-t-2 border-texto-2 hover:border-texto focus-visible:outline-2"
                style={{ left: pos(primero.t_inicio), width: `calc(${pos(Math.min(ultimo.t_inicio, ventana.fin))} - ${pos(primero.t_inicio)} + 12px)` }}
              />
            );
          })}
          {golpes[l].filter((g) => dentro(g.t_inicio)).map((g) => {
            const sel = seleccionado?.id === g.id;
            const foco = enFoco?.id === g.id;
            return (
              <button
                key={g.id}
                type="button"
                data-golpe={g.id}
                tabIndex={foco ? 0 : -1}
                aria-pressed={sel}
                aria-label={etiquetaDeGolpe(g, nombres[l], fc.nomenclatura)}
                onClick={() => onElegir(l, g)}
                className="group absolute top-0 flex h-9 w-3 -translate-x-1/2 items-center justify-center focus-visible:outline-2"
                style={{ left: pos(g.t_inicio) }}
              >
                <span
                  className={cn(
                    "block w-[3px] rounded-sm",
                    l === "A" ? "bg-a-trazo" : "bg-b-trazo",
                    l === "B" && "rounded-full",
                    sel ? "h-9 w-[5px] ring-2 ring-texto" : "h-6",
                    g.descartado && "opacity-40",
                  )}
                />
              </button>
            );
          })}
        </div>
      ))}
      {dentro(tiempo) && (
        <div aria-hidden="true" className="pointer-events-none absolute inset-y-1 left-2 right-2">
          <div className="absolute inset-y-0 w-px bg-texto" style={{ left: pos(tiempo) }} />
        </div>
      )}
      <div className="mt-1 flex justify-between px-0.5 dato text-[11px] text-tenue" aria-hidden="true">
        <span>{mmss(ventana.inicio)}</span>
        <span>{mmss(ventana.fin)}</span>
      </div>
    </div>
  );
}
