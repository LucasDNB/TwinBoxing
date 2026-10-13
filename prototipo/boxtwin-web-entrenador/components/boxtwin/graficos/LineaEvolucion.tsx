"use client";

/*
 * Golpes por minuto por sesion, en el tiempo (06, bloque 2). SVG propio (03 §5).
 *
 * Si cambio el detector entre dos sesiones la linea se corta y lo dice: comparar a traves
 * de ese corte es comparar dos programas, no al boxeador consigo mismo. Cada punto lleva a
 * su sesion.
 */
import Link from "next/link";
import type { Perfil } from "@/lib/api/esquemas";
import { decimal, fechaCorta } from "@/lib/formato";

const ALTO = 150;
const PASO = 64;

export function tramosDeEvolucion(ev: Perfil["evolucion"]) {
  const tramos: Perfil["evolucion"][] = [];
  for (const p of ev) {
    const ultimo = tramos[tramos.length - 1];
    if (ultimo && (ultimo[ultimo.length - 1].detector ?? null) === (p.detector ?? null)) ultimo.push(p);
    else tramos.push([p]);
  }
  return tramos;
}

export function LineaEvolucion({ evolucion }: { evolucion: Perfil["evolucion"] }) {
  const puntos = evolucion.filter((p) => p.golpes_por_minuto != null);
  if (puntos.length === 0) return <p className="text-sm text-texto-3">Todavía no hay sesiones con duración medida.</p>;
  const max = Math.max(...puntos.map((p) => p.golpes_por_minuto!), 1);
  const ancho = Math.max(puntos.length * PASO + 40, 280);
  const x = (i: number) => 30 + i * PASO;
  const y = (v: number) => ALTO - 10 - (v / max) * (ALTO - 30);
  const indice = new Map(puntos.map((p, i) => [p.sesion_id, i]));
  const tramos = tramosDeEvolucion(puntos);
  const lectura = `Golpes por minuto en ${puntos.length} sesiones, de ${decimal(puntos[0].golpes_por_minuto!, 1)} a ${decimal(puntos[puntos.length - 1].golpes_por_minuto!, 1)}`;

  return (
    <div>
      <div className="overflow-x-auto">
        <svg role="img" aria-label={lectura} viewBox={`0 0 ${ancho} ${ALTO + 20}`} style={{ minWidth: Math.min(ancho, 640) }} className="h-auto w-full">
          <line x1="20" x2={ancho - 10} y1={ALTO - 10} y2={ALTO - 10} className="stroke-borde-fuerte" />
          {tramos.map((t, k) => (
            <polyline
              key={k}
              fill="none"
              strokeWidth="2"
              className="stroke-texto-2"
              points={t.map((p) => `${x(indice.get(p.sesion_id)!)},${y(p.golpes_por_minuto!)}`).join(" ")}
            />
          ))}
          {puntos.map((p, i) => (
            <g key={p.sesion_id}>
              <circle cx={x(i)} cy={y(p.golpes_por_minuto!)} r="4" className="fill-texto" />
              <text x={x(i)} y={y(p.golpes_por_minuto!) - 8} textAnchor="middle" className="fill-texto-2 font-mono text-[10px]">{decimal(p.golpes_por_minuto!, 1)}</text>
              <text x={x(i)} y={ALTO + 8} textAnchor="middle" className="fill-tenue font-mono text-[10px]">{fechaCorta(p.fecha)}</text>
            </g>
          ))}
        </svg>
      </div>
      {tramos.length > 1 && (
        <p className="mt-2 text-xs text-estimado">
          La línea se corta donde cambió el detector: antes y después no se comparan.
        </p>
      )}
      <ol className="mt-2 flex flex-wrap gap-2 text-xs">
        {puntos.map((p) => (
          <li key={p.sesion_id}>
            <Link href={`/sesion/?id=${p.sesion_id}&vista=fightcard`} className="inline-flex min-h-11 items-center rounded-sm border border-borde px-2 dato underline-offset-4 hover:underline">
              {fechaCorta(p.fecha)} · {decimal(p.golpes_por_minuto!, 1)}/min
            </Link>
          </li>
        ))}
      </ol>
    </div>
  );
}
