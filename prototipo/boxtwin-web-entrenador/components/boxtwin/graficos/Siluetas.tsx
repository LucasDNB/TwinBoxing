"use client";

/*
 * Dimensiones reales: dos siluetas a escala por altura y envergadura (04 §07, punto 2).
 * Superpuestas o lado a lado. Si falta un dato la silueta no se dibuja y se dice que falta:
 * una silueta de altura inventada es un numero inventado con forma de persona.
 */
import { decimal } from "@/lib/formato";

export interface Medidas {
  nombre: string;
  altura_m: number | null;
  envergadura_m: number | null;
}

const ALTO = 220;
const ESCALA = ALTO / 2.3; // px por metro: entra hasta 2,30 m

function Figura({ m, x, clase, circulo }: { m: Medidas; x: number; clase: string; circulo: boolean }) {
  const h = m.altura_m! * ESCALA;
  const e = m.envergadura_m! * ESCALA;
  const base = ALTO;
  const cabeza = h * 0.065;
  const hombro = base - h * 0.82;
  return (
    <g className={clase} fill="none" strokeWidth="3" strokeLinecap="round">
      {circulo ? <circle cx={x} cy={base - h + cabeza} r={cabeza} /> : <rect x={x - cabeza} y={base - h} width={cabeza * 2} height={cabeza * 2} rx="2" />}
      <line x1={x} y1={base - h + cabeza * 2} x2={x} y2={base - h * 0.48} />
      <line x1={x - e / 2} y1={hombro} x2={x + e / 2} y2={hombro} />
      <line x1={x} y1={base - h * 0.48} x2={x - h * 0.12} y2={base} />
      <line x1={x} y1={base - h * 0.48} x2={x + h * 0.12} y2={base} />
    </g>
  );
}

export function Siluetas({ a, b, superpuestas }: { a: Medidas; b: Medidas; superpuestas: boolean }) {
  const completo = (m: Medidas) => m.altura_m != null && m.envergadura_m != null;
  const faltan = [a, b].filter((m) => !completo(m));
  const ancho = superpuestas ? 260 : 440;
  const xa = superpuestas ? ancho / 2 : 110;
  const xb = superpuestas ? ancho / 2 : 330;
  const lectura = [a, b].filter(completo).map((m) => `${m.nombre}: ${decimal(m.altura_m, 2)} m de altura y ${decimal(m.envergadura_m, 2)} m de envergadura`).join("; ");
  return (
    <div>
      {[a, b].some(completo) && (
        <svg role="img" aria-label={lectura} viewBox={`0 0 ${ancho} ${ALTO + 4}`} className="mx-auto h-auto w-full max-w-md">
          <line x1="0" x2={ancho} y1={ALTO} y2={ALTO} className="stroke-borde-fuerte" />
          {completo(a) && <Figura m={a} x={xa} clase="stroke-a-trazo" circulo={false} />}
          {completo(b) && <Figura m={b} x={xb} clase="stroke-b-trazo" circulo />}
        </svg>
      )}
      {faltan.map((m) => (
        <p key={m.nombre} className="mt-2 text-sm text-texto-3">
          De {m.nombre} falta {[m.altura_m == null && "la altura", m.envergadura_m == null && "la envergadura"].filter(Boolean).join(" y ")}: cargala en su ficha para dibujarlo a escala.
        </p>
      ))}
    </div>
  );
}
