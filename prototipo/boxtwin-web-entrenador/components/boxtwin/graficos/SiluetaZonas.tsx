/*
 * Silueta frontal con las cuatro zonas (cambios-14-10, W22 y W26; DC-2).
 *
 * Cabeza y torso partidos al medio. La silueta se ve como la ve el rival, de frente: la mitad
 * de la izquierda de la imagen queda enfrente de la mano izquierda del rival y lleva los golpes
 * de esa mano. No dice en que lado del cuerpo pego; el rotulo de cada mitad lo aclara con
 * palabras. Cada zona lleva su porcentaje escrito: el color nunca va solo.
 *
 * Con `colorear` en falso (mapa preliminar, menos de 50 golpes) las zonas van sin relleno y
 * solo con el numero.
 */
import type { Zona } from "@/lib/api/esquemas";
import { porcentajeAR } from "@/lib/formato";

const ANCHO = 200;
const ALTO = 260;

// Las cuatro zonas, en coordenadas del dibujo.
const FORMAS: Record<Zona, { d: string; tx: number; ty: number }> = {
  cabeza_mano_izq: { d: "M100 14 A34 38 0 0 0 100 90 Z", tx: 82, ty: 58 },
  cabeza_mano_der: { d: "M100 14 A34 38 0 0 1 100 90 Z", tx: 118, ty: 58 },
  cuerpo_mano_izq: { d: "M100 102 L52 110 Q40 116 42 132 L50 236 L100 240 Z", tx: 74, ty: 176 },
  cuerpo_mano_der: { d: "M100 102 L148 110 Q160 116 158 132 L150 236 L100 240 Z", tx: 126, ty: 176 },
};

export function SiluetaZonas({
  fracciones, colorear, etiqueta, ancho = 180,
}: {
  fracciones: Record<Zona, number | null>;
  colorear: boolean;
  etiqueta: string;
  ancho?: number;
}) {
  const max = Math.max(1e-9, ...Object.values(fracciones).map((f) => f ?? 0));
  return (
    <svg viewBox={`0 0 ${ANCHO} ${ALTO + 18}`} width={ancho} role="img" aria-label={etiqueta} className="mx-auto block max-w-full">
      {(Object.keys(FORMAS) as Zona[]).map((z) => {
        const f = fracciones[z];
        const forma = FORMAS[z];
        return (
          <g key={z} data-zona={z}>
            <path
              d={forma.d}
              className="stroke-borde-fuerte"
              strokeWidth={1.5}
              fill={colorear && f != null ? "var(--color-estimado)" : "transparent"}
              fillOpacity={colorear && f != null ? 0.12 + 0.68 * (f / max) : 0}
            />
            <text x={forma.tx} y={forma.ty} textAnchor="middle" className="fill-texto font-mono text-[13px] font-semibold">
              {f == null ? "—" : porcentajeAR(f)}
            </text>
          </g>
        );
      })}
      <line x1={100} y1={8} x2={100} y2={244} className="stroke-borde" strokeDasharray="3 3" />
      <text x={50} y={ALTO + 14} textAnchor="middle" className="fill-texto-3 text-[10px]">mano izq. del rival</text>
      <text x={150} y={ALTO + 14} textAnchor="middle" className="fill-texto-3 text-[10px]">mano der. del rival</text>
    </svg>
  );
}

export const NOMBRE_ZONA: Record<Zona, string> = {
  cabeza_mano_izq: "Cabeza, mano izquierda del rival",
  cabeza_mano_der: "Cabeza, mano derecha del rival",
  cuerpo_mano_izq: "Cuerpo, mano izquierda del rival",
  cuerpo_mano_der: "Cuerpo, mano derecha del rival",
};

/* Desde el que tira (ataque y cruces): "a la cabeza con la izquierda". */
export const NOMBRE_ZONA_ATAQUE: Record<Zona, string> = {
  cabeza_mano_izq: "A la cabeza con la izquierda",
  cabeza_mano_der: "A la cabeza con la derecha",
  cuerpo_mano_izq: "Al cuerpo con la izquierda",
  cuerpo_mano_der: "Al cuerpo con la derecha",
};
