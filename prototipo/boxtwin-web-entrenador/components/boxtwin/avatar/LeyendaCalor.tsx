"use client";

/*
 * Leyenda y controles del mapa de calor del avatar (27-10).
 *
 * La rampa es la misma del shader (lib/avatar/shaderCalor.ts, calorRampa). El color nunca va
 * solo: cada zona lleva su porcentaje en la lista. Los controles cambian uniforms y no rearman
 * la escena.
 */
import type { AjustesCalor } from "@/components/boxtwin/avatar/Avatar3D";
import { Estimado } from "@/components/boxtwin/EstadoBloque";
import { NOMBRE_ZONA } from "@/components/boxtwin/graficos/SiluetaZonas";
import { ZONAS, type Zona } from "@/lib/api/esquemas";
import { porcentajeAR } from "@/lib/formato";

const RAMPA = "linear-gradient(90deg, #0d26e6, #00d9f2 25%, #33f24d 50%, #ffe61a 75%, #ff1f0d)";

export function LeyendaCalor({
  ajustes, onCambio, fracciones, id,
}: {
  ajustes: AjustesCalor;
  onCambio: (a: AjustesCalor) => void;
  /* Por zona, la fraccion de golpes dirigidos; null si no hay dato. Sin esto, solo controles. */
  fracciones?: Record<Zona, number | null> | null;
  id: string;
}) {
  const cambiar = (parcial: Partial<AjustesCalor>) => onCambio({ ...ajustes, ...parcial });
  return (
    <div className="flex flex-col gap-3 rounded-md border border-borde p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm">Golpes dirigidos sobre el cuerpo</span>
        <label className="flex items-center gap-2 text-sm text-texto-2">
          <input type="checkbox" checked={ajustes.activo} onChange={(e) => cambiar({ activo: e.target.checked })} className="size-4 accent-[var(--color-estimado)]" />
          Mapa de calor
        </label>
      </div>
      <div aria-hidden="true">
        <div className="h-2.5 w-full rounded-full" style={{ background: RAMPA, opacity: ajustes.activo ? 1 : 0.35 }} />
        <div className="mt-1 flex justify-between text-xs text-texto-3"><span>menos</span><span>más dirigidos</span></div>
      </div>
      {fracciones && (
        <ul className="grid gap-1 text-sm sm:grid-cols-2">
          {ZONAS.map((z) => (
            <li key={z} className="flex justify-between gap-3 border-t border-borde-suave pt-1">
              <span className="text-texto-2">{NOMBRE_ZONA[z]}</span>
              <Estimado className="dato">{fracciones[z] == null ? "—" : porcentajeAR(fracciones[z])}</Estimado>
            </li>
          ))}
        </ul>
      )}
      <div className="grid gap-3 sm:grid-cols-2">
        <Deslizador id={`${id}-opacidad`} etiqueta="Intensidad" valor={ajustes.opacidad} onCambio={(v) => cambiar({ opacidad: v })} desactivado={!ajustes.activo} />
        <Deslizador id={`${id}-umbral`} etiqueta="Umbral" valor={ajustes.umbral} max={0.8} onCambio={(v) => cambiar({ umbral: v })} desactivado={!ajustes.activo} />
      </div>
    </div>
  );
}

function Deslizador({
  id, etiqueta, valor, onCambio, desactivado, max = 1,
}: {
  id: string;
  etiqueta: string;
  valor: number;
  onCambio: (v: number) => void;
  desactivado: boolean;
  max?: number;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="flex items-baseline justify-between text-sm">
        <span>{etiqueta}</span>
        <span className="dato text-texto-2">{Math.round(valor * 100)}%</span>
      </label>
      <input
        id={id}
        type="range"
        min={0}
        max={Math.round(max * 100)}
        step={5}
        value={Math.round(valor * 100)}
        disabled={desactivado}
        onChange={(e) => onCambio(Number(e.target.value) / 100)}
        aria-valuetext={`${Math.round(valor * 100)}%`}
        className="h-11 w-full accent-[var(--color-texto)] disabled:opacity-40"
      />
    </div>
  );
}
