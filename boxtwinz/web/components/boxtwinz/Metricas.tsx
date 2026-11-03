/* Tabla de metricas de una o varias columnas (usuario/rival, sesion A/B, vos/rival fantasma). */
import type { Metrica } from "@/lib/api";
import { num, pct } from "@/lib/formato";

const FILAS: { clave: keyof Metrica; nombre: string; fmt: (v: number | null) => string }[] = [
  { clave: "golpes", nombre: "Golpes detectados", fmt: (v) => num(v, 0) },
  { clave: "volumen", nombre: "Golpes detectados por minuto", fmt: (v) => num(v) },
  { clave: "pct_izquierda", nombre: "Izquierda", fmt: pct },
  { clave: "pct_mano_adelantada", nombre: "Mano adelantada", fmt: pct },
  { clave: "combinaciones_min", nombre: "Combinaciones por minuto", fmt: (v) => num(v) },
  { clave: "pct_golpes_en_combinacion", nombre: "Golpes en combinación", fmt: pct },
  { clave: "sosten_ritmo", nombre: "Sostén del ritmo", fmt: (v) => num(v, 2) },
  { clave: "iniciativa", nombre: "Iniciativa", fmt: pct },
];

export function Metricas({ columnas, caption }: { columnas: { nombre: string; m: Partial<Metrica> | null | undefined }[]; caption: string }) {
  const filas = FILAS.filter((f) => columnas.some((c) => c.m && c.m[f.clave] != null));
  return (
    <table className="w-full text-sm">
      <caption className="rotulo mb-2 text-left">{caption}</caption>
      <thead>
        <tr className="text-left text-tenue">
          <th scope="col" className="py-1 font-normal">Métrica</th>
          {columnas.map((c) => <th key={c.nombre} scope="col" className="py-1 text-right font-normal">{c.nombre}</th>)}
        </tr>
      </thead>
      <tbody className="divide-y divide-borde-suave">
        {filas.map((f) => (
          <tr key={f.clave}>
            <th scope="row" className="py-1.5 text-left font-normal text-texto-2">{f.nombre}</th>
            {columnas.map((c) => (
              <td key={c.nombre} className="dato py-1.5 text-right">{f.fmt((c.m?.[f.clave] as number | null | undefined) ?? null)}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
