"use client";
/* Stats por modalidad de otro boxeador (perfil 1): lo usan /publico y /amigos. */
import { useState } from "react";
import { Metricas } from "@/components/boxtwinz/Metricas";
import type { Metrica } from "@/lib/api";
import { MODALIDAD } from "@/lib/textos";
import { cn } from "@/lib/utils";

export type Vista = {
  perfil_id: string; usuario_id: string; nombre: string; guardia: string | null;
  stats: Record<string, { sesiones: number; promedio: Partial<Metrica>; rotulo: string | null }>;
  sesiones?: { id: string; modalidad: string; creado: string; duracion_s: number | null }[];
};

export function VistaPublica({ v }: { v: Vista }) {
  const [mod, setMod] = useState("sombra");
  const st = v.stats[mod];
  return (
    <div className="space-y-3">
      <p className="text-texto-3">Guardia {v.guardia}</p>
      <div role="tablist" aria-label="Modalidad" className="flex rounded-lg border border-borde p-1">
        {Object.keys(MODALIDAD).map((m) => (
          <button key={m} role="tab" aria-selected={mod === m} onClick={() => setMod(m)}
            className={cn("objetivo-tactil flex-1 rounded-md text-sm", mod === m ? "bg-elevado" : "text-tenue")}>{MODALIDAD[m]}</button>
        ))}
      </div>
      <div role="tabpanel">
        {st && st.sesiones > 0 ? (
          <Metricas caption={`Promedio de ${st.sesiones} sesiones`} columnas={[{ nombre: v.nombre, m: st.promedio }]} />
        ) : <p className="text-texto-3">Sin sesiones de {MODALIDAD[mod].toLowerCase()}.</p>}
        {v.sesiones && (
          <p className="mt-2 text-sm text-texto-3">{v.sesiones.filter((s) => s.modalidad === mod).length} sesiones de {MODALIDAD[mod].toLowerCase()}.</p>
        )}
      </div>
    </div>
  );
}
