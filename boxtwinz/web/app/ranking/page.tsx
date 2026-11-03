"use client";
/* Ranking de actividad (spec 12): modalidad y metrica, top 100 y la posicion propia. */
import Link from "next/link";
import { useEffect, useState } from "react";
import { Marco } from "@/components/boxtwinz/Marco";
import { api } from "@/lib/api";
import { num, pct } from "@/lib/formato";
import { MODALIDAD, TEXTOS } from "@/lib/textos";

type Fila = { perfil_id: string; nombre: string; valor: number; sesiones: number; posicion: number };
type R = { total: number; top: Fila[]; propia: Fila | null; pie: string };
const METRICAS: Record<string, { nombre: string; fmt: (v: number) => string }> = {
  volumen: { nombre: "Golpes detectados por minuto", fmt: (v) => num(v) },
  combinaciones_min: { nombre: "Combinaciones por minuto", fmt: (v) => num(v) },
  pct_golpes_en_combinacion: { nombre: "Golpes en combinación", fmt: pct },
  sosten_ritmo: { nombre: "Sostén del ritmo", fmt: (v) => num(v, 2) },
};

function Contenido() {
  const [mod, setMod] = useState("sombra");
  const [met, setMet] = useState("volumen");
  const [r, setR] = useState<R | null>(null);
  useEffect(() => { api<R>(`/ranking?modalidad=${mod}&metrica=${met}`).then(setR).catch(() => undefined); }, [mod, met]);
  const f = METRICAS[met].fmt;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-2">
        <label className="text-sm">Modalidad
          <select value={mod} onChange={(e) => setMod(e.target.value)} className="objetivo-tactil mt-1 w-full rounded-md border border-input bg-superficie-2 px-2">
            {Object.entries(MODALIDAD).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
        </label>
        <label className="text-sm">Métrica
          <select value={met} onChange={(e) => setMet(e.target.value)} className="objetivo-tactil mt-1 w-full rounded-md border border-input bg-superficie-2 px-2">
            {Object.entries(METRICAS).map(([k, v]) => <option key={k} value={k}>{v.nombre}</option>)}
          </select>
        </label>
      </div>
      {r?.propia ? (
        <p className="rounded-lg border border-borde bg-superficie p-3">Tu posición: <strong className="dato">{r.propia.posicion}</strong> de {r.total} · <span className="dato">{f(r.propia.valor)}</span></p>
      ) : (
        <p className="text-sm text-texto-3">No aparecés: para entrar tu perfil tiene que ser público (Ajustes) y tener sesiones de 3 minutos o más con buena calidad de pose.</p>
      )}
      <ol className="divide-y divide-borde-suave rounded-lg border border-borde">
        {r?.top.map((x) => (
          <li key={x.perfil_id}>
            <Link href={`/publico/?id=${x.perfil_id}`} className="objetivo-tactil flex items-center gap-3 px-3">
              <span className="dato w-8 text-tenue">{x.posicion}</span>
              <span className="flex-1">{x.nombre}</span>
              <span className="dato">{f(x.valor)}</span>
            </Link>
          </li>
        ))}
        {r && r.top.length === 0 && <li className="p-3 text-texto-3">Todavía no hay nadie en este ranking.</li>}
      </ol>
      <p className="text-xs text-tenue">{TEXTOS.pieRanking}</p>
    </div>
  );
}

export default function Ranking() {
  return <Marco titulo="Ranking"><Contenido /></Marco>;
}
