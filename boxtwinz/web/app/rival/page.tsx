"use client";
/*
 * Rival fantasma (spec 8 y 12): elegir rival (arquetipo, publico, amigo o cajon premium) y
 * modalidad; ver la comparacion, las curvas, el mapa de exposicion y los consejos.
 */
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { Consejos } from "@/components/boxtwinz/Consejos";
import { Curva } from "@/components/boxtwinz/Curva";
import { Marco, useYo } from "@/components/boxtwinz/Marco";
import { Metricas } from "@/components/boxtwinz/Metricas";
import { Rotulo } from "@/components/boxtwinz/Rotulo";
import { api, ErrorApi, type Consejos as C, type Metrica, type Perfil } from "@/lib/api";
import { pct } from "@/lib/formato";
import { MODALIDAD, TEXTOS } from "@/lib/textos";

type Arq = { id: string; nombre: string; estilo: string; solo_premium: boolean; custom: boolean };
type Opcion = { tipo: string; id: string; nombre: string; grupo: string };
type Expo = { lado_izquierdo_usuario?: number; lado_derecho_usuario?: number; izquierda?: number; derecha?: number;
  simulado: boolean; rotulo?: string; aclaracion?: string; sesiones?: number };
type R = {
  yo: { nombre: string; metricas: Partial<Metrica>; curva: number[]; sesiones: number };
  rival: { nombre: string; metricas: Partial<Metrica>; curva: number[]; rotulo: string | null };
  exposicion: Expo | null; altura: { cabeza: number; cuerpo: number } | null; consejos: C; aclaracion: string;
};

function Contenido() {
  const q = useSearchParams();
  const { yo } = useYo();
  const [opciones, setOpciones] = useState<Opcion[]>([]);
  const [propio, setPropio] = useState<string | null>(null);
  const [sel, setSel] = useState(q.get("tipo") && q.get("id") ? `${q.get("tipo")}:${q.get("id")}` : "");
  const [mod, setMod] = useState("sombra");
  const [r, setR] = useState<R | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api<Perfil[]>("/perfiles"), api<Arq[]>("/arquetipos"),
      api<{ amigos: { usuario_id: string; nombre: string }[] }>("/amigos"),
      api<{ perfil_id: string; nombre: string }[]>("/publicos"),
    ]).then(([ps, arqs, am, pubs]) => {
      const p1 = ps.find((p) => p.ranura === 1)!;
      setPropio(p1.id);
      setOpciones([
        ...arqs.map((a) => ({ tipo: "arquetipo", id: a.id, nombre: `${a.nombre} (${a.estilo})`, grupo: TEXTOS.perfilSimulado })),
        ...(yo.plan === "premium" ? ps.filter((p) => p.ranura > 1 && p.customizado).map((p) => ({ tipo: "cajon", id: p.id, nombre: p.nombre, grupo: "Tus cajones" })) : []),
        ...am.amigos.map((a) => ({ tipo: "amigo", id: a.usuario_id, nombre: a.nombre, grupo: "Amigos" })),
        ...pubs.filter((p) => p.perfil_id !== p1.id).map((p) => ({ tipo: "publico", id: p.perfil_id, nombre: p.nombre, grupo: "Perfiles públicos" })),
      ]);
    }).catch(() => undefined);
  }, [yo.plan]);

  useEffect(() => {
    if (!propio || !sel) return;
    const [tipo, id] = sel.split(":");
    api<R>(`/rival?perfil_id=${propio}&modalidad=${mod}&rival_tipo=${tipo}&rival_id=${id}`)
      .then((x) => { setR(x); setError(null); })
      .catch((e) => { setR(null); setError(e instanceof ErrorApi ? e.message : "Error"); });
  }, [propio, sel, mod]);

  const grupos = [...new Set(opciones.map((o) => o.grupo))];
  const e = r?.exposicion;
  const izq = e ? (e.simulado ? e.izquierda ?? 0 : e.lado_izquierdo_usuario ?? 0) : 0;
  const der = e ? (e.simulado ? e.derecha ?? 0 : e.lado_derecho_usuario ?? 0) : 0;
  const tot = izq + der || 1;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-2">
        <label className="text-sm">Rival
          <select value={sel} onChange={(ev) => setSel(ev.target.value)} className="objetivo-tactil mt-1 w-full rounded-md border border-input bg-superficie-2 px-2">
            <option value="">Elegí un rival</option>
            {grupos.map((g) => (
              <optgroup key={g} label={g}>
                {opciones.filter((o) => o.grupo === g).map((o) => <option key={`${o.tipo}:${o.id}`} value={`${o.tipo}:${o.id}`}>{o.nombre}</option>)}
              </optgroup>
            ))}
          </select>
        </label>
        <label className="text-sm">Modalidad
          <select value={mod} onChange={(ev) => setMod(ev.target.value)} className="objetivo-tactil mt-1 w-full rounded-md border border-input bg-superficie-2 px-2">
            {Object.entries(MODALIDAD).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
        </label>
      </div>
      {error && <p role="alert" className="text-error">{error}</p>}
      {r && (
        <>
          <div className="flex flex-wrap gap-2">{r.rival.rotulo && <Rotulo>{r.rival.rotulo}</Rotulo>}</div>
          {r.yo.sesiones === 0 && <p className="text-sm text-texto-3">No tenés sesiones de {MODALIDAD[mod].toLowerCase()} todavía: subí una para compararte.</p>}
          <Metricas caption="Lado a lado" columnas={[{ nombre: "Vos", m: r.yo.metricas }, { nombre: r.rival.nombre, m: r.rival.metricas }]} />
          <Curva titulo="Ritmo: golpes detectados cada 10 s" series={[
            { nombre: "Vos", valores: r.yo.curva, color: "var(--color-texto)" },
            { nombre: r.rival.nombre, valores: r.rival.curva, color: "var(--color-b-trazo)" },
          ]} />
          {e && (
            <section aria-labelledby="t-mapa" className="rounded-lg border border-borde bg-superficie p-3">
              <h2 id="t-mapa" className="rotulo mb-2">Por dónde llegan sus golpes {e.simulado && <Rotulo>{e.rotulo}</Rotulo>}</h2>
              <div className="grid grid-cols-2 gap-2 text-center">
                <div><span className="dato block text-2xl">{pct(izq / tot)}</span><span className="text-xs text-tenue">por tu izquierda</span></div>
                <div><span className="dato block text-2xl">{pct(der / tot)}</span><span className="text-xs text-tenue">por tu derecha</span></div>
              </div>
              <p className="mt-2 text-xs text-tenue">{e.simulado ? "Mapa simulado del arquetipo: todavía no tenés sparrings contra este estilo." : e.aclaracion}</p>
            </section>
          )}
          {r.altura && <p className="text-sm">Cabeza {r.altura.cabeza} · Cuerpo {r.altura.cuerpo} <Rotulo>{TEXTOS.estimacion}</Rotulo></p>}
          <Consejos c={r.consejos} />
          <p className="text-xs text-tenue">{r.aclaracion}</p>
        </>
      )}
    </div>
  );
}

export default function Rival() {
  return <Marco titulo="Rival fantasma"><Suspense><Contenido /></Suspense></Marco>;
}
