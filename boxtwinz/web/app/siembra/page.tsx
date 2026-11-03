"use client";
/* Siembra del sparring (spec 12): marcar quien es el usuario y quien el rival; cajon en premium. */
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { Marco, useYo } from "@/components/boxtwinz/Marco";
import { Button } from "@/components/ui/button";
import { api, ErrorApi } from "@/lib/api";
import { tiempo } from "@/lib/formato";
import { cn } from "@/lib/utils";

type Cand = { track: number; sugerido: boolean; color_guante: string | null; t_desde: number; t_hasta: number; recorte_url: string };
type Datos = { candidatos: Cand[]; cajones: { id: string; nombre: string; estilo: string }[] };

function Contenido() {
  const id = useSearchParams().get("id") ?? "";
  const { yo } = useYo();
  const router = useRouter();
  const [d, setD] = useState<Datos | null>(null);
  const [usuario, setUsuario] = useState<number | null>(null);
  const [rival, setRival] = useState<number | null>(null);
  const [cajon, setCajon] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<Datos>(`/sesiones/${id}/candidatos`).then(setD).catch((e) => setError(e instanceof ErrorApi ? e.message : "Error"));
  }, [id]);

  function elegir(t: number) {
    if (usuario === null || (usuario !== null && rival !== null)) { setUsuario(t); setRival(null); }
    else if (t !== usuario) setRival(t);
  }

  async function enviar() {
    try {
      await api(`/sesiones/${id}/siembra`, { method: "POST", json: { usuario, rival, rival_cajon_perfil_id: cajon || null } });
      router.push(`/sesion/?id=${id}`);
    } catch (e) {
      setError(e instanceof ErrorApi ? e.message : "No se pudo guardar");
    }
  }

  if (error) return <p role="alert" className="text-error">{error}</p>;
  if (!d) return <p className="rotulo">Cargando…</p>;
  const sugeridos = d.candidatos.filter((c) => c.sugerido);
  const otros = d.candidatos.filter((c) => !c.sugerido);
  const tarjeta = (c: Cand) => {
    const rol = c.track === usuario ? "Vos" : c.track === rival ? "Rival" : null;
    return (
      <li key={c.track}>
        <button type="button" onClick={() => elegir(c.track)} aria-pressed={rol !== null}
          aria-label={`Persona ${c.track}${c.color_guante ? `, guantes ${c.color_guante}` : ""}${rol ? `, marcada como ${rol}` : ""}`}
          className={cn("relative w-full overflow-hidden rounded-lg border-2 bg-superficie text-left",
            rol === "Vos" ? "border-a-trazo" : rol === "Rival" ? "border-b-trazo" : "border-borde")}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={c.recorte_url} alt="" className="aspect-[3/4] w-full object-cover" />
          <span className="block p-2 text-xs text-texto-3">{c.color_guante ? `Guantes ${c.color_guante} · ` : ""}{tiempo(c.t_desde)}–{tiempo(c.t_hasta)}</span>
          {rol && <span className="absolute top-2 left-2 rounded-sm bg-fondo/90 px-2 py-0.5 text-sm font-medium">{rol}</span>}
        </button>
      </li>
    );
  };
  return (
    <div className="space-y-4">
      <p className="text-texto-2">Tocá primero a <strong>vos</strong> y después a tu <strong>rival</strong>.</p>
      <ul className="grid grid-cols-2 gap-3">{sugeridos.map(tarjeta)}</ul>
      {otros.length > 0 && (
        <details>
          <summary className="objetivo-tactil cursor-pointer py-2 text-sm text-texto-3">Otras personas detectadas ({otros.length})</summary>
          <ul className="mt-2 grid grid-cols-3 gap-2">{otros.map(tarjeta)}</ul>
        </details>
      )}
      {yo.plan === "premium" && d.cajones.length > 0 && (
        <label className="block space-y-1.5">
          <span className="text-sm">Guardar al rival en un cajón</span>
          <select value={cajon} onChange={(e) => setCajon(e.target.value)} className="objetivo-tactil w-full rounded-md border border-input bg-superficie-2 px-3">
            <option value="">No guardar</option>
            {d.cajones.map((c) => <option key={c.id} value={c.id}>{c.nombre}</option>)}
          </select>
        </label>
      )}
      <Button className="objetivo-tactil w-full" disabled={usuario === null || rival === null} onClick={enviar}>Contar golpes</Button>
    </div>
  );
}

export default function Siembra() {
  return <Marco titulo="¿Quién es quién?"><Suspense><Contenido /></Suspense></Marco>;
}
