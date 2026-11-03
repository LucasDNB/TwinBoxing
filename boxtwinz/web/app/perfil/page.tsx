"use client";
/*
 * Perfil (spec 12): pestanas Sombra, Bolsa y Sparring con stat general, tendencia, historial y
 * "Comparar dos sesiones". En premium, una ranura 2 a 4 se puede volver cajon (customizado).
 */
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { Marco, useYo } from "@/components/boxtwinz/Marco";
import { Metricas } from "@/components/boxtwinz/Metricas";
import { Rotulo } from "@/components/boxtwinz/Rotulo";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api, ErrorApi, type Metrica, type Perfil, type SesionResumen } from "@/lib/api";
import { MODALIDAD, TEXTOS } from "@/lib/textos";
import { cn } from "@/lib/utils";

type Stat = {
  simulado: boolean; sesiones: number; ventana?: number; rotulo: string | null;
  promedio: Partial<Metrica>; tendencia: Record<string, string | null>; guardia_supuesta?: boolean;
};
const MODS = ["sombra", "bolsa", "sparring"] as const;
const FLECHA: Record<string, string> = { subiendo: "↑ subiendo", estable: "→ estable", bajando: "↓ bajando" };

function Personalizar({ p, alGuardar }: { p: Perfil; alGuardar: () => void }) {
  const [nombre, setNombre] = useState(p.nombre);
  const [estilo, setEstilo] = useState(p.estilo);
  const [error, setError] = useState<string | null>(null);
  async function guardar() {
    try {
      await api(`/perfiles/${p.id}`, { method: "PATCH", json: { nombre, estilo, customizado: true } });
      alGuardar();
    } catch (e) { setError(e instanceof ErrorApi ? e.message : "No se pudo guardar"); }
  }
  return (
    <details className="rounded-lg border border-borde p-3">
      <summary className="cursor-pointer text-sm">{p.customizado ? "Editar cajón" : "Convertir en cajón para un sparring real"}</summary>
      <div className="mt-3 space-y-3">
        <label className="block text-sm">Nombre<Input value={nombre} onChange={(e) => setNombre(e.target.value)} className="mt-1" /></label>
        <label className="block text-sm">Estilo
          <select value={estilo} onChange={(e) => setEstilo(e.target.value)} className="objetivo-tactil mt-1 w-full rounded-md border border-input bg-superficie-2 px-3">
            <option value="fajador">Fajador</option><option value="estilista">Estilista</option><option value="hibrido">Híbrido</option>
          </select>
        </label>
        {error && <p role="alert" className="text-sm text-error">{error}</p>}
        <Button className="objetivo-tactil" onClick={guardar}>Guardar</Button>
      </div>
    </details>
  );
}

function Contenido() {
  const id = useSearchParams().get("id") ?? "";
  const { yo } = useYo();
  const router = useRouter();
  const [p, setP] = useState<Perfil | null>(null);
  const [mod, setMod] = useState<(typeof MODS)[number]>("sombra");
  const [stat, setStat] = useState<Stat | null>(null);
  const [sesiones, setSesiones] = useState<SesionResumen[]>([]);
  const [marcadas, setMarcadas] = useState<string[]>([]);
  const [n, setN] = useState(0);

  useEffect(() => {
    api<Perfil[]>("/perfiles").then((ps) => setP(ps.find((x) => x.id === id) ?? null)).catch(() => undefined);
  }, [id, n]);
  useEffect(() => {
    api<Stat>(`/perfiles/${id}/stats?modalidad=${mod}`).then(setStat).catch(() => setStat(null));
    api<SesionResumen[]>(`/sesiones?perfil_id=${id}&modalidad=${mod}`).then(setSesiones).catch(() => undefined);
  }, [id, mod, n]);

  if (!p) return <p className="rotulo">Cargando…</p>;
  const listas = sesiones.filter((s) => s.estado === "lista");
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-xl font-medium">{p.nombre}</h2>
        {p.rotulo && <Rotulo>{p.rotulo}</Rotulo>}
        <span className="text-sm text-texto-3 capitalize">{p.ranura === 1 ? `guardia ${p.guardia}` : p.estilo}</span>
      </div>
      {p.ranura > 1 && yo.plan === "premium" && <Personalizar p={p} alGuardar={() => setN((x) => x + 1)} />}
      {p.ranura > 1 && !p.customizado && (
        <Link href={`/rival/?tipo=arquetipo&id=${p.arquetipo?.id ?? ""}`} className="text-sm underline underline-offset-4">Usar como rival fantasma</Link>
      )}

      <div role="tablist" aria-label="Modalidad" className="flex rounded-lg border border-borde p-1">
        {MODS.map((m) => (
          <button key={m} role="tab" aria-selected={mod === m} onClick={() => { setMod(m); setMarcadas([]); }}
            className={cn("objetivo-tactil flex-1 rounded-md text-sm", mod === m ? "bg-elevado text-texto" : "text-tenue")}>{MODALIDAD[m]}</button>
        ))}
      </div>

      <section role="tabpanel" className="space-y-3">
        {stat && (
          <>
            <p className="text-sm text-texto-3">
              {stat.simulado ? TEXTOS.perfilSimulado : `Promedio de las últimas ${stat.sesiones} sesiones (ventana de ${stat.ventana}).`}
              {stat.rotulo && !stat.simulado && <> <Rotulo>{stat.rotulo}</Rotulo></>}
            </p>
            <Metricas caption="Stat general" columnas={[{ nombre: "Promedio", m: stat.promedio }]} />
            {!stat.simulado && stat.sesiones > 1 && (
              <ul className="text-sm text-texto-2">
                <li>Volumen: {FLECHA[stat.tendencia.volumen ?? ""] ?? "—"}</li>
                <li>Sostén del ritmo: {FLECHA[stat.tendencia.sosten_ritmo ?? ""] ?? "—"}</li>
              </ul>
            )}
            {(stat.guardia_supuesta || stat.simulado) && <p className="text-xs text-tenue">Mano adelantada calculada con guardia supuesta ortodoxa.</p>}
          </>
        )}

        {!stat?.simulado && (
          <>
            <h3 className="rotulo">Historial</h3>
            {listas.length === 0 && <p className="text-texto-3">No hay sesiones de {MODALIDAD[mod].toLowerCase()} todavía.</p>}
            <ul className="divide-y divide-borde-suave rounded-lg border border-borde">
              {listas.map((s) => (
                <li key={s.id} className="flex items-center gap-3 px-3">
                  <input type="checkbox" className="size-5" aria-label={`Comparar la sesión del ${new Date(s.creado).toLocaleDateString("es-AR")}`}
                    checked={marcadas.includes(s.id)} disabled={!marcadas.includes(s.id) && marcadas.length >= 2}
                    onChange={(e) => setMarcadas((m) => (e.target.checked ? [...m, s.id] : m.filter((x) => x !== s.id)))} />
                  <Link href={`/sesion/?id=${s.id}`} className="objetivo-tactil flex flex-1 items-center justify-between">
                    <span className="dato">{new Date(s.creado).toLocaleString("es-AR", { dateStyle: "short", timeStyle: "short" })}</span>
                    <span className="text-sm text-texto-3">{s.duracion_s ? `${Math.round(s.duracion_s / 60)} min` : ""}</span>
                  </Link>
                </li>
              ))}
            </ul>
            <Button variant="outline" className="objetivo-tactil w-full" disabled={marcadas.length !== 2}
              onClick={() => router.push(`/comparar/?a=${marcadas[0]}&b=${marcadas[1]}`)}>Comparar dos sesiones</Button>
          </>
        )}
      </section>
    </div>
  );
}

export default function PaginaPerfil() {
  return <Marco titulo="Perfil"><Suspense><Contenido /></Suspense></Marco>;
}
