"use client";
/*
 * Panel de administracion (spec 11): uso, capacidad, rendimiento, calidad, moderacion,
 * ingresos y notificaciones, con exportacion CSV de cada bloque.
 */
import { useCallback, useEffect, useState } from "react";
import { Marco, useYo } from "@/components/boxtwinz/Marco";
import { Rotulo } from "@/components/boxtwinz/Rotulo";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { num, pct } from "@/lib/formato";
import { cn } from "@/lib/utils";

type Fila = Record<string, string | number | null>;
type Bloque = { tablas: Record<string, Fila[]>; [k: string]: unknown };
type Cap = Bloque & {
  umbrales: number[]; cola: { en_cola: number; tomados: number }; espera_s: { p50: number | null; p95: number | null };
  disparadores: Record<string, { racha: number; dias: number; activo: boolean; texto: string }>;
  proyeccion: { horas_video_por_hora_gpu: number | null; horas_video_mes_70: number | null; horas_video_mes_100: number | null };
};
type Den = { id: string; motivo: string; detalle: string | null; tanda_id: string | null; perfil: { nombre: string } | null };
type Noti = { id: string; tipo: string; payload: Record<string, string>; leida: boolean; creado: string };
const PESTANAS = ["resumen", "capacidad", "rendimiento", "calidad", "moderacion", "ingresos"] as const;
const NOMBRE: Record<string, string> = { resumen: "Uso", capacidad: "Capacidad", rendimiento: "Rendimiento", calidad: "Calidad", moderacion: "Moderación", ingresos: "Ingresos" };

function Tabla({ nombre, filas }: { nombre: string; filas: Fila[] }) {
  if (!filas.length) return <p className="text-sm text-texto-3">{nombre.replaceAll("_", " ")}: sin datos.</p>;
  const cols = Object.keys(filas[0]);
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <caption className="rotulo mb-1 text-left">{nombre.replaceAll("_", " ")}</caption>
        <thead><tr>{cols.map((c) => <th key={c} scope="col" className="py-1 pr-2 text-left font-normal text-tenue">{c}</th>)}</tr></thead>
        <tbody className="divide-y divide-borde-suave">
          {filas.map((f, i) => <tr key={i}>{cols.map((c) => <td key={c} className="dato py-1 pr-2">{typeof f[c] === "number" ? num(f[c] as number, Number.isInteger(f[c]) ? 0 : 2) : f[c] ?? "—"}</td>)}</tr>)}
        </tbody>
      </table>
    </div>
  );
}

function Utilizacion({ filas, umbrales }: { filas: Fila[]; umbrales: number[] }) {
  const W = 300, H = 100, b = W / Math.max(1, filas.length);
  return (
    <figure className="rounded-lg border border-borde p-3">
      <figcaption className="rotulo mb-2">Utilización diaria de la GPU, 30 días</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-28 w-full" role="img" aria-label="Utilización diaria con líneas en 70% y 90%">
        {filas.map((f, i) => {
          const u = (f.utilizacion as number | null) ?? 0;
          return <rect key={i} x={i * b + 1} width={b - 2} y={H - u * H} height={u * H} fill={u > umbrales[1] ? "var(--color-error)" : u > umbrales[0] ? "var(--color-estimado)" : "var(--color-texto-3)"} />;
        })}
        {umbrales.map((u) => <line key={u} x1={0} x2={W} y1={H - u * H} y2={H - u * H} stroke="var(--color-estimado-borde)" strokeDasharray="4 3" />)}
      </svg>
      <p className="text-xs text-tenue">Líneas en {umbrales.map((u) => pct(u)).join(" y ")}.</p>
    </figure>
  );
}

function Moderacion() {
  const [d, setD] = useState<{ abiertas: Den[]; tandas: { id: string; inicio: string; fin: string | null; cantidad_revisada: number }[] } | null>(null);
  const cargar = useCallback(() => { api<typeof d>("/admin/denuncias").then(setD).catch(() => undefined); }, []);
  useEffect(cargar, [cargar]);
  const accion = (id: string, a: string) => api(`/admin/denuncias/${id}/accion`, { method: "POST", json: { accion: a } }).then(cargar);
  if (!d) return <p className="rotulo">Cargando…</p>;
  return (
    <div className="space-y-4">
      <Button className="objetivo-tactil" disabled={!d.abiertas.some((x) => !x.tanda_id)} onClick={() => api("/admin/tandas", { method: "POST" }).then(cargar)}>Empezar tanda</Button>
      {d.abiertas.length === 0 && <p className="text-texto-3">No hay denuncias abiertas.</p>}
      <ul className="space-y-2">
        {d.abiertas.map((x) => (
          <li key={x.id} className="rounded-lg border border-borde p-3">
            <p><strong>{x.perfil?.nombre}</strong> · {x.motivo.replaceAll("_", " ")} {x.tanda_id && <span className="rotulo">en tanda</span>}</p>
            {x.detalle && <p className="text-sm text-texto-2">{x.detalle}</p>}
            <div className="mt-2 flex flex-wrap gap-2">
              <Button size="sm" variant="outline" onClick={() => accion(x.id, "desestimar")}>Desestimar</Button>
              <Button size="sm" variant="outline" onClick={() => accion(x.id, "sacar_ranking")}>Sacar del ranking</Button>
              <Button size="sm" variant="destructive" onClick={() => accion(x.id, "suspender")}>Suspender</Button>
            </div>
          </li>
        ))}
      </ul>
      <Tabla nombre="historial_de_tandas" filas={d.tandas.map((t) => ({ inicio: new Date(t.inicio).toLocaleString("es-AR"), fin: t.fin ? new Date(t.fin).toLocaleString("es-AR") : "abierta", revisadas: t.cantidad_revisada }))} />
    </div>
  );
}

function Contenido() {
  const { yo } = useYo();
  const [tab, setTab] = useState<(typeof PESTANAS)[number]>("resumen");
  const [b, setB] = useState<Bloque | null>(null);
  const [notis, setNotis] = useState<{ no_leidas: number; notificaciones: Noti[] } | null>(null);
  const [verNotis, setVerNotis] = useState(false);
  const cargarNotis = useCallback(() => { api<typeof notis>("/admin/notificaciones").then(setNotis).catch(() => undefined); }, []);
  useEffect(cargarNotis, [cargarNotis]);
  useEffect(() => { if (tab !== "moderacion") api<Bloque>(`/admin/${tab}`).then(setB).catch(() => setB(null)); }, [tab]);
  if (yo.rol !== "admin") return <p role="alert" className="text-error">Solo para administración.</p>;
  const cap = tab === "capacidad" ? (b as Cap | null) : null;
  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <Button variant="outline" size="sm" aria-expanded={verNotis} onClick={() => setVerNotis((v) => !v)}>
          Notificaciones{notis?.no_leidas ? ` (${notis.no_leidas} sin leer)` : ""}
        </Button>
      </div>
      {verNotis && notis && (
        <ul className="space-y-1 rounded-lg border border-borde p-2 text-sm">
          {notis.notificaciones.length === 0 && <li className="text-texto-3">Sin notificaciones.</li>}
          {notis.notificaciones.map((x) => (
            <li key={x.id} className={cn("flex items-center justify-between gap-2", x.leida && "text-tenue")}>
              <span>{x.tipo.replaceAll("_", " ")} · {x.payload.texto ?? x.payload.motivo ?? x.payload.etapa ?? ""}</span>
              {!x.leida && <Button size="sm" variant="ghost" onClick={() => api(`/admin/notificaciones/${x.id}/leida`, { method: "POST" }).then(cargarNotis)}>Leída</Button>}
            </li>
          ))}
        </ul>
      )}
      <div role="tablist" aria-label="Bloques" className="flex flex-wrap gap-1">
        {PESTANAS.map((p) => (
          <button key={p} role="tab" aria-selected={tab === p} onClick={() => { setB(null); setTab(p); }}
            className={cn("objetivo-tactil rounded-md px-3 text-sm", tab === p ? "bg-elevado" : "text-tenue")}>{NOMBRE[p]}</button>
        ))}
      </div>
      <section role="tabpanel" className="space-y-4">
        {tab === "moderacion" ? <Moderacion /> : !b ? <p className="rotulo">Cargando…</p> : (
          <>
            {cap && (
              <>
                <Utilizacion filas={cap.tablas.utilizacion_diaria} umbrales={cap.umbrales} />
                <p className="text-sm">Cola: <span className="dato">{cap.cola.en_cola}</span> en espera, <span className="dato">{cap.cola.tomados}</span> procesando · espera p95 <span className="dato">{cap.espera_s.p95 == null ? "—" : `${num(cap.espera_s.p95 / 60)} min`}</span></p>
                <ul className="text-sm">
                  {Object.entries(cap.disparadores).map(([k, v]) => (
                    <li key={k}>{k}: racha {v.racha} de {v.dias} días {v.activo && <Rotulo>{v.texto}</Rotulo>}</li>
                  ))}
                </ul>
                <p className="text-sm">Proyección: <span className="dato">{num(cap.proyeccion.horas_video_mes_70, 0)}</span> h de video por mes al 70% y <span className="dato">{num(cap.proyeccion.horas_video_mes_100, 0)}</span> h al 100%.</p>
              </>
            )}
            {tab === "ingresos" && <p><Rotulo>Pago simulado — versión de demostración</Rotulo> Premium: <span className="dato">{String(b.premium_simulado)}</span></p>}
            {tab === "calidad" && <p className="text-sm">Sesiones con consentimiento de reentrenamiento: <span className="dato">{String(b.sesiones_con_consentimiento_reentrenamiento)}</span></p>}
            {Object.entries(b.tablas).filter(([k]) => k !== "utilizacion_diaria").map(([k, f]) => <Tabla key={k} nombre={k} filas={f} />)}
            <a href={`/api/admin/exportar/${tab}.csv`} className="inline-block text-sm underline underline-offset-4" download>Exportar CSV</a>
          </>
        )}
      </section>
    </div>
  );
}

export default function Admin() {
  return <Marco titulo="Administración"><Contenido /></Marco>;
}
