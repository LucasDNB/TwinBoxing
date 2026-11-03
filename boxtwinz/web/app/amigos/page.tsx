"use client";
/* Amigos (spec 12): solicitudes, lista y perfil del amigo (solo su perfil 1). */
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";
import { Marco } from "@/components/boxtwinz/Marco";
import { VistaPublica, type Vista } from "@/components/boxtwinz/VistaPublica";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api, ErrorApi } from "@/lib/api";

type Item = { id: string; usuario_id: string; nombre: string; email?: string };
type Lista = { amigos: Item[]; recibidas: Item[]; enviadas: Item[] };

function PerfilAmigo({ uid }: { uid: string }) {
  const [v, setV] = useState<Vista | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api<Vista>(`/amigos/${uid}/perfil`).then(setV).catch((e) => setError(e instanceof ErrorApi ? e.message : "Error")); }, [uid]);
  if (error) return <p role="alert" className="text-error">{error}</p>;
  if (!v) return <p className="rotulo">Cargando…</p>;
  return (
    <div className="space-y-4">
      <h2 className="text-xl font-medium">{v.nombre}</h2>
      <VistaPublica v={v} />
      <Link href={`/rival/?tipo=amigo&id=${uid}`} className="objetivo-tactil inline-flex items-center rounded-md bg-primary px-4 text-sm font-medium text-primary-foreground">Usar como rival fantasma</Link>
      <p><Link href="/amigos/" className="text-sm underline underline-offset-4">Volver a amigos</Link></p>
    </div>
  );
}

function Contenido() {
  const uid = useSearchParams().get("id");
  const [l, setL] = useState<Lista | null>(null);
  const [email, setEmail] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const cargar = useCallback(() => { api<Lista>("/amigos").then(setL).catch(() => undefined); }, []);
  useEffect(cargar, [cargar]);
  if (uid) return <PerfilAmigo uid={uid} />;

  async function pedir(e: React.FormEvent) {
    e.preventDefault();
    try {
      await api("/amigos/solicitud", { method: "POST", json: { email } });
      setMsg("Si esa persona tiene cuenta, le llega la solicitud.");
      setEmail("");
      cargar();
    } catch (err) { setMsg(err instanceof ErrorApi ? err.message : "No se pudo enviar"); }
  }
  async function responder(id: string, accion: "aceptar" | "rechazar") {
    await api(`/amigos/${id}/${accion}`, { method: "POST" }).catch(() => undefined);
    cargar();
  }
  return (
    <div className="space-y-5">
      <form onSubmit={pedir} className="space-y-2" aria-label="Agregar amigo">
        <label htmlFor="email-amigo" className="text-sm">Agregar por email</label>
        <div className="flex gap-2">
          <Input id="email-amigo" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
          <Button type="submit" className="objetivo-tactil">Enviar</Button>
        </div>
        {msg && <p role="status" className="text-sm text-texto-3">{msg}</p>}
      </form>
      {l && l.recibidas.length > 0 && (
        <section>
          <h2 className="rotulo mb-2">Solicitudes</h2>
          <ul className="space-y-2">
            {l.recibidas.map((x) => (
              <li key={x.id} className="flex items-center justify-between rounded-lg border border-borde p-2">
                <span>{x.nombre} <span className="text-xs text-tenue">{x.email}</span></span>
                <span className="flex gap-1">
                  <Button size="sm" onClick={() => responder(x.id, "aceptar")}>Aceptar</Button>
                  <Button size="sm" variant="outline" onClick={() => responder(x.id, "rechazar")}>Rechazar</Button>
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
      <section>
        <h2 className="rotulo mb-2">Amigos</h2>
        {l?.amigos.length === 0 && <p className="text-texto-3">Todavía no tenés amigos agregados.</p>}
        <ul className="divide-y divide-borde-suave rounded-lg border border-borde">
          {l?.amigos.map((x) => (
            <li key={x.id}><Link href={`/amigos/?id=${x.usuario_id}`} className="objetivo-tactil flex items-center px-3">{x.nombre}</Link></li>
          ))}
        </ul>
      </section>
      {l && l.enviadas.length > 0 && <p className="text-sm text-texto-3">Pendientes de respuesta: {l.enviadas.map((x) => x.nombre).join(", ")}</p>}
    </div>
  );
}

export default function Amigos() {
  return <Marco titulo="Amigos"><Suspense><Contenido /></Suspense></Marco>;
}
