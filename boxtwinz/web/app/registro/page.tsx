"use client";
/* Registro con los dos checks de la spec 3: uso obligatorio y reentrenamiento opcional. */
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api, ErrorApi } from "@/lib/api";
import { TEXTOS } from "@/lib/textos";

export default function Registro() {
  const router = useRouter();
  const [f, setF] = useState({ email: "", clave: "", nombre: "", guardia: "ortodoxa" });
  const [uso, setUso] = useState(false);
  const [reent, setReent] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [enviando, setEnviando] = useState(false);

  async function enviar(e: React.FormEvent) {
    e.preventDefault();
    setEnviando(true);
    setError(null);
    try {
      await api("/auth/registro", { method: "POST", json: { ...f, consentimiento_uso: uso, consentimiento_reentrenamiento: reent } });
      router.push("/inicio/");
    } catch (err) {
      setError(err instanceof ErrorApi ? err.message : "No se pudo crear la cuenta");
    } finally {
      setEnviando(false);
    }
  }

  return (
    <main className="mx-auto max-w-sm px-4 py-10">
      <h1 className="font-display text-4xl font-semibold">Crear cuenta</h1>
      <form onSubmit={enviar} className="mt-6 space-y-4" aria-label="Registro">
        <div className="space-y-1.5">
          <Label htmlFor="nombre">Nombre visible</Label>
          <Input id="nombre" required maxLength={120} value={f.nombre} onChange={(e) => setF({ ...f, nombre: e.target.value })} />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="email">Email</Label>
          <Input id="email" type="email" autoComplete="email" required value={f.email} onChange={(e) => setF({ ...f, email: e.target.value })} />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="clave">Contraseña (8 caracteres o más)</Label>
          <Input id="clave" type="password" autoComplete="new-password" minLength={8} required value={f.clave} onChange={(e) => setF({ ...f, clave: e.target.value })} />
        </div>
        <fieldset className="space-y-2">
          <legend className="text-sm font-medium">Tu guardia</legend>
          {(["ortodoxa", "zurda"] as const).map((g) => (
            <label key={g} className="objetivo-tactil mr-4 inline-flex items-center gap-2">
              <input type="radio" name="guardia" value={g} checked={f.guardia === g} onChange={() => setF({ ...f, guardia: g })} />
              {g === "ortodoxa" ? "Ortodoxa" : "Zurda"}
            </label>
          ))}
        </fieldset>
        <label className="flex items-start gap-3 rounded-md border border-borde p-3 text-sm">
          <input type="checkbox" className="mt-1 size-5 shrink-0" checked={uso} onChange={(e) => setUso(e.target.checked)} required aria-describedby="uso-obl" />
          <span>{TEXTOS.consentimientoUso} <span id="uso-obl" className="text-tenue">(obligatorio)</span></span>
        </label>
        <label className="flex items-start gap-3 rounded-md border border-borde p-3 text-sm">
          <input type="checkbox" className="mt-1 size-5 shrink-0" checked={reent} onChange={(e) => setReent(e.target.checked)} />
          <span>{TEXTOS.consentimientoReent} <span className="text-tenue">(opcional, lo podés revocar en Ajustes)</span></span>
        </label>
        {error && <p role="alert" className="text-sm text-error">{error}</p>}
        <Button type="submit" className="objetivo-tactil w-full" disabled={enviando || !uso}>Crear cuenta</Button>
      </form>
      <p className="mt-6 text-sm text-texto-2">¿Ya tenés cuenta? <Link href="/" className="underline underline-offset-4">Ingresá</Link></p>
    </main>
  );
}
