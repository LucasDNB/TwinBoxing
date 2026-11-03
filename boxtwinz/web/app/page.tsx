"use client";
/* Bienvenida: ingresar o ir al registro (spec 12, "/"). */
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api, ErrorApi } from "@/lib/api";
import { TEXTOS } from "@/lib/textos";

export default function Bienvenida() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [clave, setClave] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [enviando, setEnviando] = useState(false);

  async function ingresar(e: React.FormEvent) {
    e.preventDefault();
    setEnviando(true);
    setError(null);
    try {
      await api("/auth/ingreso", { method: "POST", json: { email, clave } });
      router.push("/inicio/");
    } catch (err) {
      setError(err instanceof ErrorApi ? err.message : "No se pudo ingresar");
    } finally {
      setEnviando(false);
    }
  }

  return (
    <main className="mx-auto flex min-h-dvh max-w-sm flex-col justify-center px-4 py-10">
      <h1 className="font-display text-5xl font-semibold tracking-wide">BoxTwin<span className="text-a-trazo">Z</span></h1>
      <p className="mt-2 text-texto-2">
        Subí tu sombra, tu bolsa o tu sparring y mirá los {TEXTOS.golpes} de izquierda y derecha, round por round.
      </p>
      <form onSubmit={ingresar} className="mt-8 space-y-4" aria-label="Ingresar">
        <div className="space-y-1.5">
          <Label htmlFor="email">Email</Label>
          <Input id="email" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="clave">Contraseña</Label>
          <Input id="clave" type="password" autoComplete="current-password" required value={clave} onChange={(e) => setClave(e.target.value)} />
        </div>
        {error && <p role="alert" className="text-sm text-error">{error}</p>}
        <Button type="submit" className="objetivo-tactil w-full" disabled={enviando}>Ingresar</Button>
      </form>
      <p className="mt-6 text-sm text-texto-2">
        ¿No tenés cuenta? <Link href="/registro/" className="underline underline-offset-4">Registrate</Link>
      </p>
      <p className="mt-10 text-xs text-tenue">
        El conteo es de {TEXTOS.golpes}: el detector encuentra menos golpes que los reales (recall medido 0,456) y no
        sabe si un golpe conectó.
      </p>
    </main>
  );
}
