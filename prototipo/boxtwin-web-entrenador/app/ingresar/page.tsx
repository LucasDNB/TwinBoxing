"use client";

/*
 * 00 · Ingreso (04 §00). Una columna de 400 px maximo. "Tengo una invitacion" despliega el
 * codigo y cambia el boton a "Crear cuenta". Ningun error dice cual de los dos campos fallo.
 */
import { Suspense, useState, type FormEvent } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Marca } from "@/components/boxtwin/Marca";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ErrorApi } from "@/lib/api/cliente";
import { ingresar, registrarse, useProducto } from "@/lib/api/consultas";
import { guardarSesion } from "@/lib/sesion-usuario";
import { destinoSeguro } from "@/lib/navegacion";

function Formulario() {
  const router = useRouter();
  const siguiente = useSearchParams().get("siguiente");
  const producto = useProducto();
  const [email, setEmail] = useState("");
  const [clave, setClave] = useState("");
  const [invitacion, setInvitacion] = useState("");
  const [creando, setCreando] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [enviando, setEnviando] = useState(false);

  const registro = producto.data?.registro;

  async function enviar(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setEnviando(true);
    try {
      const t = creando
        ? await registrarse(email, clave, invitacion)
        : await ingresar(email, clave);
      guardarSesion(t.token, t.usuario);
      router.replace(destinoSeguro(siguiente));
    } catch (err) {
      if (err instanceof ErrorApi && err.estado === 401) setError("Email o clave incorrectos.");
      else if (err instanceof ErrorApi && err.estado === 403)
        setError(registro === "invitacion" && invitacion ? "El código de invitación no es correcto." : "El registro está cerrado: pedí una invitación.");
      else if (err instanceof ErrorApi && err.estado === 409) setError("Ya hay una cuenta con ese email.");
      else if (err instanceof ErrorApi && err.estado === 422) setError("Revisá el email; la clave tiene que tener 8 caracteres o más.");
      else setError(err instanceof Error ? err.message : "No se pudo entrar.");
    } finally {
      setEnviando(false);
    }
  }

  return (
    <main className="mx-auto flex min-h-dvh w-full max-w-[400px] flex-col justify-center px-4 py-10">
      <Marca />
      <h1 className="mt-6 font-display text-[32px] font-bold leading-none">
        {creando ? "Crear cuenta" : "Entrar"}
      </h1>
      <form onSubmit={enviar} className="mt-6 flex flex-col gap-4" noValidate>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="email">Email</Label>
          <Input id="email" type="email" autoComplete="email" inputMode="email" required
            value={email} onChange={(e) => setEmail(e.target.value)} className="h-11 text-base" />
        </div>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="clave">Clave</Label>
          <Input id="clave" type="password" autoComplete={creando ? "new-password" : "current-password"}
            required minLength={8} value={clave} onChange={(e) => setClave(e.target.value)} className="h-11 text-base" />
        </div>
        {creando && registro === "invitacion" && (
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="invitacion">Código de invitación</Label>
            <Input id="invitacion" autoComplete="off" value={invitacion}
              onChange={(e) => setInvitacion(e.target.value)} className="h-11 text-base" />
          </div>
        )}
        {error && (
          <p role="alert" className="rounded-md border border-error-borde bg-error-fondo px-3 py-2 text-sm text-error">
            {error}
          </p>
        )}
        <Button type="submit" disabled={enviando} className="min-h-11 text-base">
          {enviando ? "Un momento…" : creando ? "Crear cuenta" : "Entrar"}
        </Button>
      </form>
      {registro && registro !== "cerrado" && (
        <button type="button" className="mt-4 min-h-11 w-fit text-sm text-texto-2 underline underline-offset-4"
          onClick={() => { setCreando((c) => !c); setError(null); }}>
          {creando ? "Ya tengo cuenta" : registro === "invitacion" ? "Tengo una invitación" : "Crear una cuenta"}
        </button>
      )}
    </main>
  );
}

export default function Ingresar() {
  return (
    <Suspense>
      <Formulario />
    </Suspense>
  );
}
