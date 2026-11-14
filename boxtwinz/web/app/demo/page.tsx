"use client";
/*
 * Selector de usuarios de la demo: se llega desde "Probar demo" con la contraseña validada.
 * Usuarios estándar (Fajador, Estilista, Zurdo) más los creados acá, todos amigos entre sí.
 * Sin la cookie de demo la API responde 401 y se vuelve a la bienvenida.
 */
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api, ErrorApi, type UsuarioDemo } from "@/lib/api";

const ESTILOS = [
  { valor: "", texto: "No definido" },
  { valor: "fajador", texto: "Fajador" },
  { valor: "estilista", texto: "Estilista" },
  { valor: "hibrido", texto: "Híbrido" },
] as const;

const textoEstilo = (e: UsuarioDemo["estilo"]) => ESTILOS.find((x) => x.valor === (e ?? ""))?.texto ?? "No definido";
const textoGuardia = (g: UsuarioDemo["guardia"]) => (g === "zurda" ? "Zurda" : "Ortodoxa");
const textoPeso = (p: number | null) => (p == null ? "" : ` · ${p.toLocaleString("es-AR")} kg`);

export default function SelectorDemo() {
  const router = useRouter();
  const [usuarios, setUsuarios] = useState<UsuarioDemo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [f, setF] = useState({ nombre: "", estilo: "", guardia: "ortodoxa", peso: "" });
  const [ocupado, setOcupado] = useState(false);

  const cargar = useCallback(() => {
    api<UsuarioDemo[]>("/demo/usuarios").then(setUsuarios).catch((e) => {
      if (e instanceof ErrorApi && (e.status === 401 || e.status === 404)) router.replace("/");
      else setError(e instanceof ErrorApi ? e.message : "No se pudo cargar la demo");
    });
  }, [router]);
  useEffect(cargar, [cargar]);

  async function ingresar(id: string) {
    setOcupado(true);
    setError(null);
    try {
      await api(`/demo/usuarios/${id}/ingresar`, { method: "POST" });
      router.push("/inicio/");
    } catch (e) {
      setError(e instanceof ErrorApi ? e.message : "No se pudo ingresar");
      setOcupado(false);
    }
  }

  async function crear(e: React.FormEvent) {
    e.preventDefault();
    setOcupado(true);
    setError(null);
    try {
      const u = await api<UsuarioDemo>("/demo/usuarios", {
        method: "POST",
        json: { nombre: f.nombre, estilo: f.estilo || null, guardia: f.guardia, peso_kg: Number(f.peso.replace(",", ".")) },
      });
      await ingresar(u.id);
    } catch (err) {
      setError(err instanceof ErrorApi ? err.message : "No se pudo crear el usuario");
      setOcupado(false);
    }
  }

  const estandar = usuarios?.filter((u) => u.estandar) ?? [];
  const creados = usuarios?.filter((u) => !u.estandar) ?? [];
  const lista = (items: UsuarioDemo[], etiqueta: string) => (
    <ul className="space-y-2" aria-label={etiqueta}>
      {items.map((u) => (
        <li key={u.id}>
          <button
            type="button"
            disabled={ocupado}
            onClick={() => ingresar(u.id)}
            className="objetivo-tactil flex w-full flex-col items-start rounded-md border border-borde px-4 py-3 text-left hover:bg-accent disabled:opacity-50"
            aria-label={`Entrar como ${u.nombre}`}
          >
            <span className="font-medium">{u.nombre}</span>
            <span className="text-sm text-texto-2">{textoEstilo(u.estilo)} · {textoGuardia(u.guardia)}{textoPeso(u.peso_kg)}</span>
          </button>
        </li>
      ))}
    </ul>
  );

  return (
    <main className="mx-auto max-w-sm px-4 py-10">
      <h1 className="font-display text-4xl font-semibold">Elegí un usuario</h1>
      <p className="mt-2 text-sm text-texto-2">
        Demo de BoxTwin<span className="text-a-trazo">Z</span>. Todos los usuarios de la demo son amigos entre sí: se comparan
        y aparecen en el ranking. Los tres estándar traen sesiones sintéticas, no mediciones.
      </p>
      {error && <p role="alert" className="mt-4 text-sm text-error">{error}</p>}
      {!usuarios ? (
        <p className="rotulo mt-6" aria-busy="true">Cargando…</p>
      ) : (
        <div className="mt-6 space-y-6">
          <section className="space-y-2">
            <h2 className="text-lg font-medium">Usuarios estándar</h2>
            {lista(estandar, "Usuarios estándar")}
          </section>
          {creados.length > 0 && (
            <section className="space-y-2">
              <h2 className="text-lg font-medium">Usuarios creados</h2>
              {lista(creados, "Usuarios creados")}
            </section>
          )}
          <section className="space-y-3 border-t border-borde-suave pt-6">
            <h2 className="text-lg font-medium">Crear usuario rápido</h2>
            <form onSubmit={crear} className="space-y-4" aria-label="Crear usuario rápido">
              <div className="space-y-1.5">
                <Label htmlFor="demo-nombre">Nombre</Label>
                <Input id="demo-nombre" required maxLength={60} value={f.nombre} onChange={(e) => setF({ ...f, nombre: e.target.value })} />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="demo-estilo">Estilo</Label>
                <select
                  id="demo-estilo"
                  value={f.estilo}
                  onChange={(e) => setF({ ...f, estilo: e.target.value })}
                  className="objetivo-tactil w-full rounded-md border border-input bg-transparent px-3 text-sm"
                >
                  {ESTILOS.map((x) => <option key={x.valor} value={x.valor}>{x.texto}</option>)}
                </select>
              </div>
              <fieldset className="space-y-2">
                <legend className="text-sm font-medium">Guardia</legend>
                {(["ortodoxa", "zurda"] as const).map((g) => (
                  <label key={g} className="objetivo-tactil mr-4 inline-flex items-center gap-2">
                    <input type="radio" name="demo-guardia" value={g} checked={f.guardia === g} onChange={() => setF({ ...f, guardia: g })} />
                    {textoGuardia(g)}
                  </label>
                ))}
              </fieldset>
              <div className="space-y-1.5">
                <Label htmlFor="demo-peso">Peso (kg)</Label>
                <Input id="demo-peso" type="number" inputMode="decimal" min={30} max={200} step="0.1" required value={f.peso} onChange={(e) => setF({ ...f, peso: e.target.value })} />
              </div>
              <Button type="submit" className="objetivo-tactil w-full" disabled={ocupado}>Crear y entrar</Button>
            </form>
          </section>
        </div>
      )}
      <p className="mt-8 text-sm text-texto-2"><Link href="/" className="underline underline-offset-4">Volver al ingreso</Link></p>
    </main>
  );
}
