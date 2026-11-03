"use client";
/*
 * El marco de las pantallas con sesion: cabecera con la marca y navegacion inferior, mobile
 * first (375 px). Si la API dice 401 manda a la bienvenida.
 */
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api, ErrorApi, type Yo } from "@/lib/api";
import { cn } from "@/lib/utils";

const CtxYo = createContext<{ yo: Yo; recargar: () => void } | null>(null);
export function useYo() {
  const c = useContext(CtxYo);
  if (!c) throw new Error("useYo fuera del Marco");
  return c;
}

const NAV = [
  { href: "/inicio/", texto: "Inicio" },
  { href: "/subir/", texto: "Subir" },
  { href: "/rival/", texto: "Rival" },
  { href: "/ranking/", texto: "Ranking" },
  { href: "/ajustes/", texto: "Ajustes" },
];

export function Marco({ titulo, children }: { titulo: string; children: ReactNode }) {
  const [yo, setYo] = useState<Yo | null>(null);
  const [n, setN] = useState(0);
  const router = useRouter();
  const ruta = usePathname();
  useEffect(() => {
    api<Yo>("/yo").then(setYo).catch((e) => {
      if (e instanceof ErrorApi && e.status === 401) router.replace("/");
    });
  }, [router, n]);
  if (!yo) return <main className="p-4 rotulo" aria-busy="true">Cargando…</main>;
  return (
    <CtxYo.Provider value={{ yo, recargar: () => setN((x) => x + 1) }}>
      <div className="min-h-dvh pb-20">
        <header className="sticky top-0 z-10 flex items-center justify-between border-b border-borde-suave bg-nav/95 px-4 py-3 backdrop-blur">
          <Link href="/inicio/" className="font-display text-xl font-semibold tracking-wide">BoxTwin<span className="text-a-trazo">Z</span></Link>
          <span className="rotulo">{yo.plan === "premium" ? "Premium" : "Free"}{yo.rol === "admin" ? " · admin" : ""}</span>
        </header>
        <main className="mx-auto max-w-xl px-4 py-4">
          <h1 className="mb-4 font-display text-3xl font-semibold">{titulo}</h1>
          {children}
        </main>
        <nav aria-label="Principal" className="fixed inset-x-0 bottom-0 z-10 border-t border-borde-suave bg-nav pb-[env(safe-area-inset-bottom)]">
          <ul className="mx-auto flex max-w-xl justify-around">
            {NAV.map((x) => (
              <li key={x.href}>
                <Link href={x.href} className={cn("objetivo-tactil flex items-center px-2 text-sm", ruta?.startsWith(x.href.slice(0, -1)) ? "text-texto" : "text-tenue")}>
                  {x.texto}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </CtxYo.Provider>
  );
}
