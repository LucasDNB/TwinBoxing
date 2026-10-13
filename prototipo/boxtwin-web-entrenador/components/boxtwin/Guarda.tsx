"use client";

/*
 * La guarda de sesion (02 §6). Corre en el cliente: con exportacion estatica no hay
 * middleware de servidor. Mientras resuelve se ve el esqueleto, no una pantalla en blanco.
 *
 * El token se lee con useSyncExternalStore: en el prerender no hay localStorage (snapshot
 * del servidor null, se ve el esqueleto) y en el cliente se lee sin un render de mas.
 */
import { useEffect, useSyncExternalStore, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { leerToken, leerUsuario } from "@/lib/sesion-usuario";
import { Esqueleto } from "@/components/boxtwin/Estados";

function suscribir(f: () => void) {
  window.addEventListener("storage", f);
  return () => window.removeEventListener("storage", f);
}

export function useUsuario() {
  return useSyncExternalStore(suscribir, leerUsuario, () => "");
}

export function Guarda({ children }: { children: ReactNode }) {
  const router = useRouter();
  const token = useSyncExternalStore<string | null>(suscribir, leerToken, () => null);
  useEffect(() => {
    if (token !== "") return;
    const aca = window.location.pathname + window.location.search;
    router.replace(`/ingresar/?siguiente=${encodeURIComponent(aca)}`);
  }, [router, token]);
  if (!token) return <Esqueleto />;
  return <>{children}</>;
}
