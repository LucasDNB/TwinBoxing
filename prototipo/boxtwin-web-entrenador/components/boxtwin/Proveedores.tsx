"use client";

/*
 * Lo que envuelve a toda la app en el cliente.
 *
 * MotionConfig con reducedMotion="user": con prefers-reduced-motion no hay desplazamientos
 * ni conteos animados (03 §7). El QueryClient es uno por pestana: todo dato del servidor
 * pasa por TanStack Query (02 §5).
 */
import { useState, type ReactNode } from "react";
import { MotionConfig } from "motion/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/sonner";
import { ErrorApi } from "@/lib/api/cliente";

export function crearClienteDeConsultas() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // Un 401 o un 404 no se arreglan reintentando; un corte de red puede que si.
        retry: (intentos, error) =>
          !(error instanceof ErrorApi && error.estado >= 400 && error.estado < 500) &&
          intentos < 2,
        refetchOnWindowFocus: false,
        staleTime: 5_000,
      },
    },
  });
}

export function Proveedores({ children }: { children: ReactNode }) {
  const [cliente] = useState(crearClienteDeConsultas);
  return (
    <MotionConfig reducedMotion="user" transition={{ duration: 0.2, ease: "easeOut" }}>
      <QueryClientProvider client={cliente}>
        {children}
        <Toaster position="top-center" />
      </QueryClientProvider>
    </MotionConfig>
  );
}
