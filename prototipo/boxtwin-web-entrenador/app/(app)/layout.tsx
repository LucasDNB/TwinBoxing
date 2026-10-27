"use client";

/*
 * Todo lo que pide sesion: guarda en el cliente y navegacion (02 §6, 03 §3).
 * Margen lateral 16 px en la base, 24 en md, 40 en lg; ancho maximo 1200 px desde xl.
 */
import type { ReactNode } from "react";
import { BarraInferior, BarraLateral } from "@/components/boxtwin/Navegacion";
import { Guarda, useUsuario } from "@/components/boxtwin/Guarda";

export default function LayoutApp({ children }: { children: ReactNode }) {
  const usuario = useUsuario();
  return (
    <Guarda>
      <BarraLateral usuario={usuario} />
      <div className="lg:pl-[232px]">
        <main
          id="contenido"
          className="mx-auto w-full max-w-[1200px] px-4 pb-28 pt-5 md:px-6 lg:px-10 lg:pb-12 lg:pt-8"
        >
          {children}
        </main>
      </div>
      <BarraInferior />
      <div aria-live="polite" id="avisos-vivos" className="sr-only" />
    </Guarda>
  );
}
