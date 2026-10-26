"use client";

/*
 * Todo lo que pide sesion: guarda en el cliente y navegacion (02 §6, 03 §3).
 * Margen lateral 16 px en la base, 24 en md, 40 en lg; ancho maximo 1200 px desde xl.
 *
 * Versiones 26-10 (03 §1): la guarda espera tambien a la cuenta, porque la navegacion y el
 * inicio dependen de la version. Mientras tanto, esqueleto.
 */
import type { ReactNode } from "react";
import { BarraInferior, BarraLateral, FranjaDemo } from "@/components/boxtwin/Navegacion";
import { Guarda, useUsuario } from "@/components/boxtwin/Guarda";
import { ErrorConReintento, Esqueleto } from "@/components/boxtwin/Estados";
import { useCuenta } from "@/lib/cuenta";

export default function LayoutApp({ children }: { children: ReactNode }) {
  return (
    <Guarda>
      <ConCuenta>{children}</ConCuenta>
    </Guarda>
  );
}

function ConCuenta({ children }: { children: ReactNode }) {
  const usuario = useUsuario();
  const cuenta = useCuenta();
  if (cuenta.isPending) {
    return <main className="mx-auto w-full max-w-[1200px] px-4 pt-5"><Esqueleto filas={4} /></main>;
  }
  if (cuenta.isError) {
    return (
      <main className="mx-auto w-full max-w-[1200px] px-4 pt-5">
        <ErrorConReintento error={cuenta.error} alReintentar={() => cuenta.refetch()} />
      </main>
    );
  }
  return (
    <>
      <FranjaDemo cuenta={cuenta.data} />
      <BarraLateral usuario={usuario} cuenta={cuenta.data} />
      <div className="lg:pl-[232px]">
        <main
          id="contenido"
          className="mx-auto w-full max-w-[1200px] px-4 pb-28 pt-5 md:px-6 lg:px-10 lg:pb-12 lg:pt-8"
        >
          {children}
        </main>
      </div>
      <BarraInferior cuenta={cuenta.data} />
      <div aria-live="polite" id="avisos-vivos" className="sr-only" />
    </>
  );
}
