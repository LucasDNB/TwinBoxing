"use client";

/*
 * Encabezado de pantalla (03 §3): titulo y una accion a la derecha; la flecha atras aparece
 * en las pantallas de detalle. Titulo display 700, 32 px en el celular y 46 px desde lg.
 */
import type { ReactNode } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft } from "@/components/animate-ui/icons/arrow-left";
import { AnimateIcon } from "@/components/animate-ui/icons/icon";
import { Rotulo } from "@/components/boxtwin/Rotulo";

export function Encabezado({
  titulo, rotulo, accion, atras, children,
}: {
  titulo: ReactNode;
  rotulo?: ReactNode;
  accion?: ReactNode;
  atras?: string;
  children?: ReactNode;
}) {
  const router = useRouter();
  return (
    <header className="mb-5 flex flex-col gap-2 lg:mb-8">
      {atras && (
        <AnimateIcon animateOnHover asChild>
          <button
            type="button"
            onClick={() => (window.history.length > 1 ? router.back() : router.push(atras))}
            className="-ml-2 inline-flex min-h-11 w-fit items-center gap-1 px-2 text-sm text-texto-2"
          >
            <ArrowLeft size={18} /> Volver
          </button>
        </AnimateIcon>
      )}
      <div className="flex items-end justify-between gap-3">
        <div className="min-w-0">
          {rotulo && <Rotulo as="p">{rotulo}</Rotulo>}
          <h1 className="font-display text-[32px] font-bold leading-none lg:text-[46px]">{titulo}</h1>
        </div>
        {accion && <div className="shrink-0">{accion}</div>}
      </div>
      {children}
    </header>
  );
}
