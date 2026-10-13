"use client";

/*
 * Envuelve cualquier bloque segun su estado (03 §6, 01 §4).
 *
 *   medido         numero en --color-texto, sin adornos; el margen va cerca
 *   estimacion     "≈", borde punteado ambar y rotulo "estimación", con la version del modelo
 *   apagado        condicionado sin su criterio: NO muestra numeros, dice que criterio falta
 *   no_disponible  plegado, con que pieza falta y por que
 *   en_construccion  se ve pero no se usa todavia; sin numeros
 *   fuera          no se dibuja
 *
 * Lo que decide el estado es lib/banderas.ts con /producto; este componente solo lo dibuja.
 * Con "apagado", "en_construccion" y "no_disponible" los hijos NO se renderizan: un numero condicionado no
 * puede llegar al DOM ni escondido (I3).
 */
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import type { EstadoBloque as Estado } from "@/lib/banderas";
import { Rotulo } from "@/components/boxtwin/Rotulo";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/animate-ui/components/radix/accordion";

interface Props {
  estado: Estado;
  titulo: string;
  children?: ReactNode;
  /* Texto del criterio que falta (apagado) o de la pieza que falta (no_disponible). */
  motivo?: string;
  /* Version y exactitud del modelo (estimacion). */
  nota?: ReactNode;
  className?: string;
  accion?: ReactNode;
}

export function EstadoBloque({ estado, titulo, children, motivo, nota, className, accion }: Props) {
  if (estado === "fuera") return null;

  if (estado === "no_disponible") {
    return (
      <Accordion type="single" collapsible className={cn("rounded-md border border-borde bg-superficie-2", className)}>
        <AccordionItem value="nd" className="border-0">
          <AccordionTrigger className="min-h-11 px-4 hover:no-underline">
            <span className="flex flex-col items-start gap-1 text-left">
              <Rotulo>no disponible en esta versión</Rotulo>
              <span className="font-display text-lg font-semibold text-texto-2">{titulo}</span>
            </span>
          </AccordionTrigger>
          <AccordionContent className="px-4 pb-4 text-sm text-texto-2">{motivo}</AccordionContent>
        </AccordionItem>
      </Accordion>
    );
  }

  if (estado === "en_construccion") {
    return (
      <section
        aria-label={titulo}
        data-estado="en_construccion"
        className={cn("rounded-md border border-dashed border-borde-fuerte bg-superficie-2 p-4", className)}
      >
        <Rotulo className="text-estimado">en construcción</Rotulo>
        <h3 className="mt-1 font-display text-lg font-semibold text-texto-2">{titulo}</h3>
        <p className="mt-1 text-sm text-texto-3">{motivo}</p>
      </section>
    );
  }

  if (estado === "apagado") {
    return (
      <section
        aria-label={titulo}
        data-estado="apagado"
        className={cn("rounded-md border border-borde-fuerte bg-superficie p-4", className)}
      >
        <Rotulo>en validación</Rotulo>
        <h3 className="mt-1 font-display text-lg font-semibold text-texto-2">{titulo}</h3>
        <p className="mt-1 text-sm text-texto-3">{motivo}</p>
      </section>
    );
  }

  const estimado = estado === "estimacion";
  return (
    <section
      aria-label={titulo}
      data-estado={estado}
      className={cn(
        "rounded-md border p-4",
        estimado ? "border-dashed border-estimado-borde bg-estimado-fondo/40" : "border-borde bg-superficie",
        className,
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          {estimado && <Rotulo className="text-estimado">estimación</Rotulo>}
          <h3 className="font-display text-xl font-semibold lg:text-2xl">{titulo}</h3>
        </div>
        {accion}
      </div>
      <div className="mt-3">{children}</div>
      {nota && <div className="mt-3 text-xs text-texto-3">{nota}</div>}
    </section>
  );
}

/* Un numero en estilo de estimacion: "≈ 12" en ambar. */
export function Estimado({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span className={cn("text-estimado", className)}>
      <span aria-hidden="true">≈ </span>
      <span className="sr-only">aproximadamente </span>
      {children}
    </span>
  );
}
