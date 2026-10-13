/*
 * Una caja con scroll horizontal propio (03 §3: las tablas anchas no mueven la pagina).
 * Toma foco y tiene nombre para que se pueda recorrer con teclado (WCAG 2.1.1; axe
 * scrollable-region-focusable, encontrado en el e2e).
 */
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Desplazable({ nombre, children, className }: { nombre: string; children: ReactNode; className?: string }) {
  return (
    <div role="region" aria-label={nombre} tabIndex={0} className={cn("overflow-x-auto rounded-sm", className)}>
      {children}
    </div>
  );
}
