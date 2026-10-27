"use client";

/*
 * Un numero grande que cuenta hasta su valor cuando cambia (03 §7). Con menos movimiento
 * pedido aparece con su valor final. El lector de pantalla lee siempre el valor final, no
 * los intermedios.
 */
import { useReducedMotion } from "motion/react";
import { CountingNumber } from "@/components/animate-ui/primitives/texts/counting-number";
import { entero } from "@/lib/formato";

export function Numero({ valor, className }: { valor: number; className?: string }) {
  const quieto = useReducedMotion();
  if (quieto) return <span className={className}>{entero(valor)}</span>;
  return (
    <span className={className}>
      <CountingNumber number={valor} aria-hidden="true" initiallyStable={false} />
      <span className="sr-only">{entero(valor)}</span>
    </span>
  );
}
