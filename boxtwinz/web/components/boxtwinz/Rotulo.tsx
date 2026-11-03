/* Rotulo ambar para todo lo que es estimacion, simulado o sin validar. */
import type { ReactNode } from "react";

export function Rotulo({ children }: { children: ReactNode }) {
  return <span className="inline-block rounded-sm border border-estimado-borde bg-estimado-fondo px-1.5 py-0.5 text-xs text-estimado">{children}</span>;
}
