/*
 * El margen del detector, en lenguaje llano, cerca de todo numero de golpes (03 §1, punto 4).
 * Sale del documento: si el detector mejora, la pantalla mejora sola (CA-05.3).
 */
import { textos } from "@/lib/textos";
import { cn } from "@/lib/utils";

export function Margen({ precision, recall, className }: { precision: number; recall: number; className?: string }) {
  return (
    <p className={cn("rounded-md border-l-2 border-estimado-borde bg-estimado-fondo px-3 py-2 text-sm text-texto-2", className)}>
      <span className="font-medium text-texto">Detectados: </span>
      {textos.margenDetector(precision, recall)}
    </p>
  );
}
