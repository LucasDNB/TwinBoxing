/*
 * El margen del detector, en lenguaje llano, cerca de todo numero de golpes (03 §1, punto 4).
 * Sale del documento: si el detector mejora, la pantalla mejora sola (CA-05.3).
 */
import { textos } from "@/lib/textos";
import { cn } from "@/lib/utils";

/*
 * B8b: si el documento dice en que contexto se midio y la sesion es de otro (bolsa, sombra),
 * el numero no se muestra: un "acierta 9 de cada 10" medido en sparring no vale para bolsa.
 */
export function Margen({ precision, recall, medidoEn, contexto, className }: {
  precision: number; recall: number; medidoEn?: string | null; contexto?: string | null; className?: string;
}) {
  const otro = medidoEn && contexto && medidoEn !== contexto;
  return (
    <p className={cn("rounded-md border-l-2 border-estimado-borde bg-estimado-fondo px-3 py-2 text-sm text-texto-2", className)}>
      <span className="font-medium text-texto">Detectados: </span>
      {otro ? textos.margenOtroContexto(medidoEn, contexto) : textos.margenDetector(precision, recall)}
    </p>
  );
}
