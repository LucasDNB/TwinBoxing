/*
 * La marca de un peleador. A es rojo y cuadrado; B es azul y circulo (03 §1): la forma y la
 * letra van siempre junto al color, para que la distincion no dependa solo del tono.
 *
 * Contorno y letra en el color de trazo, sobre fondo oscuro: la letra clara sobre el relleno
 * no llega a 4,5:1 (medido en W1, scripts/contraste.mjs).
 */
import { cn } from "@/lib/utils";

export function Esquina({ lado, className, tamano = 20 }: { lado: "A" | "B"; className?: string; tamano?: number }) {
  const a = lado === "A";
  return (
    <span
      aria-label={`peleador ${lado}`}
      role="img"
      className={cn(
        "inline-flex shrink-0 items-center justify-center border-2 bg-superficie-2 font-mono font-medium leading-none",
        a ? "rounded-sm border-a-trazo text-a-trazo" : "rounded-full border-b-trazo text-b-trazo",
        className,
      )}
      style={{ width: tamano, height: tamano, fontSize: Math.round(tamano * 0.55) }}
    >
      {lado}
    </span>
  );
}

/* "A · Ríos" o "A · sin nombre" */
export function NombreEsquina({ lado, nombre }: { lado: "A" | "B"; nombre?: string | null }) {
  return (
    <span className="inline-flex min-w-0 items-center gap-2">
      <Esquina lado={lado} />
      <span className="truncate">{nombre || `Peleador ${lado}`}</span>
    </span>
  );
}
