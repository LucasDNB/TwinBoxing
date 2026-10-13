"use client";

/*
 * Controles de formulario con el objetivo tactil de 44 px y el texto de 16 px en campos
 * (03 §1). La casilla usa el checkbox de Animate UI con su <label> real (03 §8).
 */
import { useId, type ReactNode, type SelectHTMLAttributes } from "react";
import { Checkbox } from "@/components/animate-ui/components/radix/checkbox";
import { cn } from "@/lib/utils";

export function Casilla({
  checked, onChange, children, id, descripcion, className,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  children: ReactNode;
  id?: string;
  descripcion?: ReactNode;
  className?: string;
}) {
  const auto = useId();
  const cid = id ?? auto;
  return (
    <div className={cn("flex min-h-11 items-start gap-3 py-1", className)}>
      <Checkbox
        id={cid}
        checked={checked}
        onCheckedChange={(v) => onChange(v === true)}
        className="mt-0.5 size-5 border-borde-fuerte"
      />
      <label htmlFor={cid} className="flex-1 cursor-pointer text-[15px] leading-snug">
        {children}
        {descripcion && <span className="mt-0.5 block text-sm text-texto-3">{descripcion}</span>}
      </label>
    </div>
  );
}

/*
 * <select> nativo estilizado: en el celular abre el selector del sistema, que es mas usable
 * que un menu propio (03 §4).
 */
export function SelectNativo({ className, children, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...props}
      className={cn(
        "h-11 w-full appearance-none rounded-md border border-borde-fuerte bg-superficie-2 px-3 pr-9 text-base text-texto",
        "bg-[url('data:image/svg+xml;utf8,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%2212%22 height=%228%22><path d=%22M1 1l5 5 5-5%22 stroke=%22%23A3ACB4%22 fill=%22none%22 stroke-width=%221.6%22/></svg>')] bg-[length:12px_8px] bg-[position:right_12px_center] bg-no-repeat",
        className,
      )}
    >
      {children}
    </select>
  );
}

/*
 * Un paso del formulario de 04 §02: numerado, plegable cuando esta completo. Plegado muestra
 * el resumen y "Cambiar".
 */
export function Paso({
  numero, titulo, completo, abierto, onAbrir, resumen, children,
}: {
  numero: number;
  titulo: string;
  completo: boolean;
  abierto: boolean;
  onAbrir: () => void;
  resumen?: ReactNode;
  children: ReactNode;
}) {
  const plegado = completo && !abierto;
  return (
    <section aria-labelledby={`paso-${numero}`} className="rounded-md border border-borde bg-superficie p-4">
      <div className="flex items-center justify-between gap-3">
        <h2 id={`paso-${numero}`} className="flex items-center gap-2 font-display text-xl font-semibold lg:text-2xl">
          <span
            aria-hidden="true"
            className={cn(
              "inline-flex size-6 items-center justify-center rounded-full border font-mono text-xs",
              completo ? "border-ok-borde bg-ok-fondo text-ok" : "border-borde-fuerte text-tenue",
            )}
          >
            {completo ? "✓" : numero}
          </span>
          {titulo}
        </h2>
        {plegado && (
          <button type="button" onClick={onAbrir} className="min-h-11 px-2 text-sm text-texto-2 underline underline-offset-4">
            Cambiar
          </button>
        )}
      </div>
      {plegado ? (
        <div className="mt-1 pl-8 text-sm text-texto-2">{resumen}</div>
      ) : (
        <div className="mt-4">{children}</div>
      )}
    </section>
  );
}
