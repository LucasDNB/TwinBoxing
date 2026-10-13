import type { EstadoSesion } from "@/lib/api/esquemas";
import { NOMBRE_ESTADO } from "@/lib/textos";
import { cn } from "@/lib/utils";

const ESTILO: Record<string, string> = {
  listo: "border-ok-borde bg-ok-fondo text-ok",
  espera_siembra: "border-estimado-borde bg-estimado-fondo text-estimado",
  fallo: "border-error-borde bg-error-fondo text-error",
};

export function InsigniaEstado({ estado }: { estado: EstadoSesion }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-sm border px-2 py-0.5 font-mono text-[11px] uppercase tracking-[0.1em]",
        ESTILO[estado] ?? "border-curso-borde bg-curso-fondo text-curso",
      )}
    >
      {NOMBRE_ESTADO[estado] ?? estado}
    </span>
  );
}
