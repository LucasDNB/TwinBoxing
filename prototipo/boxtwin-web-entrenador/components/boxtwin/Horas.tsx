"use client";

/*
 * Horas de video del mes contra las incluidas (versiones 26-10, B20). Sale de cuenta.horas o
 * de panel.horas: ningun numero de aca es una constante (CA-I01.1). En Individual y Gimnasio
 * es el "01.plan" que D4 habia quitado; vuelve porque es el limite de cada version.
 */
import type { Horas as TipoHoras } from "@/lib/api/esquemas";
import { textos } from "@/lib/textos";

const MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
  "septiembre", "octubre", "noviembre", "diciembre"];

export function nombreDeMes(mes: string): string {
  const m = Number(mes.split("-")[1]);
  return MESES[m - 1] ?? mes;
}

export function Horas({ horas, className }: { horas: TipoHoras; className?: string }) {
  const fraccion = horas.incluidas > 0 ? Math.min(horas.usadas / horas.incluidas, 1) : 0;
  const excedido = horas.usadas > horas.incluidas;
  return (
    <section aria-label="Horas del mes" className={`rounded-md border border-borde bg-superficie p-4 ${className ?? ""}`}>
      <p className="rotulo">Horas del mes</p>
      <p className="mt-1 text-sm" data-testid="horas-uso">
        {textos.cupoUso(horas.usadas, horas.incluidas, nombreDeMes(horas.mes))}
      </p>
      <div
        role="meter"
        aria-label="Horas usadas del mes"
        aria-valuemin={0}
        aria-valuemax={horas.incluidas}
        aria-valuenow={horas.usadas}
        className="mt-2 h-2 w-full overflow-hidden rounded-full bg-elevado"
      >
        <div className={`h-full ${excedido ? "bg-error" : "bg-texto"}`} style={{ width: `${fraccion * 100}%` }} />
      </div>
      {excedido && horas.modo === "informar" && (
        <p className="mt-2 text-xs text-estimado">{textos.cupoExcedidoInformar(horas.incluidas)}</p>
      )}
      {horas.sin_duracion > 0 && (
        <p className="mt-2 text-xs text-texto-3">
          {horas.sin_duracion === 1 ? "Una sesión todavía no tiene" : `${horas.sin_duracion} sesiones todavía no tienen`} duración medida y no suman.
        </p>
      )}
    </section>
  );
}
