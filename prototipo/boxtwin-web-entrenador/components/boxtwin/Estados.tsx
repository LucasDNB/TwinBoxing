"use client";

/*
 * Los tres estados de carga que tiene toda pantalla (04, convenciones): esqueleto mientras
 * llega el dato, error con "Reintentar" y el mensaje de la API, y vacio con la accion que lo
 * resuelve.
 */
import type { ReactNode } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { ErrorApi, ErrorDeContrato } from "@/lib/api/cliente";

export function Esqueleto({ filas = 3 }: { filas?: number }) {
  return (
    <div aria-busy="true" aria-label="Cargando" className="flex flex-col gap-3">
      <Skeleton className="h-8 w-40 bg-elevado" />
      {Array.from({ length: filas }).map((_, i) => (
        <Skeleton key={i} className="h-20 w-full bg-superficie" />
      ))}
    </div>
  );
}

export function mensajeDeError(e: unknown): string {
  if (e instanceof ErrorDeContrato) return e.message;
  if (e instanceof ErrorApi) return e.message;
  if (e instanceof Error) return e.message;
  return "algo salió mal";
}

export function ErrorConReintento({ error, alReintentar, titulo = "No se pudo cargar" }: { error: unknown; alReintentar?: () => void; titulo?: string }) {
  return (
    <div role="alert" className="rounded-md border border-error-borde bg-error-fondo p-4">
      <p className="font-medium text-error">{titulo}</p>
      <p className="mt-1 text-sm text-texto-2">{mensajeDeError(error)}</p>
      {alReintentar && (
        <Button variant="outline" className="mt-3 min-h-11" onClick={alReintentar}>
          Reintentar
        </Button>
      )}
    </div>
  );
}

export function Vacio({ titulo, children, accion }: { titulo: string; children?: ReactNode; accion?: ReactNode }) {
  return (
    <div className="rounded-md border border-dashed border-borde-fuerte bg-superficie p-5">
      <p className="font-display text-xl font-semibold">{titulo}</p>
      {children && <div className="mt-1 text-sm text-texto-2">{children}</div>}
      {accion && <div className="mt-4">{accion}</div>}
    </div>
  );
}
