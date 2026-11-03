"use client";
/*
 * Espacio publicitario reservado (spec 9): un recuadro rotulado "Publicidad", sin red de
 * anuncios. Registra una impresion por montaje. Solo se muestra en free.
 */
import { useEffect } from "react";
import { useYo } from "@/components/boxtwinz/Marco";
import { api } from "@/lib/api";

export function EspacioPublicitario({ ubicacion, plan }: { ubicacion: string; plan: string }) {
  useEffect(() => {
    if (plan !== "free") return;
    api("/publicidad/impresion", { method: "POST", json: { ubicacion } }).catch(() => undefined);
  }, [ubicacion, plan]);
  if (plan !== "free") return null;
  return (
    <aside aria-label="Publicidad" className="my-4 flex h-24 items-center justify-center rounded-md border border-dashed border-borde-fuerte bg-superficie-2">
      <span className="rotulo">Publicidad</span>
    </aside>
  );
}

/* Igual, leyendo el plan del Marco: para usar adentro de una pantalla con sesion. */
export function PublicidadDelPlan({ ubicacion }: { ubicacion: string }) {
  const { yo } = useYo();
  return <EspacioPublicitario ubicacion={ubicacion} plan={yo.plan} />;
}
