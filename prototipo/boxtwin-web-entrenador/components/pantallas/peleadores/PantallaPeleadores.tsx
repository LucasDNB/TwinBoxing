"use client";

/*
 * Lista de peleadores (04, "Lista de peleadores"): nombre, categoria, guardia y sesiones.
 * "Agregar" abre la hoja con nombre, categoria, guardia, altura, envergadura y notas. Tocar
 * uno abre su perfil; "Editar" abre la misma hoja con sus datos.
 */
import { useState } from "react";
import Link from "next/link";
import { Encabezado } from "@/components/boxtwin/Encabezado";
import { ErrorConReintento, Esqueleto, Vacio } from "@/components/boxtwin/Estados";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { Plus } from "@/components/animate-ui/icons/plus";
import { Fade } from "@/components/animate-ui/primitives/effects/fade";
import { HojaPeleador } from "@/components/pantallas/peleadores/HojaPeleador";
import { useBoxeadores } from "@/lib/api/consultas";
import type { Boxeador } from "@/lib/api/esquemas";
import { decimal, fechaCorta } from "@/lib/formato";

export function PantallaPeleadores() {
  const boxeadores = useBoxeadores();
  const [hoja, setHoja] = useState<{ abierta: boolean; inicial: Boxeador | null }>({ abierta: false, inicial: null });

  const agregar = (
    <Button className="min-h-11" onClick={() => setHoja({ abierta: true, inicial: null })}>
      <Plus size={16} /> Agregar
    </Button>
  );

  return (
    <>
      <Encabezado titulo="Peleadores" accion={agregar} />
      {boxeadores.isPending ? (
        <Esqueleto />
      ) : boxeadores.isError ? (
        <ErrorConReintento error={boxeadores.error} alReintentar={() => boxeadores.refetch()} />
      ) : boxeadores.data.length === 0 ? (
        <Vacio titulo="Cargá a tus peleadores para armar sus perfiles" accion={agregar}>
          Cada sesión suma al perfil de los peleadores que marques en ella.
        </Vacio>
      ) : (
        <Fade>
          <ul className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {boxeadores.data.map((b) => (
              <li key={b.id} className="flex items-stretch rounded-md border border-borde bg-superficie">
                <Link href={`/peleador/?id=${b.id}`} className="min-h-16 flex-1 px-4 py-3">
                  <span className="block font-display text-lg font-semibold">{b.nombre}</span>
                  <span className="block text-sm text-texto-3">
                    {[b.categoria, b.guardia, b.altura_m ? `${decimal(b.altura_m, 2)} m` : null].filter(Boolean).join(" · ") || "sin datos cargados"}
                  </span>
                  <span className="block text-xs text-texto-3">
                    {b.sesiones ?? 0} {b.sesiones === 1 ? "sesión" : "sesiones"}
                    {b.ultima_sesion ? ` · última ${fechaCorta(b.ultima_sesion)}` : ""}
                  </span>
                </Link>
                <button
                  type="button"
                  onClick={() => setHoja({ abierta: true, inicial: b })}
                  className="min-h-11 min-w-16 border-l border-borde-suave px-3 text-sm text-texto-2 underline underline-offset-4"
                  aria-label={`Editar a ${b.nombre}`}
                >
                  Editar
                </button>
              </li>
            ))}
          </ul>
        </Fade>
      )}
      {hoja.abierta && (
        <HojaPeleador key={hoja.inicial?.id ?? "nuevo"} abierta inicial={hoja.inicial} onCerrar={() => setHoja({ abierta: false, inicial: null })} />
      )}
    </>
  );
}
