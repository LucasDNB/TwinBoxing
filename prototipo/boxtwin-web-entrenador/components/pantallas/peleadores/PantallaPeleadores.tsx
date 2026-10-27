"use client";

/*
 * Lista de peleadores (04, "Lista de peleadores"): nombre, categoria, guardia y sesiones.
 * "Agregar" abre la hoja con nombre, categoria, guardia, altura, envergadura y notas. Tocar
 * uno abre su perfil; "Editar" abre la misma hoja con sus datos.
 *
 * En Gimnasio (versiones 26-10, 03 §G-07, W33) se titula "Plantel": cada atleta suma el
 * entrenador a cargo y su consentimiento, con filtros "A mi cargo" y "Falta consentimiento".
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
import { ToggleGroup, ToggleGroupItem } from "@/components/animate-ui/components/radix/toggle-group";
import { InsigniaConsentimiento } from "@/components/pantallas/peleadores/Consentimiento";
import { esGimnasio, useCuenta } from "@/lib/cuenta";

type Filtro = "todos" | "a_cargo" | "falta";

export function PantallaPeleadores() {
  const boxeadores = useBoxeadores();
  const [hoja, setHoja] = useState<{ abierta: boolean; inicial: Boxeador | null }>({ abierta: false, inicial: null });
  const cuenta = useCuenta().data;
  const gimnasio = esGimnasio(cuenta);
  const [filtro, setFiltro] = useState<Filtro>("todos");
  const yo = cuenta?.usuario.id;
  const lista = (boxeadores.data ?? []).filter((b) =>
    filtro === "a_cargo" ? !!yo && b.entrenador?.id === yo
      : filtro === "falta" ? b.consentimiento?.estado !== "vigente"
        : true);

  const agregar = (
    <Button className="min-h-11" onClick={() => setHoja({ abierta: true, inicial: null })}>
      <Plus size={16} /> Agregar
    </Button>
  );

  return (
    <>
      <Encabezado titulo={gimnasio ? "Plantel" : "Peleadores"} accion={agregar}>
        {gimnasio && (
          <ToggleGroup type="single" value={filtro} onValueChange={(v) => v && setFiltro(v as Filtro)} aria-label="Filtrar el plantel" className="flex w-max flex-wrap gap-1">
            <ToggleGroupItem value="todos" className="min-h-11 border border-borde px-3">Todos</ToggleGroupItem>
            <ToggleGroupItem value="a_cargo" className="min-h-11 border border-borde px-3">A mi cargo</ToggleGroupItem>
            <ToggleGroupItem value="falta" className="min-h-11 border border-borde px-3">Falta consentimiento</ToggleGroupItem>
          </ToggleGroup>
        )}
      </Encabezado>
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
            {lista.map((b) => (
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
                  {gimnasio && (
                    <span className="mt-1 flex flex-wrap items-center gap-2 text-xs text-texto-3">
                      <InsigniaConsentimiento c={b.consentimiento} />
                      {b.entrenador?.nombre ? `a cargo de ${b.entrenador.nombre}` : "sin entrenador a cargo"}
                    </span>
                  )}
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
        <HojaPeleador key={hoja.inicial?.id ?? "nuevo"} abierta inicial={hoja.inicial} onCerrar={() => setHoja({ abierta: false, inicial: null })} gimnasio={gimnasio} />
      )}
    </>
  );
}
