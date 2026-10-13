"use client";

/*
 * Lista de sesiones (04, "Lista de sesiones"): la misma tarjeta del panel, con filtro por
 * estado y por contexto y busqueda por nombre de peleador. Mas reciente primero. El filtro de
 * estado va en la query (?estado=) para que el panel pueda enlazar "Ver las N pendientes".
 */
import { useMemo, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Encabezado } from "@/components/boxtwin/Encabezado";
import { ErrorConReintento, Esqueleto, Vacio } from "@/components/boxtwin/Estados";
import { Fade } from "@/components/animate-ui/primitives/effects/fade";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/animate-ui/components/radix/toggle-group";
import { Input } from "@/components/ui/input";
import { TarjetaSesion } from "@/components/pantallas/sesiones/TarjetaSesion";
import { EN_CURSO, useSesiones } from "@/lib/api/consultas";
import { CONTEXTOS } from "@/lib/api/esquemas";
import { NOMBRE_CONTEXTO } from "@/lib/textos";

const FILTROS_ESTADO = [
  { valor: "todas", nombre: "Todas" },
  { valor: "espera_siembra", nombre: "Falta un paso" },
  { valor: "en_curso", nombre: "En curso" },
  { valor: "listo", nombre: "Listas" },
  { valor: "fallo", nombre: "Fallaron" },
];

export function PantallaSesiones() {
  const router = useRouter();
  const q = useSearchParams();
  const estado = q.get("estado") ?? "todas";
  const [contexto, setContexto] = useState("todos");
  const [busqueda, setBusqueda] = useState("");
  const sesiones = useSesiones();

  const filtradas = useMemo(() => {
    const t = busqueda.trim().toLowerCase();
    return (sesiones.data ?? []).filter((s) => {
      if (estado === "en_curso" ? !EN_CURSO.includes(s.estado) : estado !== "todas" && s.estado !== estado) return false;
      if (contexto !== "todos" && (s.contexto ?? "sparring") !== contexto) return false;
      if (t) {
        const nombres = [s.boxeadores?.A?.nombre, s.boxeadores?.B?.nombre, s.nombre].filter(Boolean).join(" ").toLowerCase();
        if (!nombres.includes(t)) return false;
      }
      return true;
    });
  }, [sesiones.data, estado, contexto, busqueda]);

  return (
    <>
      <Encabezado
        titulo="Sesiones"
        accion={<Button asChild className="min-h-11"><Link href="/sesiones/nueva/">Subir</Link></Button>}
      />
      <div className="mb-4 flex flex-col gap-3">
        <label className="sr-only" htmlFor="buscar">Buscar por peleador</label>
        <Input id="buscar" type="search" placeholder="Buscar por peleador" value={busqueda} onChange={(e) => setBusqueda(e.target.value)} className="h-11 text-base" />
        <div className="-mx-4 overflow-x-auto px-4">
          <ToggleGroup type="single" value={estado} aria-label="Filtrar por estado"
            onValueChange={(v) => v && router.replace(v === "todas" ? "/sesiones/" : `/sesiones/?estado=${v}`)}
            className="flex w-max gap-1">
            {FILTROS_ESTADO.map((f) => (
              <ToggleGroupItem key={f.valor} value={f.valor} className="min-h-11 border border-borde px-3 whitespace-nowrap">{f.nombre}</ToggleGroupItem>
            ))}
          </ToggleGroup>
        </div>
        <div className="-mx-4 overflow-x-auto px-4">
          <ToggleGroup type="single" value={contexto} aria-label="Filtrar por contexto" onValueChange={(v) => v && setContexto(v)} className="flex w-max gap-1">
            <ToggleGroupItem value="todos" className="min-h-11 border border-borde px-3">Todos</ToggleGroupItem>
            {CONTEXTOS.map((c) => (
              <ToggleGroupItem key={c} value={c} className="min-h-11 border border-borde px-3">{NOMBRE_CONTEXTO[c]}</ToggleGroupItem>
            ))}
          </ToggleGroup>
        </div>
      </div>
      {sesiones.isPending ? (
        <Esqueleto />
      ) : sesiones.isError ? (
        <ErrorConReintento error={sesiones.error} alReintentar={() => sesiones.refetch()} />
      ) : (sesiones.data ?? []).length === 0 ? (
        <Vacio titulo="Todavía no hay sesiones" accion={<Button asChild className="min-h-11"><Link href="/sesiones/nueva/">Subir la primera</Link></Button>} />
      ) : filtradas.length === 0 ? (
        <Vacio titulo="Ninguna sesión coincide con el filtro" />
      ) : (
        <Fade className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {filtradas.map((s) => <TarjetaSesion key={s.id} s={s} />)}
        </Fade>
      )}
    </>
  );
}
