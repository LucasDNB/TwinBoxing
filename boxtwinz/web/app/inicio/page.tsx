"use client";
/* Inicio: las 4 ranuras, "Subir sesion", ultimas sesiones y el espacio publicitario en free. */
import Link from "next/link";
import { useEffect, useState } from "react";
import { EspacioPublicitario } from "@/components/boxtwinz/EspacioPublicitario";
import { Marco, useYo } from "@/components/boxtwinz/Marco";
import { api, type Perfil, type SesionResumen } from "@/lib/api";
import { MODALIDAD } from "@/lib/textos";

const ESTADO: Record<string, string> = {
  subida: "En cola", procesando: "Procesando", esperando_siembra: "Marcá quién es quién",
  completando: "Contando golpes", lista: "Lista", error: "Error",
};

function Contenido() {
  const { yo } = useYo();
  const [perfiles, setPerfiles] = useState<Perfil[]>([]);
  const [sesiones, setSesiones] = useState<SesionResumen[]>([]);
  useEffect(() => {
    api<Perfil[]>("/perfiles").then(setPerfiles).catch(() => undefined);
    api<SesionResumen[]>("/sesiones").then(setSesiones).catch(() => undefined);
  }, []);
  return (
    <>
      <Link href="/subir/" className="objetivo-tactil flex w-full items-center justify-center rounded-lg bg-primary py-4 font-display text-2xl font-semibold text-primary-foreground">
        Subir sesión
      </Link>
      <h2 className="rotulo mt-6 mb-2">Perfiles</h2>
      <ul className="grid grid-cols-2 gap-3">
        {perfiles.map((p) => (
          <li key={p.id}>
            <Link href={`/perfil/?id=${p.id}`} className="block h-full rounded-lg border border-borde bg-superficie p-3">
              <span className="rotulo">{p.ranura === 1 ? "Tu perfil" : `Ranura ${p.ranura}`}</span>
              <span className="mt-1 block font-medium">{p.nombre}</span>
              <span className="block text-sm text-texto-3 capitalize">{p.ranura === 1 ? p.guardia : p.estilo}</span>
              {p.rotulo && <span className="mt-2 inline-block rounded-sm border border-estimado-borde bg-estimado-fondo px-1.5 text-xs text-estimado">{p.rotulo}</span>}
            </Link>
          </li>
        ))}
      </ul>
      <h2 className="rotulo mt-6 mb-2">Últimas sesiones</h2>
      {sesiones.length === 0 && <p className="text-texto-3">Todavía no subiste ninguna sesión.</p>}
      <ul className="divide-y divide-borde-suave rounded-lg border border-borde">
        {sesiones.slice(0, 8).map((s) => (
          <li key={s.id}>
            <Link href={s.estado === "esperando_siembra" ? `/siembra/?id=${s.id}` : `/sesion/?id=${s.id}`} className="objetivo-tactil flex items-center justify-between px-3 py-2">
              <span>{MODALIDAD[s.modalidad]} <span className="dato text-tenue">{new Date(s.creado).toLocaleDateString("es-AR")}</span></span>
              <span className="text-sm text-texto-3">{ESTADO[s.estado] ?? s.estado}</span>
            </Link>
          </li>
        ))}
      </ul>
      <EspacioPublicitario ubicacion="lista_sesiones" plan={yo.plan} />
    </>
  );
}

export default function Inicio() {
  return (
    <Marco titulo="Inicio">
      <Contenido />
    </Marco>
  );
}
