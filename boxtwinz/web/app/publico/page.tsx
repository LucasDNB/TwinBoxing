"use client";
/* Perfil publico (spec 12): stats, "Usar como rival fantasma" y "Denunciar". */
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { Denunciar } from "@/components/boxtwinz/Denunciar";
import { Marco } from "@/components/boxtwinz/Marco";
import { VistaPublica, type Vista } from "@/components/boxtwinz/VistaPublica";
import { api, ErrorApi } from "@/lib/api";

function Contenido() {
  const id = useSearchParams().get("id") ?? "";
  const [v, setV] = useState<Vista | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api<Vista>(`/publicos/${id}`).then(setV).catch((e) => setError(e instanceof ErrorApi ? e.message : "Error")); }, [id]);
  if (error) return <p role="alert" className="text-error">{error}</p>;
  if (!v) return <p className="rotulo">Cargando…</p>;
  return (
    <div className="space-y-4">
      <h2 className="text-xl font-medium">{v.nombre}</h2>
      <VistaPublica v={v} />
      <div className="flex gap-2">
        <Link href={`/rival/?tipo=publico&id=${v.perfil_id}`} className="objetivo-tactil inline-flex items-center rounded-md bg-primary px-4 text-sm font-medium text-primary-foreground">Usar como rival fantasma</Link>
        <Denunciar perfilId={v.perfil_id} />
      </div>
    </div>
  );
}

export default function Publico() {
  return <Marco titulo="Perfil público"><Suspense><Contenido /></Suspense></Marco>;
}
