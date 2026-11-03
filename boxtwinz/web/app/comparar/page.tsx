"use client";
/* Dos sesiones de la misma modalidad lado a lado, con las curvas superpuestas (spec 12). */
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { Curva } from "@/components/boxtwinz/Curva";
import { Marco } from "@/components/boxtwinz/Marco";
import { Metricas } from "@/components/boxtwinz/Metricas";
import { Rotulo } from "@/components/boxtwinz/Rotulo";
import { api, ErrorApi, type BloqueMetricas, type SesionDetalle } from "@/lib/api";
import { MODALIDAD } from "@/lib/textos";

type Lado = { sesion: SesionDetalle; metricas: { usuario?: BloqueMetricas } };
type Comp = { a: Lado; b: Lado; rotulo_rounds: string };
const fecha = (s: SesionDetalle) => new Date(s.creado).toLocaleDateString("es-AR");

function Contenido() {
  const q = useSearchParams();
  const [c, setC] = useState<Comp | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api<Comp>(`/comparar?a=${q.get("a")}&b=${q.get("b")}`).then(setC).catch((e) => setError(e instanceof ErrorApi ? e.message : "Error"));
  }, [q]);
  if (error) return <p role="alert" className="text-error">{error}</p>;
  if (!c) return <p className="rotulo">Cargando…</p>;
  const na = `A · ${fecha(c.a.sesion)}`, nb = `B · ${fecha(c.b.sesion)}`;
  return (
    <div className="space-y-4">
      <p className="text-texto-2">{MODALIDAD[c.a.sesion.modalidad]}</p>
      {c.a.sesion.rotulo && <Rotulo>{c.a.sesion.rotulo}</Rotulo>}
      <Metricas caption="Lado a lado" columnas={[{ nombre: na, m: c.a.metricas.usuario?.total }, { nombre: nb, m: c.b.metricas.usuario?.total }]} />
      <Curva series={[
        { nombre: na, valores: c.a.metricas.usuario?.total?.curva ?? [], color: "var(--color-texto)" },
        { nombre: nb, valores: c.b.metricas.usuario?.total?.curva ?? [], color: "var(--color-estimado-borde)" },
      ]} />
      <p className="text-xs text-tenue"><Rotulo>{c.rotulo_rounds}</Rotulo></p>
    </div>
  );
}

export default function Comparar() {
  return <Marco titulo="Comparar"><Suspense><Contenido /></Suspense></Marco>;
}
