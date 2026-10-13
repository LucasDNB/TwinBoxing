"use client";
import { Suspense } from "react";
import { Esqueleto } from "@/components/boxtwin/Estados";
import { PantallaSesiones } from "@/components/pantallas/sesiones/PantallaSesiones";
export default function Pagina() {
  return (
    <Suspense fallback={<Esqueleto />}>
      <PantallaSesiones />
    </Suspense>
  );
}
