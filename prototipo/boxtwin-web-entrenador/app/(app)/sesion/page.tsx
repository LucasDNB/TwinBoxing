"use client";
import { Suspense } from "react";
import { Esqueleto } from "@/components/boxtwin/Estados";
import { PantallaSesion } from "@/components/pantallas/sesion/PantallaSesion";
export default function Pagina() {
  return (
    <Suspense fallback={<Esqueleto />}>
      <PantallaSesion />
    </Suspense>
  );
}
