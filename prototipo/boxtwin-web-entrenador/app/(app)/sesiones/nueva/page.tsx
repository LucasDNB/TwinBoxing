"use client";
import { Suspense } from "react";
import { Esqueleto } from "@/components/boxtwin/Estados";
import { PantallaNueva } from "@/components/pantallas/nueva/PantallaNueva";
export default function Pagina() {
  return (
    <Suspense fallback={<Esqueleto />}>
      <PantallaNueva />
    </Suspense>
  );
}
