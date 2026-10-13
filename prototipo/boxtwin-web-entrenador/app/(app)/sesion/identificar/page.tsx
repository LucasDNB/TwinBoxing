"use client";
import { Suspense } from "react";
import { Esqueleto } from "@/components/boxtwin/Estados";
import { PantallaIdentificar } from "@/components/pantallas/identificar/PantallaIdentificar";
export default function Pagina() {
  return (
    <Suspense fallback={<Esqueleto />}>
      <PantallaIdentificar />
    </Suspense>
  );
}
