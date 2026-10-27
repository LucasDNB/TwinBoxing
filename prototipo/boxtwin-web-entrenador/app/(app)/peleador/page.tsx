"use client";
import { Suspense } from "react";
import { Esqueleto } from "@/components/boxtwin/Estados";
import { PantallaPeleador } from "@/components/pantallas/peleador/PantallaPeleador";
export default function Pagina() {
  return (
    <Suspense fallback={<Esqueleto />}>
      <PantallaPeleador />
    </Suspense>
  );
}
