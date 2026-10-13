"use client";
import { Suspense } from "react";
import { Esqueleto } from "@/components/boxtwin/Estados";
import { PantallaPeleadores } from "@/components/pantallas/peleadores/PantallaPeleadores";
export default function Pagina() {
  return (
    <Suspense fallback={<Esqueleto />}>
      <PantallaPeleadores />
    </Suspense>
  );
}
