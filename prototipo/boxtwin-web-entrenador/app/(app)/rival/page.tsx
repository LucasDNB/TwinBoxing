"use client";
import { Suspense } from "react";
import { Esqueleto } from "@/components/boxtwin/Estados";
import { PantallaRival } from "@/components/pantallas/rival/PantallaRival";
export default function Pagina() {
  return (
    <Suspense fallback={<Esqueleto />}>
      <PantallaRival />
    </Suspense>
  );
}
