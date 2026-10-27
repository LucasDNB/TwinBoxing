"use client";
import { Suspense } from "react";
import { Esqueleto } from "@/components/boxtwin/Estados";
import { FueraDeVersion } from "@/components/boxtwin/FueraDeVersion";
import { PantallaPeleadores } from "@/components/pantallas/peleadores/PantallaPeleadores";
import { useCuenta } from "@/lib/cuenta";
export default function Pagina() {
  // Versiones 26-10: en Individual la cuenta tiene un peleador, el propio (01 §2.2).
  const cuenta = useCuenta().data;
  if (cuenta?.funciones.peleadores === "propio") return <FueraDeVersion titulo="Peleadores" />;
  return (
    <Suspense fallback={<Esqueleto />}>
      <PantallaPeleadores />
    </Suspense>
  );
}
