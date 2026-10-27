"use client";
import { Suspense } from "react";
import { Esqueleto } from "@/components/boxtwin/Estados";
import { FueraDeVersion } from "@/components/boxtwin/FueraDeVersion";
import { PantallaRival } from "@/components/pantallas/rival/PantallaRival";
import { tiene, useCuenta } from "@/lib/cuenta";
export default function Pagina() {
  // Versiones 26-10: en Individual el Rival Fantasma no existe (CA-I06.1).
  const cuenta = useCuenta().data;
  if (cuenta && !tiene(cuenta, "rival")) return <FueraDeVersion titulo="Rival Fantasma" />;
  return (
    <Suspense fallback={<Esqueleto />}>
      <PantallaRival />
    </Suspense>
  );
}
