"use client";
/*
 * Inicio. La misma ruta dibuja Mi progreso en Individual y el panel en las otras versiones
 * (versiones 26-10, 03 §I-01). La cuenta ya la espero el layout.
 */
import { PantallaPanel } from "@/components/pantallas/panel/PantallaPanel";
import { PantallaProgreso } from "@/components/pantallas/progreso/PantallaProgreso";
import { useCuenta } from "@/lib/cuenta";

export default function Pagina() {
  const cuenta = useCuenta().data;
  if (cuenta?.version === "individual") return <PantallaProgreso cuenta={cuenta} />;
  return <PantallaPanel />;
}
