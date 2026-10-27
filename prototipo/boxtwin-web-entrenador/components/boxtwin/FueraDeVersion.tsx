"use client";

/*
 * Una pagina que no es de la version, abierta por URL (versiones 26-10, 03 §1). No es "no
 * disponible": en esa version no existe. La API la rechaza igual (B22).
 */
import Link from "next/link";
import type { ReactNode } from "react";
import { Encabezado } from "@/components/boxtwin/Encabezado";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { NOMBRE_VERSION, useCuenta } from "@/lib/cuenta";
import { textos } from "@/lib/textos";

export function FueraDeVersion({ titulo, mensaje }: { titulo: string; mensaje?: ReactNode }) {
  const cuenta = useCuenta().data;
  return (
    <>
      <Encabezado titulo={titulo} />
      <div role="status" className="rounded-md border border-borde bg-superficie p-4">
        <p>{mensaje ?? textos.fueraDeVersion(cuenta ? NOMBRE_VERSION[cuenta.version] : "")}</p>
        <Button asChild variant="outline" className="mt-3 min-h-11">
          <Link href="/">Ir al inicio</Link>
        </Button>
      </div>
    </>
  );
}
