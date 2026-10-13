"use client";

/*
 * "Perfiles de la sesion" (04 §04, encabezado): lleva a los perfiles de A y B. Si un lado no
 * tiene peleador asignado se puede asignar aca: A y B se asignan por posicion en pantalla y
 * no significan nada entre videos, asi que sin esto la sesion no suma a ningun perfil.
 * Asignar no toca la Fight-Card.
 */
import { useState } from "react";
import Link from "next/link";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { Esquina } from "@/components/boxtwin/Esquina";
import { SelectNativo } from "@/components/boxtwin/Controles";
import { mensajeDeError } from "@/components/boxtwin/Estados";
import { asignarBoxeadores, claves, useBoxeadores } from "@/lib/api/consultas";
import type { Sesion } from "@/lib/api/esquemas";
import { lados as ladosDe, type FightCard, type Lado } from "@/lib/fightcard";

export function VistaPerfiles({ s, fc }: { s: Sesion; fc: FightCard }) {
  const qc = useQueryClient();
  const boxeadores = useBoxeadores();
  const lados = ladosDe(fc);
  const actual: Record<Lado, string> = {
    A: fc.sesion?.boxeadores?.A?.id ?? s.boxeadores?.A?.id ?? "",
    B: fc.sesion?.boxeadores?.B?.id ?? s.boxeadores?.B?.id ?? "",
  };
  const [elegidos, setElegidos] = useState(actual);
  const [guardando, setGuardando] = useState(false);
  const cambio = elegidos.A !== actual.A || elegidos.B !== actual.B;
  const mismo = !!elegidos.A && elegidos.A === elegidos.B;

  async function guardar() {
    setGuardando(true);
    try {
      await asignarBoxeadores(s.id, elegidos.A || null, lados.includes("B") ? elegidos.B || null : null);
      await Promise.all([
        qc.invalidateQueries({ queryKey: claves.fightcard(s.id) }),
        qc.invalidateQueries({ queryKey: claves.sesion(s.id) }),
        qc.invalidateQueries({ queryKey: claves.sesiones }),
        qc.invalidateQueries({ queryKey: claves.boxeadores }),
      ]);
      toast.success("Peleadores asignados. La sesión ya suma a sus perfiles.");
    } catch (e) {
      toast.error(mensajeDeError(e));
    } finally {
      setGuardando(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className={`grid gap-3 ${lados.length > 1 ? "md:grid-cols-2" : ""}`}>
        {lados.map((l) => {
          const asignado = (boxeadores.data ?? []).find((b) => b.id === actual[l]);
          return (
            <section key={l} aria-label={`Peleador ${l}`} className="rounded-md border border-borde bg-superficie p-4">
              <p className="flex items-center gap-2 font-display text-lg font-semibold"><Esquina lado={l} /> {l === "A" ? "Rincón rojo" : "Rincón azul"}</p>
              {asignado ? (
                <Link href={`/peleador/?id=${asignado.id}`} className="mt-2 inline-flex min-h-11 items-center text-[15px] underline underline-offset-4">
                  Ver el perfil de {asignado.nombre}
                </Link>
              ) : (
                <p className="mt-2 text-sm text-texto-3">Sin peleador asignado: esta sesión no suma a ningún perfil.</p>
              )}
              <label htmlFor={`asignar-${l}`} className="mt-3 block text-sm text-texto-2">Asignar</label>
              <SelectNativo id={`asignar-${l}`} className="mt-1" value={elegidos[l]} onChange={(e) => setElegidos((x) => ({ ...x, [l]: e.target.value }))}>
                <option value="">Sin perfil</option>
                {(boxeadores.data ?? []).map((b) => <option key={b.id} value={b.id}>{b.nombre}</option>)}
              </SelectNativo>
            </section>
          );
        })}
      </div>
      {mismo && <p role="alert" className="text-sm text-error">El mismo peleador no puede estar en los dos rincones.</p>}
      <div className="flex flex-wrap gap-3">
        <Button className="min-h-11" disabled={!cambio || mismo || guardando} onClick={guardar}>Guardar asignación</Button>
        <Button asChild variant="outline" className="min-h-11"><Link href="/peleadores/">Agregar peleador</Link></Button>
      </div>
    </div>
  );
}
