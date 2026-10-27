"use client";

/*
 * Lo que se ve cuando la sesion no esta lista (04 §04): en cola, procesando con etapa y
 * porcentaje, espera de siembra con el boton a la identificacion, completando, o fallo con
 * el error y "Reintentar". El sondeo lo hace useSesion; aca solo se dibuja.
 *
 * En cola, con B27 (versiones 26-10, W35): cuantas sesiones van antes, de todas las cuentas,
 * y una espera aproximada con el factor de procesamiento medido. No dice de quien son.
 */
import Link from "next/link";
import { AnimatePresence, motion } from "motion/react";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { Progress } from "@/components/animate-ui/components/radix/progress";
import { LoaderCircle } from "@/components/animate-ui/icons/loader-circle";
import { Rotulo } from "@/components/boxtwin/Rotulo";
import { useSembrar } from "@/lib/api/consultas";
import type { Sesion } from "@/lib/api/esquemas";
import { porcentaje } from "@/lib/formato";
import { NOMBRE_ESTADO, NOMBRE_ETAPA } from "@/lib/textos";
import { esIndividual, useCuenta } from "@/lib/cuenta";

export function textoDeCola(c: NonNullable<Sesion["cola"]>): string {
  const antes = c.delante === 1 ? "Hay 1 sesión antes que la tuya." : `Hay ${c.delante} sesiones antes que la tuya.`;
  const [min, max] = c.espera_min;
  if (max <= 0) return antes;
  return `${antes} Unos ${min} a ${max} minutos para empezar.`;
}

export function EsperaSesion({ s }: { s: Sesion }) {
  const sembrar = useSembrar(s.id);
  const vos = esIndividual(useCuenta().data);
  const p = s.progreso;
  return (
    <AnimatePresence mode="wait">
      <motion.section
        key={s.estado}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2 }}
        aria-live="polite"
        className="rounded-md border border-borde bg-superficie p-5"
      >
        <Rotulo as="p">{NOMBRE_ESTADO[s.estado]}</Rotulo>
        {s.estado === "espera_siembra" ? (
          <>
            <p className="mt-2 text-lg">{vos ? "Falta un paso: marcate una vez." : "Falta un paso: marcá una vez a cada peleador."}</p>
            <Button asChild className="mt-4 min-h-11">
              <Link href={`/sesion/identificar/?id=${s.id}`}>{vos ? "Marcarme" : "Marcar peleadores"}</Link>
            </Button>
          </>
        ) : s.estado === "fallo" ? (
          <>
            <p className="mt-2 text-lg text-error">El análisis falló.</p>
            {s.error && <p className="mt-1 text-sm text-texto-2">{s.error}</p>}
            {s.siembra ? (
              <Button variant="outline" className="mt-4 min-h-11" disabled={sembrar.isPending}
                onClick={() => sembrar.mutate({ track_a: s.siembra!.track_a, track_b: s.siembra!.track_b })}>
                Reintentar
              </Button>
            ) : (
              <Button asChild variant="outline" className="mt-4 min-h-11"><Link href="/sesiones/nueva/">Subir de nuevo</Link></Button>
            )}
          </>
        ) : (
          <>
            <p className="mt-2 flex items-center gap-2 text-lg">
              <LoaderCircle animate loop size={20} aria-hidden="true" />
              {s.estado === "en_cola" && (s.cola && s.cola.delante > 0 ? textoDeCola(s.cola) : "Esperando su turno en la estación.")}
              {s.estado === "subida" && "Recibiendo el video."}
              {s.estado === "procesando" && "Buscando a las personas y su pose."}
              {s.estado === "completando" && "Siguiendo a cada peleador y contando golpes."}
            </p>
            {p && (
              <div className="mt-4">
                <p className="mb-1 text-sm text-texto-2">
                  {NOMBRE_ETAPA[p.etapa] ?? p.etapa}{p.fraccion != null ? ` · ${porcentaje(p.fraccion)}` : ""}
                </p>
                <Progress value={(p.fraccion ?? 0) * 100} aria-label={`Progreso de ${NOMBRE_ETAPA[p.etapa] ?? p.etapa}`} />
              </div>
            )}
            <p className="mt-4 text-sm text-texto-3">Podés cerrar la página: el análisis sigue y lo vas a ver en el panel.</p>
          </>
        )}
      </motion.section>
    </AnimatePresence>
  );
}
