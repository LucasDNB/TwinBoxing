"use client";

/*
 * 01 · Panel del gimnasio (04 §01).
 *
 * Celular, de arriba hacia abajo: encabezado, pendientes, cuatro indicadores en 2x2,
 * sesiones recientes como tarjetas, mis peleadores. Escritorio: indicadores en una fila de
 * cuatro, y debajo sesiones como tabla (1,6 fr) y peleadores (1 fr). Sin el bloque del plan.
 *
 * Ningun numero sale de una constante (CA-01.2): todo viene de /panel, /jobs y /boxeadores.
 *
 * En Gimnasio (versiones 26-10, 03 §G-01, W34): el nombre del gimnasio, quien subio cada
 * pendiente con un filtro "Todas / Mías" (todas por defecto: el plantel es compartido, DV-3),
 * un quinto indicador con las horas del mes y, solo para el titular, "El equipo este mes".
 */
import Link from "next/link";
import { useState } from "react";
import { useQueries } from "@tanstack/react-query";
import { Encabezado } from "@/components/boxtwin/Encabezado";
import { ErrorConReintento, Esqueleto, Vacio } from "@/components/boxtwin/Estados";
import { Numero } from "@/components/boxtwin/Numero";
import { Rotulo } from "@/components/boxtwin/Rotulo";
import { Fade } from "@/components/animate-ui/primitives/effects/fade";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { FilaSesion, TarjetaSesion } from "@/components/pantallas/sesiones/TarjetaSesion";
import { pedirValidado } from "@/lib/api/cliente";
import { claves, useBoxeadores, usePanel, useProducto, useSesion, useSesiones, useSembrar } from "@/lib/api/consultas";
import { Perfil, type Boxeador, type Cuenta, type Panel, type Sesion } from "@/lib/api/esquemas";
import { visible } from "@/lib/banderas";
import { decimal, fechaCorta, fechaLarga, porcentajeAR } from "@/lib/formato";
import { CONDICIONES_FILMACION } from "@/lib/subida";
import { NOMBRE_CONTEXTO, textos } from "@/lib/textos";
import { ToggleGroup, ToggleGroupItem } from "@/components/animate-ui/components/radix/toggle-group";
import { Horas } from "@/components/boxtwin/Horas";
import { esGimnasio, useCuenta } from "@/lib/cuenta";

const MAX_PENDIENTES = 2;
const RECIENTES = 6;

export function PantallaPanel() {
  const panel = usePanel();
  const sesiones = useSesiones();
  const boxeadores = useBoxeadores();
  const producto = useProducto();
  const cuenta = useCuenta().data ?? null;

  const cargando = panel.isPending || sesiones.isPending;
  const error = panel.error ?? sesiones.error;

  return (
    <>
      <Encabezado titulo={esGimnasio(cuenta) ? cuenta!.nombre : "Panel"} rotulo={fechaLarga(new Date())} />
      {cargando ? (
        <Esqueleto filas={4} />
      ) : error ? (
        <ErrorConReintento error={error} alReintentar={() => { panel.refetch(); sesiones.refetch(); }} />
      ) : (
        <Contenido panel={panel.data!} sesiones={sesiones.data!} boxeadores={boxeadores.data ?? []} tendencia={visible("01.tendencia", producto.data)} cuenta={cuenta} />
      )}
    </>
  );
}

function Contenido({ panel, sesiones, boxeadores, tendencia, cuenta }: { panel: Panel; sesiones: Sesion[]; boxeadores: Boxeador[]; tendencia: boolean; cuenta: Cuenta | null }) {
  const gimnasio = esGimnasio(cuenta);
  const [solo, setSolo] = useState<"todas" | "mias">("todas");
  // Pendientes desde la lista de sesiones, que se sondea: asi un cambio de estado aparece
  // sin recargar (CA-01.1) aunque /panel no haya vuelto a pedirse.
  const pendientes = sesiones.filter((s) => s.estado === "espera_siembra"
    && (solo === "todas" || s.subida_por?.id === cuenta?.usuario.id));
  const fallidas = sesiones.filter((s) => s.estado === "fallo");
  const sinSesiones = sesiones.length === 0;

  return (
    <div className="flex flex-col gap-6">
      {gimnasio && sesiones.some((s) => s.estado === "espera_siembra") && (
        <ToggleGroup type="single" value={solo} onValueChange={(v) => v && setSolo(v as "todas" | "mias")} aria-label="Pendientes de quién" className="flex w-max gap-1">
          <ToggleGroupItem value="todas" className="min-h-11 border border-borde px-3">Todas</ToggleGroupItem>
          <ToggleGroupItem value="mias" className="min-h-11 border border-borde px-3">Mías</ToggleGroupItem>
        </ToggleGroup>
      )}
      {(pendientes.length > 0 || fallidas.length > 0) && (
        <section aria-label="Pendientes" className="flex flex-col gap-3">
          {pendientes.slice(0, MAX_PENDIENTES).map((s) => <AvisoPendiente key={s.id} s={s} gimnasio={gimnasio} />)}
          {pendientes.length > MAX_PENDIENTES && (
            <Link href="/sesiones/?estado=espera_siembra" className="min-h-11 text-sm text-estimado underline underline-offset-4">
              Ver las {pendientes.length} pendientes
            </Link>
          )}
          {fallidas.slice(0, MAX_PENDIENTES).map((s) => <AvisoFallo key={s.id} s={s} />)}
        </section>
      )}

      {sinSesiones ? (
        <Vacio
          titulo="Subí tu primer sparring"
          accion={<Button asChild className="min-h-11"><Link href="/sesiones/nueva/">Subir sesión</Link></Button>}
        >
          <ul className="mt-2 list-disc pl-5">
            {CONDICIONES_FILMACION.map((c) => <li key={c.id}>{c.texto}</li>)}
          </ul>
        </Vacio>
      ) : (
        <>
          <Indicadores panel={panel} />
          {gimnasio && (panel.horas ?? cuenta?.horas) && <Horas horas={(panel.horas ?? cuenta!.horas)!} />}
        </>
      )}
      {panel.por_entrenador && <EquipoDelMes filas={panel.por_entrenador} />}

      <div className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
        {!sinSesiones && (
          <section aria-labelledby="recientes">
            <div className="mb-3 flex items-end justify-between">
              <h2 id="recientes" className="font-display text-xl font-semibold lg:text-2xl">Sesiones recientes</h2>
              <Link href="/sesiones/" className="min-h-11 content-center text-sm text-texto-2 underline underline-offset-4">Ver todas</Link>
            </div>
            <Fade className="flex flex-col gap-3 lg:hidden">
              {sesiones.slice(0, RECIENTES).map((s) => <TarjetaSesion key={s.id} s={s} />)}
            </Fade>
            <div className="hidden overflow-x-auto rounded-md border border-borde bg-superficie px-4 lg:block">
              <table className="w-full text-sm">
                <caption className="sr-only">Sesiones recientes</caption>
                <thead>
                  <tr className="text-left">
                    {["Fecha", "Peleadores", "Contexto", "Duración", "Estado", ""].map((h) => (
                      <th key={h} scope="col" className="rotulo py-3 pr-3 font-normal">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>{sesiones.slice(0, RECIENTES).map((s) => <FilaSesion key={s.id} s={s} />)}</tbody>
              </table>
            </div>
          </section>
        )}

        <section aria-labelledby="mis-peleadores">
          <div className="mb-3 flex items-end justify-between">
            <h2 id="mis-peleadores" className="font-display text-xl font-semibold lg:text-2xl">{gimnasio ? "Plantel" : "Mis peleadores"}</h2>
            <Link href="/peleadores/" className="min-h-11 content-center text-sm text-texto-2 underline underline-offset-4">Ver todos</Link>
          </div>
          {boxeadores.length === 0 ? (
            <Vacio titulo="Cargá a tus peleadores para armar sus perfiles" accion={<Button asChild variant="outline" className="min-h-11"><Link href="/peleadores/">Agregar</Link></Button>} />
          ) : (
            <MisPeleadores boxeadores={boxeadores} tendencia={tendencia} />
          )}
        </section>
      </div>
    </div>
  );
}

function AvisoPendiente({ s, gimnasio = false }: { s: Sesion; gimnasio?: boolean }) {
  return (
    <Link
      href={`/sesion/identificar/?id=${s.id}`}
      className="block rounded-md border border-estimado-borde bg-estimado-fondo p-4"
    >
      <Rotulo className="text-estimado">falta un paso</Rotulo>
      <p className="mt-1 text-texto">
        Marcá una vez a cada peleador {s.contexto === "combate" ? "del combate" : "del sparring"}
        {s.boxeadores?.A || s.boxeadores?.B ? ` de ${[s.boxeadores?.A?.nombre, s.boxeadores?.B?.nombre].filter(Boolean).join(" vs ")}` : ` "${s.nombre}"`}.
      </p>
      {gimnasio && s.subida_por?.nombre && <p className="mt-1 text-xs text-texto-2">Subió {s.subida_por.nombre}</p>}
      <span className="mt-2 inline-block text-sm underline underline-offset-4">Marcar peleadores</span>
    </Link>
  );
}

/* B24: solo el titular de un Gimnasio; la API se lo manda nulo a los demas (CA-G01.1). */
function EquipoDelMes({ filas }: { filas: NonNullable<Panel["por_entrenador"]> }) {
  return (
    <section aria-labelledby="equipo-mes" className="rounded-md border border-borde bg-superficie p-4">
      <h2 id="equipo-mes" className="font-display text-xl font-semibold lg:text-2xl">El equipo este mes</h2>
      <div className="mt-3 overflow-x-auto" tabIndex={0} role="region" aria-label="Actividad por entrenador">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left">
              {["Entrenador", "Sesiones", "Horas", "Pendientes"].map((h) => (
                <th key={h} scope="col" className="rotulo py-2 pr-3 font-normal">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filas.map((f) => (
              <tr key={f.usuario_id} className="border-t border-borde-suave">
                <td className="py-2 pr-3">{f.nombre ?? "—"}</td>
                <td className="dato py-2 pr-3">{f.sesiones_mes}</td>
                <td className="dato py-2 pr-3">{decimal(f.horas_mes, 1)}</td>
                <td className="dato py-2 pr-3">{f.pendientes}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/*
 * Reintentar: si la sesion ya tenia la siembra, se vuelve a mandar la misma (la API acepta
 * sembrar una sesion en fallo y reencola la segunda etapa). Si fallo antes, en la primera
 * etapa, no hay nada que reencolar sin el video: se ofrece subirlo de nuevo.
 */
export function AvisoFallo({ s }: { s: Sesion }) {
  const detalle = useSesion(s.id);
  const sembrar = useSembrar(s.id);
  const siembra = detalle.data?.siembra ?? s.siembra;
  return (
    <div role="alert" className="rounded-md border border-error-borde bg-error-fondo p-4">
      <Rotulo className="text-error">falló</Rotulo>
      <p className="mt-1">{s.nombre}</p>
      {s.error && <p className="mt-1 text-sm text-texto-2">{s.error}</p>}
      <div className="mt-3">
        {siembra ? (
          <Button variant="outline" className="min-h-11" disabled={sembrar.isPending}
            onClick={() => sembrar.mutate({ track_a: siembra.track_a, track_b: siembra.track_b })}>
            Reintentar
          </Button>
        ) : (
          <Button asChild variant="outline" className="min-h-11"><Link href="/sesiones/nueva/">Subir de nuevo</Link></Button>
        )}
      </div>
    </div>
  );
}

function Indicador({ titulo, valor, children }: { titulo: string; valor: number; children?: React.ReactNode }) {
  return (
    <div className="rounded-md border border-borde bg-superficie p-4">
      <Rotulo as="p">{titulo}</Rotulo>
      <Numero valor={valor} className="mt-1 block font-display text-[34px] font-semibold leading-none lg:text-[42px]" />
      {children && <div className="mt-2 text-xs text-texto-3">{children}</div>}
    </div>
  );
}

function Indicadores({ panel }: { panel: Panel }) {
  const m = panel.sesiones_mes;
  const porContexto = Object.entries(panel.rounds_mes).filter(([k, v]) => k !== "total" && v > 0);
  return (
    <section aria-label="Indicadores del mes" className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <Indicador titulo="Peleadores activos" valor={panel.peleadores_activos}>
        {panel.peleadores_preliminares > 0 && `${panel.peleadores_preliminares} con perfil preliminar`}
      </Indicador>
      <Indicador titulo="Sesiones del mes" valor={m.total}>
        {[m.listo && `${m.listo} ${m.listo === 1 ? "lista" : "listas"}`, m.en_curso && `${m.en_curso} en curso`, m.espera_siembra && `${m.espera_siembra} ${m.espera_siembra === 1 ? "espera" : "esperan"}`, m.fallo && `${m.fallo} ${m.fallo === 1 ? "falló" : "fallaron"}`].filter(Boolean).join(" · ")}
      </Indicador>
      <Indicador titulo="Rounds analizados" valor={panel.rounds_mes.total}>
        {porContexto.map(([k, v]) => `${v} ${NOMBRE_CONTEXTO[k]?.toLowerCase() ?? k}`).join(" · ")}
      </Indicador>
      <Indicador titulo="Golpes detectados" valor={panel.golpes_detectados_mes}>
        {panel.recall_detector != null ? textos.margenPanel(panel.recall_detector) : "Sin sesiones listas este mes."}
      </Indicador>
    </section>
  );
}

/*
 * Tendencia: condicionada a C2 (CA-01.3). Apagada, muestra la ultima sesion y la cantidad.
 * Encendida, sale de caida_ultimo_round del perfil (B7); si el perfil no la trae, vuelve a la
 * ultima sesion.
 */
function MisPeleadores({ boxeadores, tendencia }: { boxeadores: Boxeador[]; tendencia: boolean }) {
  const mostrados = boxeadores.slice(0, 6);
  const perfiles = useQueries({
    queries: mostrados.map((b) => ({
      queryKey: claves.perfil(b.id, "todas"),
      queryFn: () => pedirValidado(Perfil, `/boxeadores/${b.id}/perfil?contexto=todas`),
      enabled: tendencia,
      staleTime: 30_000,
    })),
  });
  return (
    <ul className="flex flex-col divide-y divide-borde-suave rounded-md border border-borde bg-superficie">
      {mostrados.map((b, i) => {
        const caidas = (perfiles[i]?.data?.por_contexto ?? []).map((c) => c.caida_ultimo_round).filter((x): x is number => x != null);
        const caida = tendencia && caidas.length ? caidas.reduce((a, c) => a + c, 0) / caidas.length : null;
        return (
          <li key={b.id}>
            <Link href={`/peleador/?id=${b.id}`} className="flex min-h-14 items-center justify-between gap-3 px-4 py-3">
              <span className="min-w-0">
                <span className="block truncate font-medium">{b.nombre}</span>
                <span className="block text-xs text-texto-3">
                  {[b.categoria, b.guardia, b.sesiones != null ? `${b.sesiones} ${b.sesiones === 1 ? "sesión" : "sesiones"}` : null].filter(Boolean).join(" · ")}
                </span>
              </span>
              <span className="shrink-0 text-right text-xs text-texto-2">
                {caida != null
                  ? `el volumen ${caida < 0 ? "cae" : "sube"} ${porcentajeAR(Math.abs(caida))} en el último round`
                  : b.ultima_sesion
                    ? `última sesión ${fechaCorta(b.ultima_sesion)}`
                    : "sin sesiones"}
              </span>
            </Link>
          </li>
        );
      })}
    </ul>
  );
}

