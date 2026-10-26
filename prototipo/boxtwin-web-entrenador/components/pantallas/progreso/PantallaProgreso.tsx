"use client";

/*
 * I-01 · Mi progreso, el inicio de la version Individual (versiones 26-10, 03 §3, W29).
 *
 * Celular, de arriba hacia abajo: encabezado, pendientes, tres tarjetas (ultima sesion, como
 * venis, tus sesiones) y las horas del mes. Escritorio: las tres tarjetas en una fila, horas
 * arriba a la derecha.
 *
 * Ningun numero sale de una constante (CA-I01.1): golpes y golpes por minuto salen de la
 * evolucion del perfil propio, el margen del detector de /panel y las horas de /cuenta. La
 * tendencia nunca pone en la misma linea sesiones de contextos distintos (CA-I01.2): cada
 * contexto mide algo distinto.
 */
import Link from "next/link";
import { Encabezado } from "@/components/boxtwin/Encabezado";
import { ErrorConReintento, Esqueleto, Vacio } from "@/components/boxtwin/Estados";
import { Horas } from "@/components/boxtwin/Horas";
import { Rotulo } from "@/components/boxtwin/Rotulo";
import { LineaEvolucion } from "@/components/boxtwin/graficos/LineaEvolucion";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { AvisoFallo } from "@/components/pantallas/panel/PantallaPanel";
import { TarjetaSesion } from "@/components/pantallas/sesiones/TarjetaSesion";
import { usePanel, usePerfil, useSesiones } from "@/lib/api/consultas";
import type { Cuenta, Perfil, Sesion } from "@/lib/api/esquemas";
import { decimal, entero, fechaCorta, fechaLarga, mmss } from "@/lib/formato";
import { NOMBRE_CONTEXTO, textos } from "@/lib/textos";

const DIAS_TENDENCIA = 30;
const ULTIMAS = 5;
const ORDEN_CONTEXTOS = ["bolsa", "sombra", "sparring", "combate"];
const DE_UNA_PERSONA = ["bolsa", "sombra"];

/* Una checklist para una persona sola frente a la camara (03 §I-01). */
export const CONDICIONES_UNA_PERSONA = [
  "Cámara fija, apoyada, con tu cuerpo entero en cuadro.",
  "Los guantes a la vista.",
  "Luz de frente, no detrás tuyo.",
  "En sombra, el espejo fuera de cuadro si se puede.",
];

export function PantallaProgreso({ cuenta }: { cuenta: Cuenta }) {
  const propio = cuenta.boxeador_propio?.id ?? null;
  const sesiones = useSesiones();
  const panel = usePanel();
  const perfil = usePerfil(propio);

  const cargando = sesiones.isPending || panel.isPending || (!!propio && perfil.isPending);
  const error = sesiones.error ?? panel.error ?? (propio ? perfil.error : null);

  return (
    <>
      <Encabezado titulo="Mi progreso" rotulo={fechaLarga(new Date())} />
      {cargando ? (
        <Esqueleto filas={3} />
      ) : error ? (
        <ErrorConReintento error={error} alReintentar={() => { sesiones.refetch(); panel.refetch(); perfil.refetch(); }} />
      ) : (
        <Contenido
          cuenta={cuenta}
          sesiones={sesiones.data!}
          perfil={perfil.data ?? null}
          recall={panel.data?.recall_detector ?? null}
        />
      )}
    </>
  );
}

function Contenido({ cuenta, sesiones, perfil, recall }: {
  cuenta: Cuenta; sesiones: Sesion[]; perfil: Perfil | null; recall: number | null;
}) {
  const pendientes = sesiones.filter((s) => s.estado === "espera_siembra");
  const fallidas = sesiones.filter((s) => s.estado === "fallo");
  const horas = cuenta.horas ? <Horas horas={cuenta.horas} /> : null;

  return (
    <div className="flex flex-col gap-6">
      {(pendientes.length > 0 || fallidas.length > 0) && (
        <section aria-label="Pendientes" className="flex flex-col gap-3">
          {pendientes.map((s) => <Pendiente key={s.id} s={s} />)}
          {fallidas.map((s) => <AvisoFallo key={s.id} s={s} />)}
        </section>
      )}

      {sesiones.length === 0 ? (
        <>
          <Vacio
            titulo="Subí tu primera sesión de bolsa"
            accion={<Button asChild className="min-h-11"><Link href="/sesiones/nueva/">Subir sesión</Link></Button>}
          >
            <ul className="mt-2 list-disc pl-5">
              {CONDICIONES_UNA_PERSONA.map((c) => <li key={c}>{c}</li>)}
            </ul>
          </Vacio>
          {horas}
        </>
      ) : (
        <div className="grid gap-4 lg:grid-cols-3">
          <div className="lg:order-last lg:col-span-3 lg:flex lg:justify-end">
            <div className="lg:w-1/3">{horas}</div>
          </div>
          <UltimaSesion sesiones={sesiones} perfil={perfil} recall={recall} />
          <ComoVenis sesiones={sesiones} perfil={perfil} />
          <TusSesiones sesiones={sesiones} />
        </div>
      )}
    </div>
  );
}

function Pendiente({ s }: { s: Sesion }) {
  const ctx = (NOMBRE_CONTEXTO[s.contexto ?? "sparring"] ?? "").toLowerCase();
  return (
    <Link href={`/sesion/identificar/?id=${s.id}`} className="block rounded-md border border-estimado-borde bg-estimado-fondo p-4">
      <Rotulo className="text-estimado">falta un paso</Rotulo>
      <p className="mt-1 text-texto">
        Falta un paso: marcate en la sesión de {ctx} del {fechaCorta(s.creada)}.
      </p>
      <span className="mt-2 inline-block text-sm underline underline-offset-4">Marcarme</span>
    </Link>
  );
}

function Tarjeta({ titulo, children }: { titulo: string; children: React.ReactNode }) {
  return (
    <section aria-label={titulo} className="flex flex-col rounded-md border border-borde bg-superficie p-4">
      <h2 className="font-display text-xl font-semibold">{titulo}</h2>
      <div className="mt-3 flex-1">{children}</div>
    </section>
  );
}

function UltimaSesion({ sesiones, perfil, recall }: { sesiones: Sesion[]; perfil: Perfil | null; recall: number | null }) {
  const lista = sesiones.find((s) => s.estado === "listo");
  if (!lista) {
    return <Tarjeta titulo="Última sesión"><p className="text-sm text-texto-3">Todavía no hay una sesión lista.</p></Tarjeta>;
  }
  const ctx = lista.contexto ?? "sparring";
  const ev = perfil?.evolucion.find((e) => e.sesion_id === lista.id) ?? null;
  return (
    <Tarjeta titulo="Última sesión">
      <Link href={`/sesion/?id=${lista.id}`} className="block">
        <p className="rotulo">{NOMBRE_CONTEXTO[ctx]} · {fechaCorta(lista.creada)}{lista.duracion_s != null ? ` · ${mmss(lista.duracion_s)}` : ""}</p>
        {ev ? (
          <dl className="mt-2 grid grid-cols-2 gap-2">
            <div>
              <dt className="text-xs text-texto-3">Golpes detectados</dt>
              <dd className="font-display text-[34px] font-semibold leading-none">{entero(ev.golpes)}</dd>
            </div>
            <div>
              <dt className="text-xs text-texto-3">Golpes por minuto</dt>
              <dd className="font-display text-[34px] font-semibold leading-none">{decimal(ev.golpes_por_minuto, 1)}</dd>
            </div>
          </dl>
        ) : (
          <p className="mt-2 text-sm text-texto-3">Abrí la sesión para ver los golpes.</p>
        )}
      </Link>
      <p className="mt-3 text-xs text-texto-3">
        {DE_UNA_PERSONA.includes(ctx)
          ? textos.margenOtroContexto("sparring", ctx)
          : recall != null ? textos.margenPanel(recall) : null}
      </p>
    </Tarjeta>
  );
}

/* Exportada para la prueba de CA-I01.2: una linea por contexto, nunca mezclados. */
export function lineasPorContexto(perfil: Perfil | null, sesiones: Sesion[], ahora = new Date()) {
  const contexto = new Map(sesiones.map((s) => [s.id, s.contexto ?? "sparring"]));
  const desde = ahora.getTime() - DIAS_TENDENCIA * 24 * 3600 * 1000;
  const lineas: Record<string, Perfil["evolucion"]> = {};
  for (const e of perfil?.evolucion ?? []) {
    if (new Date(e.fecha).getTime() < desde) continue;
    const c = contexto.get(e.sesion_id);
    if (!c) continue;
    (lineas[c] ??= []).push(e);
  }
  return ORDEN_CONTEXTOS.filter((c) => lineas[c]).map((c) => ({ contexto: c, puntos: lineas[c] }));
}

function ComoVenis({ sesiones, perfil }: { sesiones: Sesion[]; perfil: Perfil | null }) {
  const lineas = lineasPorContexto(perfil, sesiones);
  return (
    <Tarjeta titulo="Cómo venís">
      <p className="text-xs text-texto-3">Golpes por minuto de cada sesión en los últimos {DIAS_TENDENCIA} días.</p>
      {lineas.length === 0 && <p className="mt-2 text-sm text-texto-3">Todavía no hay sesiones listas este mes.</p>}
      <div className="mt-2 flex flex-col gap-4">
        {lineas.map(({ contexto, puntos }) => (
          <div key={contexto} data-testid={`linea-${contexto}`}>
            <p className="rotulo">{NOMBRE_CONTEXTO[contexto]}</p>
            {puntos.length < 2 ? (
              <p className="mt-1 text-sm text-texto-2">{textos.dosSesiones(NOMBRE_CONTEXTO[contexto].toLowerCase())}</p>
            ) : (
              <LineaEvolucion evolucion={puntos} />
            )}
          </div>
        ))}
      </div>
    </Tarjeta>
  );
}

function TusSesiones({ sesiones }: { sesiones: Sesion[] }) {
  return (
    <Tarjeta titulo="Tus sesiones">
      <div className="flex flex-col gap-3">
        {sesiones.slice(0, ULTIMAS).map((s) => <TarjetaSesion key={s.id} s={s} />)}
      </div>
      <Link href="/sesiones/" className="mt-3 inline-block min-h-11 content-center text-sm text-texto-2 underline underline-offset-4">Ver todas</Link>
    </Tarjeta>
  );
}
