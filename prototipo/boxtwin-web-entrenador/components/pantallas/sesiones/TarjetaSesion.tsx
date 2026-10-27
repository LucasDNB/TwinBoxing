"use client";

/*
 * Una sesion en una lista (04 §01, punto 4): fecha, A vs B con sus marcas, contexto,
 * duracion, insignia de estado y accion. En curso, barra con la etapa ("Procesando · pose ·
 * 64%"). Tocar lleva a donde corresponde por estado (04 §01, interaccion).
 */
import Link from "next/link";
import type { Sesion } from "@/lib/api/esquemas";
import { enCurso } from "@/lib/api/consultas";
import { Esquina } from "@/components/boxtwin/Esquina";
import { InsigniaEstado } from "@/components/boxtwin/Insignia";
import { Progress } from "@/components/animate-ui/components/radix/progress";
import { fechaCorta, mmss, porcentaje } from "@/lib/formato";
import { NOMBRE_CONTEXTO, NOMBRE_ESTADO, NOMBRE_ETAPA } from "@/lib/textos";
import { esIndividual, tiene, useCuenta } from "@/lib/cuenta";

export function destinoDeSesion(s: Pick<Sesion, "id" | "estado">): string {
  if (s.estado === "espera_siembra") return `/sesion/identificar/?id=${s.id}`;
  return `/sesion/?id=${s.id}`;
}

export function accionDeSesion(s: Pick<Sesion, "estado">): string {
  if (s.estado === "espera_siembra") return "Marcar peleadores";
  if (s.estado === "listo") return "Ver Fight-Card";
  if (s.estado === "fallo") return "Ver error";
  return "Ver estado";
}

export function textoDeProgreso(s: Sesion): string {
  const base = NOMBRE_ESTADO[s.estado] ?? s.estado;
  const p = s.progreso;
  if (!p) return base;
  const etapa = NOMBRE_ETAPA[p.etapa] ?? p.etapa;
  return p.fraccion != null ? `${base} · ${etapa} · ${porcentaje(p.fraccion)}` : `${base} · ${etapa}`;
}

function Quienes({ s }: { s: Sesion }) {
  // En Individual el compañero no tiene perfil y no se lo llama "B" (CA-I05.1).
  const individual = esIndividual(useCuenta().data);
  const a = s.boxeadores?.A?.nombre;
  const b = s.boxeadores?.B?.nombre ?? (individual ? "Compañero" : undefined);
  const deDos = s.contexto !== "bolsa" && s.contexto !== "sombra";
  return (
    <span className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
      <span className="inline-flex min-w-0 items-center gap-1.5"><Esquina lado="A" tamano={18} /><span className="truncate">{a ?? "A"}</span></span>
      {deDos && (
        <>
          <span className="text-tenue">vs</span>
          <span className="inline-flex min-w-0 items-center gap-1.5"><Esquina lado="B" tamano={18} /><span className="truncate">{b ?? "B"}</span></span>
        </>
      )}
    </span>
  );
}

/* B24: en Gimnasio cada sesion dice quien la subio (CA-G01.2). */
function SubidaPor({ s }: { s: Sesion }) {
  const compartido = tiene(useCuenta().data, "plantel_compartido");
  if (!compartido || !s.subida_por?.nombre) return null;
  return <p className="mt-0.5 text-xs text-texto-3">Subió {s.subida_por.nombre}</p>;
}

export function TarjetaSesion({ s }: { s: Sesion }) {
  const curso = enCurso(s.estado);
  return (
    <Link
      href={destinoDeSesion(s)}
      className="block rounded-md border border-borde bg-superficie p-4 transition-colors hover:border-borde-fuerte"
      aria-label={`${s.nombre}, ${NOMBRE_ESTADO[s.estado]}. ${accionDeSesion(s)}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="rotulo">
            {fechaCorta(s.creada)} · {NOMBRE_CONTEXTO[s.contexto ?? "sparring"]}
            {s.duracion_s ? ` · ${mmss(s.duracion_s)}` : ""}
          </p>
          <p className="mt-1 font-medium"><Quienes s={s} /></p>
          <p className="mt-0.5 truncate text-sm text-texto-3">{s.nombre}</p>
          <SubidaPor s={s} />
        </div>
        <InsigniaEstado estado={s.estado} />
      </div>
      {curso && (
        <div className="mt-3">
          <p className="mb-1 text-xs text-texto-2">{textoDeProgreso(s)}</p>
          <Progress value={(s.progreso?.fraccion ?? 0) * 100} aria-label={textoDeProgreso(s)} />
        </div>
      )}
      {s.estado === "fallo" && s.error && <p className="mt-2 text-sm text-error">{s.error}</p>}
      <p className="mt-3 text-sm text-texto-2 underline underline-offset-4">{accionDeSesion(s)}</p>
    </Link>
  );
}

/* La misma sesion como fila de tabla, para escritorio (04 §01). */
export function FilaSesion({ s }: { s: Sesion }) {
  return (
    <tr className="border-t border-borde-suave">
      <td className="dato py-3 pr-3 text-texto-2">{fechaCorta(s.creada)}</td>
      <td className="py-3 pr-3"><Quienes s={s} /><SubidaPor s={s} /></td>
      <td className="py-3 pr-3 text-texto-2">{NOMBRE_CONTEXTO[s.contexto ?? "sparring"]}</td>
      <td className="dato py-3 pr-3 text-texto-2">{s.duracion_s ? mmss(s.duracion_s) : "—"}</td>
      <td className="py-3 pr-3">
        <InsigniaEstado estado={s.estado} />
        {enCurso(s.estado) && <span className="mt-1 block text-xs text-texto-3">{textoDeProgreso(s)}</span>}
      </td>
      <td className="py-3 text-right">
        <Link href={destinoDeSesion(s)} className="inline-flex min-h-11 items-center text-sm underline underline-offset-4">
          {accionDeSesion(s)}
        </Link>
      </td>
    </tr>
  );
}
