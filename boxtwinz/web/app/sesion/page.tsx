"use client";
/*
 * Resultado de una sesion (spec 12). Mientras procesa, muestra el estado y se refresca.
 * Lista: video con la linea de tiempo de golpes por mano, metricas, curva, exposicion,
 * altura estimada, consejos, valoracion obligatoria la primera vez y correccion golpe a golpe.
 */
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { Consejos } from "@/components/boxtwinz/Consejos";
import { Curva } from "@/components/boxtwinz/Curva";
import { PublicidadDelPlan } from "@/components/boxtwinz/EspacioPublicitario";
import { Marco } from "@/components/boxtwinz/Marco";
import { Metricas } from "@/components/boxtwinz/Metricas";
import { Rotulo } from "@/components/boxtwinz/Rotulo";
import { Button } from "@/components/ui/button";
import { api, ErrorApi, type GolpeR, type Resultado, type SesionDetalle } from "@/lib/api";
import { num, pct, tiempo } from "@/lib/formato";
import { MODALIDAD, TEXTOS } from "@/lib/textos";
import { cn } from "@/lib/utils";

const ESTADO: Record<string, string> = {
  subida: "En cola para procesar", procesando: "Buscando a las personas en el video",
  completando: "Contando los golpes", esperando_siembra: "Falta marcar quién es quién",
};
const COLOR_MANO = { izquierda: "var(--color-a-trazo)", derecha: "var(--color-b-trazo)" } as const;

function Valoracion({ id, alValorar }: { id: string; alValorar: () => void }) {
  const [estrellas, setEstrellas] = useState(0);
  const [comentario, setComentario] = useState("");
  const [enviando, setEnviando] = useState(false);
  async function enviar() {
    setEnviando(true);
    try {
      await api(`/sesiones/${id}/valoracion`, { method: "POST", json: { estrellas, comentario: comentario || null } });
      alValorar();
    } finally {
      setEnviando(false);
    }
  }
  return (
    <div className="space-y-3">
      <fieldset>
        <legend className="mb-2 text-sm">¿Qué tan bien contó tus golpes?</legend>
        <div className="flex gap-1">
          {[1, 2, 3, 4, 5].map((n) => (
            <button key={n} type="button" onClick={() => setEstrellas(n)} aria-pressed={estrellas >= n}
              aria-label={`${n} estrella${n > 1 ? "s" : ""}`}
              className={cn("objetivo-tactil text-3xl", estrellas >= n ? "text-estimado" : "text-borde-fuerte")}>★</button>
          ))}
        </div>
      </fieldset>
      <label className="block text-sm">
        Comentario (opcional)
        <textarea value={comentario} onChange={(e) => setComentario(e.target.value)} maxLength={2000} rows={2}
          className="mt-1 w-full rounded-md border border-input bg-superficie-2 p-2" />
      </label>
      <Button className="objetivo-tactil w-full" disabled={!estrellas || enviando} onClick={enviar}>Enviar valoración</Button>
    </div>
  );
}

function LineaDeTiempo({ golpes, duracion, actual, elegido, alElegir }: {
  golpes: GolpeR[]; duracion: number; actual: number; elegido: string | null; alElegir: (g: GolpeR) => void;
}) {
  const filas: ("usuario" | "rival")[] = golpes.some((g) => g.quien === "rival") ? ["usuario", "rival"] : ["usuario"];
  return (
    <div className="space-y-1" aria-label="Línea de tiempo de golpes detectados">
      {filas.map((q) => (
        <div key={q} className="flex items-center gap-2">
          <span className="w-12 text-xs text-tenue">{q === "usuario" ? "Vos" : "Rival"}</span>
          <div className="relative h-8 flex-1 rounded bg-superficie-2">
            {golpes.filter((g) => g.quien === q).map((g) => (
              <button key={g.id} type="button" onClick={() => alElegir(g)}
                aria-label={`Golpe de ${g.mano} en ${tiempo(g.t_inicio)}`}
                className={cn("absolute top-1 bottom-1 w-1.5 -translate-x-1/2 rounded-sm", elegido === g.id && "ring-2 ring-texto")}
                style={{ left: `${(g.t_inicio / duracion) * 100}%`, background: COLOR_MANO[g.mano] }} />
            ))}
            <span aria-hidden className="absolute top-0 bottom-0 w-px bg-texto" style={{ left: `${(actual / duracion) * 100}%` }} />
          </div>
        </div>
      ))}
      <p className="flex gap-4 text-xs text-texto-3">
        <span><span aria-hidden className="mr-1 inline-block size-2 rounded-sm" style={{ background: COLOR_MANO.izquierda }} />Izquierda</span>
        <span><span aria-hidden className="mr-1 inline-block size-2 rounded-sm" style={{ background: COLOR_MANO.derecha }} />Derecha</span>
      </p>
    </div>
  );
}

function Lista({ r, recargar }: { r: Resultado; recargar: () => void }) {
  const router = useRouter();
  const video = useRef<HTMLVideoElement>(null);
  const dialogo = useRef<HTMLDialogElement>(null);
  const [actual, setActual] = useState(0);
  const [elegido, setElegido] = useState<GolpeR | null>(null);
  const [aviso, setAviso] = useState<string | null>(null);
  const [destino, setDestino] = useState("/inicio/");
  const s = r.sesion;
  const dur = s.duracion_s ?? 1;
  const u = r.metricas.usuario?.total;
  const rv = r.metricas.rival?.total;
  const pendiente = r.valoracion_obligatoria;

  const interceptar = useCallback((href: string) => {
    if (!pendiente) return false;
    setDestino(href);
    dialogo.current?.showModal();
    return true;
  }, [pendiente]);

  async function corregir(tipo: "no_fue_golpe" | "otra_mano" | "falto_golpe") {
    const cuerpo = tipo === "falto_golpe" ? { tipo, t: video.current?.currentTime ?? actual } : { tipo, golpe_id: elegido?.id };
    try {
      await api(`/sesiones/${s.id}/correcciones`, { method: "POST", json: cuerpo });
      setAviso("Gracias: la corrección queda para mejorar el modelo. Los números de esta sesión no cambian.");
    } catch (e) {
      setAviso(e instanceof ErrorApi ? e.message : "No se pudo guardar la corrección");
    }
  }

  return (
    <Marco titulo={`${MODALIDAD[s.modalidad]} · resultado`} interceptar={interceptar}>
      <div className="space-y-5">
        <div className="flex flex-wrap gap-2">
          {r.rotulos.evaluacion && <Rotulo>{r.rotulos.evaluacion}</Rotulo>}
          <span className="text-xs text-tenue">Se cuentan {TEXTOS.golpes}; el detector encuentra menos que los reales.</span>
        </div>

        <video ref={video} src={`/api/sesiones/${s.id}/video`} controls playsInline preload="metadata"
          onTimeUpdate={(e) => setActual(e.currentTarget.currentTime)} className="w-full rounded-lg bg-black" />
        <LineaDeTiempo golpes={r.golpes} duracion={dur} actual={actual} elegido={elegido?.id ?? null}
          alElegir={(g) => { setElegido(g); if (video.current) video.current.currentTime = Math.max(0, g.t_inicio - 0.5); }} />

        <section aria-label="Corregir" className="rounded-lg border border-borde p-3">
          <p className="mb-2 text-sm text-texto-2">
            {elegido ? `Golpe de ${elegido.mano} en ${tiempo(elegido.t_inicio)}` : "Tocá un golpe en la línea de tiempo para corregirlo."}
          </p>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" disabled={!elegido} onClick={() => corregir("no_fue_golpe")}>No fue un golpe</Button>
            <Button size="sm" variant="outline" disabled={!elegido} onClick={() => corregir("otra_mano")}>Era la otra mano</Button>
            <Button size="sm" variant="outline" onClick={() => corregir("falto_golpe")}>Faltó uno acá ({tiempo(actual)})</Button>
          </div>
          {aviso && <p role="status" className="mt-2 text-xs text-texto-3">{aviso}</p>}
        </section>

        <Metricas caption="Métricas de la sesión" columnas={[{ nombre: "Vos", m: u }, ...(rv ? [{ nombre: "Rival", m: rv }] : [])]} />
        {r.rotulos.guardia_rival && <p className="text-xs text-tenue">Rival: {r.rotulos.guardia_rival} (ortodoxa).</p>}

        {(r.metricas.usuario?.segmentos.length ?? 0) > 1 && (
          <section>
            <h2 className="rotulo mb-2">Por round <Rotulo>{TEXTOS.roundsSinValidar}</Rotulo></h2>
            <ul className="grid grid-cols-3 gap-2">
              {r.metricas.usuario!.segmentos.map((g) => (
                <li key={g.segmento} className="rounded-md border border-borde p-2 text-center">
                  <span className="rotulo block">R{g.segmento}</span>
                  <span className="dato block text-lg">{num(g.volumen)}</span>
                  <span className="text-xs text-tenue">por minuto</span>
                </li>
              ))}
            </ul>
          </section>
        )}

        <Curva series={[
          { nombre: "Vos", valores: u?.curva ?? [], color: "var(--color-texto)" },
          ...(rv ? [{ nombre: "Rival", valores: rv.curva ?? [], color: "var(--color-b-trazo)" }] : []),
        ]} />

        {r.exposicion && (
          <section aria-labelledby="t-expo" className="rounded-lg border border-borde bg-superficie p-3">
            <h2 id="t-expo" className="rotulo mb-2">Golpes del rival por lado</h2>
            <div className="grid grid-cols-2 gap-2 text-center">
              <div><span className="dato block text-2xl">{r.exposicion.lado_izquierdo_usuario}</span><span className="text-xs text-tenue">por tu izquierda</span></div>
              <div><span className="dato block text-2xl">{r.exposicion.lado_derecho_usuario}</span><span className="text-xs text-tenue">por tu derecha</span></div>
            </div>
            <p className="mt-2 text-xs text-tenue">{r.exposicion.aclaracion}</p>
          </section>
        )}

        {r.altura && (
          <section className="rounded-lg border border-borde bg-superficie p-3">
            <h2 className="rotulo mb-2">Cabeza y cuerpo <Rotulo>{TEXTOS.estimacion}</Rotulo></h2>
            <p className="text-sm">Cabeza <span className="dato">{r.altura.cabeza}</span> · Cuerpo <span className="dato">{r.altura.cuerpo}</span></p>
            <p className="mt-1 text-xs text-tenue">Es una {TEXTOS.estimacion} por altura del golpe; la regla acertó 0,726 en la medición del 13-10.</p>
          </section>
        )}

        <Consejos c={r.consejos} />
        <p className="text-xs text-tenue">Calidad de pose {pct(s.calidad_pose)}.</p>

        {pendiente ? (
          <section aria-labelledby="t-val" className="rounded-lg border border-estimado-borde p-3">
            <h2 id="t-val" className="mb-2 font-medium">Valorá esta sesión para cerrar el resultado</h2>
            <Valoracion id={s.id} alValorar={recargar} />
          </section>
        ) : (
          <Link href="/inicio/" className="block text-center text-sm underline underline-offset-4">Volver al inicio</Link>
        )}
        <PublicidadDelPlan ubicacion="resultado" />
      </div>

      <dialog ref={dialogo} aria-labelledby="t-dlg" className="m-auto w-[min(92vw,24rem)] rounded-lg border border-borde bg-superficie p-4 text-texto backdrop:bg-black/70">
        <h2 id="t-dlg" className="mb-3 font-medium">Antes de salir, valorá la sesión</h2>
        <Valoracion id={s.id} alValorar={() => { dialogo.current?.close(); router.push(destino); }} />
      </dialog>
    </Marco>
  );
}

function Contenido() {
  const id = useSearchParams().get("id") ?? "";
  const [s, setS] = useState<SesionDetalle | null>(null);
  const [r, setR] = useState<Resultado | null>(null);
  const [n, setN] = useState(0);
  useEffect(() => {
    let vivo = true;
    let t: ReturnType<typeof setTimeout>;
    const mirar = async () => {
      try {
        const x = await api<SesionDetalle>(`/sesiones/${id}`);
        if (!vivo) return;
        setS(x);
        if (x.estado === "lista") setR(await api<Resultado>(`/sesiones/${id}/resultado`));
        else if (x.estado !== "error" && x.estado !== "esperando_siembra") t = setTimeout(mirar, 4000);
      } catch { /* la pagina muestra lo ultimo que tiene */ }
    };
    mirar();
    return () => { vivo = false; clearTimeout(t); };
  }, [id, n]);
  if (r) return <Lista r={r} recargar={() => setN((x) => x + 1)} />;
  return (
    <Marco titulo="Sesión">
      {!s && <p className="rotulo">Cargando…</p>}
      {s?.estado === "error" && <p role="alert" className="text-error">No se pudo procesar ({s.error_etapa}). Los minutos de cupo se devolvieron.</p>}
      {s?.estado === "esperando_siembra" && (
        <Link href={`/siembra/?id=${s.id}`} className="objetivo-tactil flex items-center justify-center rounded-lg bg-primary font-medium text-primary-foreground">Marcar quién es quién</Link>
      )}
      {s && ESTADO[s.estado] && s.estado !== "esperando_siembra" && (
        <p aria-live="polite" className="text-texto-2">{ESTADO[s.estado]}… <span className="rotulo">se actualiza solo</span></p>
      )}
    </Marco>
  );
}

export default function Sesion() {
  return <Suspense><Contenido /></Suspense>;
}
