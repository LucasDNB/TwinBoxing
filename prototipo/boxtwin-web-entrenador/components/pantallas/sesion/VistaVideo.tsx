"use client";

/*
 * 04 · Video y golpes (04 §04).
 *
 * Celular: video arriba y pegado al tope, linea de tiempo, selector A / B y la lista de
 * golpes de ese peleador cerca del tiempo actual, nota fija al pie. Escritorio: video y
 * linea a la izquierda, dos consolas (A y B) a la derecha.
 *
 * Tocar un golpe lleva el video a t_inicio - 0,5 s y lo reproduce (CA-04.1). El video se
 * pide por rangos con un ticket en la URL, que se renueva a los 25 minutos sin perder la
 * posicion (02 §7).
 */
import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { Switch } from "@/components/animate-ui/components/radix/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/animate-ui/components/radix/toggle-group";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { Esquina } from "@/components/boxtwin/Esquina";
import { Rotulo } from "@/components/boxtwin/Rotulo";
import { Estimado } from "@/components/boxtwin/EstadoBloque";
import { mensajeDeError } from "@/components/boxtwin/Estados";
import { Correccion } from "@/components/pantallas/sesion/Correccion";
import { LineaTiempo, type Ventana } from "@/components/pantallas/sesion/LineaTiempo";
import { nombresDe } from "@/components/pantallas/sesion/PantallaSesion";
import { useProcesado, useProducto, useRender, useTicket } from "@/lib/api/consultas";
import type { Golpe, Sesion } from "@/lib/api/esquemas";
import { visible } from "@/lib/banderas";
import { guardiaPorGolpe, secuenciaPorGolpe, lados as ladosDe, ladosConGuardiaSupuesta, tipoEfectivo, type FightCard, type Lado } from "@/lib/fightcard";
import { decimal, mmss, nombreDeTipo } from "@/lib/formato";
import { useEsEscritorio } from "@/lib/medios";
import { textos } from "@/lib/textos";
import { cn } from "@/lib/utils";

export const ANTES_DEL_GOLPE_S = 0.5;
const VELOCIDADES = ["0.25", "0.5", "1"];

export function urlDeVideo(id: string, ticket: string, procesado: boolean): string {
  return `/videos/${id}/${procesado ? "procesado" : "stream"}?t=${encodeURIComponent(ticket)}`;
}

export function VistaVideo({ s, fc }: { s: Sesion; fc: FightCard }) {
  const q = useSearchParams();
  const producto = useProducto();
  const ticket = useTicket(s.id);
  const procesado = useProcesado(s.id, ticket.data?.ticket);
  const render = useRender(s.id);
  const escritorio = useEsEscritorio();
  const video = useRef<HTMLVideoElement>(null);
  const posicionAntesDeCambiar = useRef<number | null>(null);
  const inicial = useRef<number | null>(Number(q.get("t")) || null);

  const [esqueleto, setEsqueleto] = useState(true);
  const [velocidad, setVelocidad] = useState("1");
  const [tiempo, setTiempo] = useState(0);
  const [lado, setLado] = useState<Lado>("A");
  const [seleccionado, setSeleccionado] = useState<{ lado: Lado; id: string } | null>(null);
  const [zoom, setZoom] = useState<"sesion" | "round">("sesion");

  const nombres = nombresDe(s, fc);
  const nombreDe = { A: nombres.A ?? "Peleador A", B: nombres.B ?? "Peleador B" };
  const lados = ladosDe(fc);
  const conGuardia = visible("04.guardia", producto.data);
  const conAltura = visible("04.altura", producto.data);
  const hayProcesado = procesado.data === true;
  const usarProcesado = hayProcesado && esqueleto;
  const src = ticket.data ? urlDeVideo(s.id, ticket.data.ticket, usarProcesado) : undefined;

  // Cambiar de fuente (ticket renovado o esqueleto) no pierde la posicion.
  const srcPrevio = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (srcPrevio.current && srcPrevio.current !== src && video.current) {
      posicionAntesDeCambiar.current = video.current.currentTime;
    }
    srcPrevio.current = src;
  }, [src]);

  useEffect(() => {
    if (video.current) video.current.playbackRate = Number(velocidad);
  }, [velocidad]);

  const saltar = useCallback((l: Lado, g: Golpe) => {
    setLado(l);
    setSeleccionado({ lado: l, id: g.id });
    const v = video.current;
    if (!v) return;
    v.currentTime = Math.max(0, g.t_inicio - ANTES_DEL_GOLPE_S);
    setTiempo(v.currentTime);
    void v.play()?.catch(() => {});
  }, []);

  const ronda = fc.video.rounds.find((r) => tiempo >= r.inicio_s && tiempo < r.fin_s) ?? fc.video.rounds[0];
  const ventana: Ventana = zoom === "round" && ronda ? { inicio: ronda.inicio_s, fin: ronda.fin_s } : { inicio: 0, fin: fc.video.duracion_s };

  // Atajos de escritorio: solo con el foco dentro del reproductor o de la linea (04 §04).
  function atajos(e: KeyboardEvent<HTMLDivElement>) {
    const v = video.current;
    if (!v) return;
    const destino = e.target as HTMLElement;
    if (e.key === " " && destino.tagName !== "VIDEO" && destino.tagName !== "BUTTON") {
      e.preventDefault();
      if (v.paused) void v.play()?.catch(() => {});
      else v.pause();
    }
    if (e.key === "j" || e.key === "l" || e.key === "J" || e.key === "L") {
      const propios = [...(fc.peleadores[lado]?.golpes ?? [])].sort((a, b) => a.t_inicio - b.t_inicio);
      const t = v.currentTime + ANTES_DEL_GOLPE_S;
      const g = e.key.toLowerCase() === "l"
        ? propios.find((x) => x.t_inicio > t + 0.05)
        : [...propios].reverse().find((x) => x.t_inicio < t - 0.05);
      if (g) {
        e.preventDefault();
        saltar(lado, g);
      }
    }
  }

  const reproductor = (
    <div className="flex flex-col gap-3">
      <div className="sticky top-0 z-20 -mx-4 bg-fondo px-4 pb-2 pt-1 lg:static lg:mx-0 lg:p-0">
        <div className="aspect-video w-full overflow-hidden rounded-md border border-borde bg-black">
          {src ? (
            <video
              ref={video}
              src={src}
              controls
              playsInline
              preload="metadata"
              className="h-full w-full"
              aria-label="Video de la sesión"
              onTimeUpdate={(e) => setTiempo(e.currentTarget.currentTime)}
              onSeeked={(e) => setTiempo(e.currentTarget.currentTime)}
              onLoadedMetadata={(e) => {
                const v = e.currentTarget;
                v.playbackRate = Number(velocidad);
                const volver = posicionAntesDeCambiar.current ?? inicial.current;
                if (volver != null) {
                  v.currentTime = volver;
                  posicionAntesDeCambiar.current = null;
                  inicial.current = null;
                }
              }}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-texto-3">
              {ticket.isError ? mensajeDeError(ticket.error) : "Cargando video…"}
            </div>
          )}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <ToggleGroup type="single" value={velocidad} onValueChange={(v) => v && setVelocidad(v)} aria-label="Velocidad" className="gap-1">
          {VELOCIDADES.map((x) => (
            <ToggleGroupItem key={x} value={x} className="min-h-11 min-w-11 border border-borde px-2 dato">{decimal(Number(x), x === "1" ? 0 : 2)}×</ToggleGroupItem>
          ))}
        </ToggleGroup>
        {hayProcesado ? (
          <label className="flex min-h-11 items-center gap-2 text-sm">
            <Switch checked={esqueleto} onCheckedChange={setEsqueleto} aria-label="Esqueleto" />
            Esqueleto
          </label>
        ) : procesado.isFetched ? (
          <Button variant="outline" className="min-h-11" disabled={render.isPending || render.isSuccess}
            onClick={() => render.mutate(undefined, {
              onSuccess: () => toast("Se está armando el video con esqueleto. Mientras, ves el original."),
              onError: (e) => toast.error(mensajeDeError(e)),
            })}>
            {render.isSuccess ? "Armando video con esqueleto…" : "Armar video con esqueleto"}
          </Button>
        ) : null}
      </div>
      <div className="flex items-center justify-between gap-3">
        <h2 className="font-display text-xl font-semibold lg:text-2xl">Golpes detectados</h2>
        {fc.video.rounds.length > 1 && (
          <ToggleGroup type="single" value={zoom} onValueChange={(v) => v && setZoom(v as "sesion" | "round")} aria-label="Zoom de la línea de tiempo" className="gap-1">
            <ToggleGroupItem value="sesion" className="min-h-11 border border-borde px-3">Sesión</ToggleGroupItem>
            <ToggleGroupItem value="round" className="min-h-11 border border-borde px-3">Round</ToggleGroupItem>
          </ToggleGroup>
        )}
      </div>
      <LineaTiempo fc={fc} nombres={nombreDe} tiempo={tiempo} seleccionado={seleccionado} onElegir={saltar} ventana={ventana} />
    </div>
  );

  const nota = (
    <p className="mt-4 rounded-md border border-borde bg-superficie-2 p-3 text-xs text-texto-3">
      {textos.notaCorreccion(fc.clasificador.checkpoint)} &quot;No fue un golpe&quot; tampoco cambia el conteo de detectados: se cuenta aparte como descartados por el entrenador.
      {/* Modo académico E7 (25-10): la licencia de SMPL, también al pie del video. */}
      {textos.modoAcademico(fc.clasificador.metodo, fc.clasificador.licencia) && (
        <span role="note" data-modo-academico className="mt-2 block text-texto-2">
          {textos.modoAcademico(fc.clasificador.metodo, fc.clasificador.licencia)}
        </span>
      )}
      {ladosConGuardiaSupuesta(fc).map((l) => (
        <span key={l} role="note" data-aviso-guardia={l} className="mt-2 block text-texto-2">
          {textos.avisoGuardiaSupuesta(nombres[l] ?? l)}
        </span>
      ))}
    </p>
  );

  return (
    <div onKeyDown={atajos}>
      {escritorio ? (
        <div className="grid grid-cols-[1.5fr_1fr] gap-6">
          <div>{reproductor}{nota}</div>
          <div className="flex flex-col gap-4">
            {lados.map((l) => (
              <Consola key={l} s={s} fc={fc} lado={l} nombre={nombreDe[l]} tiempo={tiempo} seleccionado={seleccionado} onElegir={saltar} conGuardia={conGuardia} conAltura={conAltura} />
            ))}
          </div>
        </div>
      ) : (
        <>
          {reproductor}
          {lados.length > 1 && (
            <ToggleGroup type="single" value={lado} onValueChange={(v) => v && setLado(v as Lado)} aria-label="Peleador" className="mt-4 grid w-full grid-cols-2 gap-1">
              {lados.map((l) => (
                <ToggleGroupItem key={l} value={l} className="min-h-11 border border-borde">
                  <span className="flex items-center gap-2"><Esquina lado={l} tamano={18} />{nombreDe[l]}</span>
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
          )}
          <div className="mt-3">
            <Consola s={s} fc={fc} lado={lado} nombre={nombreDe[lado]} tiempo={tiempo} seleccionado={seleccionado} onElegir={saltar} conGuardia={conGuardia} conAltura={conAltura} sinTitulo={lados.length > 1} />
          </div>
          {nota}
        </>
      )}
    </div>
  );
}

/* La lista de golpes de un peleador, que sigue al video mientras corre (04 §04, punto 3). */
function Consola({
  s, fc, lado, nombre, tiempo, seleccionado, onElegir, conGuardia, conAltura, sinTitulo = false,
}: {
  s: Sesion; fc: FightCard; lado: Lado; nombre: string; tiempo: number;
  seleccionado: { lado: Lado; id: string } | null; onElegir: (l: Lado, g: Golpe) => void;
  conGuardia: boolean; conAltura: boolean; sinTitulo?: boolean;
}) {
  const p = fc.peleadores[lado];
  const golpes = useMemo(() => [...(p?.golpes ?? [])].sort((a, b) => a.t_inicio - b.t_inicio), [p]);
  const guardia = useMemo(() => (p ? guardiaPorGolpe(p) : new Map()), [p]);
  const secuencia = useMemo(() => secuenciaPorGolpe(fc, lado), [fc, lado]);
  const lista = useRef<HTMLOListElement>(null);
  const actual = useMemo(() => {
    let k = -1;
    for (let i = 0; i < golpes.length; i++) if (golpes[i].t_inicio <= tiempo + ANTES_DEL_GOLPE_S) k = i;
    return k;
  }, [golpes, tiempo]);

  useEffect(() => {
    const el = lista.current?.children[Math.max(actual, 0)] as HTMLElement | undefined;
    if (el && lista.current) lista.current.scrollTop = el.offsetTop - lista.current.offsetTop - 40;
  }, [actual]);

  if (!p) return null;
  return (
    <section aria-label={`Golpes de ${nombre}`} className="rounded-md border border-borde bg-superficie">
      {!sinTitulo && (
        <h3 className="flex items-center gap-2 border-b border-borde-suave px-4 py-3 font-display text-lg font-semibold">
          <Esquina lado={lado} /> {nombre} <span className="dato ml-auto text-texto-2">{p.total.total} detectados</span>
        </h3>
      )}
      {golpes.length === 0 ? (
        <p className="p-4 text-sm text-texto-3">El sistema no detectó golpes de {nombre} en esta sesión.</p>
      ) : (
        <ol ref={lista} className="max-h-[360px] divide-y divide-borde-suave overflow-y-auto lg:max-h-[420px]">
          {golpes.map((g, i) => {
            const t = tipoEfectivo(g);
            const baja = !g.corregido && g.confianza_tipo != null && g.confianza_tipo < 0.5;
            const ev = guardia.get(g.id);
            const sel = seleccionado?.id === g.id;
            return (
              <li key={g.id} className={cn("flex items-center gap-2 px-3 py-1.5", i === actual && "bg-elevado/60", sel && "bg-elevado")}>
                <button type="button" onClick={() => onElegir(lado, g)} className="flex min-h-11 flex-1 items-center gap-3 text-left" aria-label={`Ir al golpe de ${mmss(g.t_inicio)}`}>
                  <span className="dato w-12 shrink-0 text-texto-2">{mmss(g.t_inicio)}</span>
                  <span className="min-w-0 flex-1">
                    {g.descartado ? (
                      <span className="text-ok">no fue un golpe <Rotulo className="ml-1 text-ok">corregido</Rotulo></span>
                    ) : g.corregido ? (
                      <span>
                        <span className="text-ok">{nombreDeTipo(t, fc.nomenclatura)}</span>
                        <Rotulo className="ml-1 text-ok">corregido</Rotulo>
                        {g.corregido.tipo_original && <s className="ml-2 text-texto-3">{nombreDeTipo(g.corregido.tipo_original, fc.nomenclatura)}</s>}
                      </span>
                    ) : t ? (
                      <Estimado>
                        {nombreDeTipo(t, fc.nomenclatura)}
                        {g.confianza_tipo != null && <span className={cn("ml-1 dato text-xs", baja ? "text-estimado" : "text-texto-3")}>{decimal(g.confianza_tipo, 2)}{baja ? " baja" : ""}</span>}
                      </Estimado>
                    ) : (
                      <span className="text-texto-3">sin tipo</span>
                    )}
                    <span className="block text-xs text-texto-3">
                      {secuencia.has(g.id) && (
                        <span className="mr-2 inline-flex items-center rounded-sm border border-borde-fuerte px-1 font-mono text-[10px] uppercase tracking-[0.08em] text-texto-2">{secuencia.get(g.id)}</span>
                      )}
                      brazo {g.brazo}
                      {/* W14: la altura es estimacion; corregida a mano va en verde, como el tipo. */}
                      {conAltura && !g.descartado && g.objetivo && (
                        g.objetivo_corregido
                          ? <span className="text-ok"> · {g.objetivo} (corregido)</span>
                          : <span className="text-estimado"> · ≈ {g.objetivo}</span>
                      )}
                      {conGuardia && ev?.mano_opuesta_caida != null && ` · guardia ${ev.mano_opuesta_caida ? "caída" : "arriba"}`}
                    </span>
                  </span>
                </button>
                <Correccion sesionId={s.id} golpe={g} nomenclatura={fc.nomenclatura} conAltura={conAltura} />
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
