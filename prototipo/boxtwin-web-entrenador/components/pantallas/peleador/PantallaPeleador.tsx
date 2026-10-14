"use client";

/*
 * 06 · Perfil tactico (04 §06). Ruta /peleador/?id=&contexto=.
 *
 * Se llama "perfil" y no "gemelo digital": no hay un modelo que simule al boxeador (04 §06).
 * Cada bloque lleva su estado (01 §4): la distribucion es estimacion, las combinaciones
 * son medidas (por tiempo y brazo, regla del 14-10; sus tipos van aparte como estimacion, W13),
 * la evolucion y la tabla por contexto son medidas, lo de rounds espera a C2 y lo de guardia
 * a C3, y el mapa de exposicion, el avatar y la distancia no estan en esta version.
 *
 * El perfil no se cachea (CA-06.2): una correccion se ve al volver.
 */
import { Desplazable } from "@/components/boxtwin/Desplazable";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Encabezado } from "@/components/boxtwin/Encabezado";
import { EstadoBloque, Estimado } from "@/components/boxtwin/EstadoBloque";
import { ErrorConReintento, Esqueleto } from "@/components/boxtwin/Estados";
import { Rotulo } from "@/components/boxtwin/Rotulo";
import { LineaEvolucion } from "@/components/boxtwin/graficos/LineaEvolucion";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/animate-ui/components/radix/toggle-group";
import { usePerfil, useProducto } from "@/lib/api/consultas";
import { CONTEXTOS, TIPOS_GOLPE, type Perfil, type Producto } from "@/lib/api/esquemas";
import { estadoDe } from "@/lib/banderas";
import { decimal, nombreDeTipo, porcentajeAR } from "@/lib/formato";
import { NOMBRE_CONTEXTO, textos } from "@/lib/textos";

export function PantallaPeleador() {
  const q = useSearchParams();
  const router = useRouter();
  const id = q.get("id");
  const contexto = q.get("contexto") ?? "todas";
  const perfil = usePerfil(id, contexto);
  const producto = useProducto();

  if (!id) return <ErrorConReintento error={new Error("Falta el identificador del peleador.")} />;
  if (perfil.isPending) return <Esqueleto filas={4} />;
  if (perfil.isError) return <ErrorConReintento error={perfil.error} alReintentar={() => perfil.refetch()} />;
  const p = perfil.data;

  function filtrar(v: string) {
    router.replace(v === "todas" ? `/peleador/?id=${id}` : `/peleador/?id=${id}&contexto=${v}`);
  }

  return (
    <Contenido p={p} producto={producto.data} contexto={contexto} onContexto={filtrar} />
  );
}

function Contenido({ p, producto, contexto, onContexto }: { p: Perfil; producto?: Producto; contexto: string; onContexto: (v: string) => void }) {
  const b = p.boxeador;
  const porCtx = p.sesiones_por_contexto ?? {};
  const total = Object.values(porCtx).reduce((a, n) => a + n, 0) || p.totales.sesiones;
  const preliminar = total < (producto?.dominio.perfil_preliminar_sesiones ?? 5);
  const datos = [b.categoria, b.guardia, b.altura_m ? `${decimal(b.altura_m, 2)} m` : null, b.envergadura_m ? `envergadura ${decimal(b.envergadura_m, 2)} m` : null].filter(Boolean).join(" · ");

  return (
    <>
      <Encabezado
        atras="/peleadores/"
        rotulo="Perfil táctico"
        titulo={b.nombre}
        accion={<Button asChild variant="outline" className="min-h-11"><Link href={`/rival/?a=${b.id}`}>Comparar en Rival Fantasma</Link></Button>}
      >
        <p className="text-texto-2">{datos || "sin datos cargados"}</p>
        <p className="rotulo">{p.totales.sesiones} {p.totales.sesiones === 1 ? "sesión" : "sesiones"} · {p.rounds_total ?? 0} rounds</p>
      </Encabezado>

      {preliminar && (
        <p className="mb-4 rounded-md border border-estimado-borde bg-estimado-fondo px-3 py-2 text-sm">
          Perfil preliminar: con {producto?.dominio.perfil_preliminar_sesiones ?? 5} sesiones o más las tendencias se leen mejor.
        </p>
      )}

      <div className="-mx-4 mb-4 overflow-x-auto px-4">
        <ToggleGroup type="single" value={contexto} onValueChange={(v) => v && onContexto(v)} aria-label="Contexto" className="flex w-max gap-1">
          <ToggleGroupItem value="todas" className="min-h-11 border border-borde px-3">Todas · {total}</ToggleGroupItem>
          {CONTEXTOS.map((c) => (
            <ToggleGroupItem key={c} value={c} disabled={!porCtx[c]} className="min-h-11 border border-borde px-3">
              {NOMBRE_CONTEXTO[c]} · {porCtx[c] ?? 0}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>

      {p.avisos.length > 0 && (
        <ul className="mb-4 space-y-1 text-xs text-texto-3">
          {p.avisos.map((a) => <li key={a}>{a}</li>)}
        </ul>
      )}

      <div className="grid gap-4 lg:grid-cols-3">
        <Distribucion p={p} producto={producto} />
        <EstadoBloque estado={estadoDe("06.evolucion", producto)} titulo="Evolución entre sesiones" className="lg:col-span-2">
          <LineaEvolucion evolucion={p.evolucion} />
        </EstadoBloque>
        <Contextos p={p} producto={producto} />
        <VolumenPorRound p={p} producto={producto} />
        <Combinaciones p={p} producto={producto} />
        <div className="flex flex-col gap-3 lg:col-span-3">
          <EstadoBloque estado={estadoDe("06.exposicion", producto)} titulo="Mapa de exposición" motivo={textos.nodispExposicion} />
          <EstadoBloque estado={estadoDe("06.avatar", producto)} titulo="Avatar 3D" motivo={textos.nodisp3d} />
          <EstadoBloque estado={estadoDe("06.distancia", producto)} titulo="Distancia de trabajo" motivo={textos.nodispDistancia} />
        </div>
      </div>
    </>
  );
}

function Distribucion({ p, producto }: { p: Perfil; producto?: Producto }) {
  const mezcla = p.totales.mezcla;
  const clasificados = TIPOS_GOLPE.reduce((a, t) => a + (mezcla[t] ?? 0), 0);
  const clasificador = p.sesiones.find((s) => s.clasificador)?.clasificador ?? null;
  const exactitud = p.sesiones.find((s) => s.exactitud_familia != null)?.exactitud_familia ?? null;
  return (
    <EstadoBloque
      estado={estadoDe("06.distribucion", producto)}
      titulo="Distribución de golpes"
      nota={textos.estimacionTipo(clasificador, exactitud)}
    >
      {clasificados === 0 ? (
        <p className="text-sm text-texto-3">Ningún golpe tiene tipo estimado todavía.</p>
      ) : (
        <ul className="flex flex-col gap-3">
          {TIPOS_GOLPE.map((t) => {
            const n = mezcla[t] ?? 0;
            const f = n / clasificados;
            return (
              <li key={t}>
                <div className="flex items-baseline justify-between text-sm">
                  <span>{nombreDeTipo(t, p.nomenclatura)}{t === "hook" && <span className="ml-2 text-xs text-estimado">tipo menos confiable</span>}</span>
                  <Estimado className="dato">{porcentajeAR(f)} · {n}</Estimado>
                </div>
                <div aria-hidden="true" className="mt-1 h-1.5 rounded-full bg-superficie-2">
                  <div className="h-full rounded-full bg-estimado/70" style={{ width: `${Math.round(f * 100)}%` }} />
                </div>
              </li>
            );
          })}
        </ul>
      )}
      <p className="mt-3 text-xs text-texto-3">{p.totales.golpes} detectados, {p.totales.sin_clasificar} sin clasificar.</p>
    </EstadoBloque>
  );
}

function Contextos({ p, producto }: { p: Perfil; producto?: Producto }) {
  const conGuardia = estadoDe("06.contextos_guardia", producto) === "medido";
  const conR3 = estadoDe("06.contextos_r3", producto) === "medido";
  const filas = p.por_contexto ?? [];
  return (
    <EstadoBloque estado={estadoDe("06.contextos", producto)} titulo="Mismo peleador en distintos contextos" className="lg:col-span-2">
      {filas.length === 0 ? (
        <p className="text-sm text-texto-3">Todavía no hay sesiones con contexto.</p>
      ) : (
        <Desplazable nombre="Tabla por contexto">
          <table className="w-full min-w-[480px] text-sm">
            <caption className="sr-only">Métricas por contexto</caption>
            <thead>
              <tr className="text-left">
                <th scope="col" className="rotulo py-2 pr-3 font-normal">Contexto</th>
                <th scope="col" className="rotulo py-2 pr-3 text-right font-normal">Golpes/min</th>
                <th scope="col" className="rotulo py-2 pr-3 text-right font-normal">Jab / total</th>
                <th scope="col" className="rotulo py-2 pr-3 text-right font-normal">Guardia arriba</th>
                <th scope="col" className="rotulo py-2 text-right font-normal">Caída último round</th>
              </tr>
            </thead>
            <tbody>
              {filas.map((f) => (
                <tr key={f.contexto} className="border-t border-borde-suave">
                  <th scope="row" className="py-2 pr-3 text-left font-normal">{NOMBRE_CONTEXTO[f.contexto] ?? f.contexto}</th>
                  <td className="dato py-2 pr-3 text-right">{decimal(f.golpes_por_minuto, 1)}</td>
                  <td className="dato py-2 pr-3 text-right">{f.jab_fraccion == null ? "—" : <Estimado>{porcentajeAR(f.jab_fraccion)}</Estimado>}</td>
                  <td className="py-2 pr-3 text-right">{conGuardia ? <span className="dato">{porcentajeAR(f.guardia_opuesta_arriba)}</span> : <Rotulo>en validación</Rotulo>}</td>
                  <td className="py-2 text-right">{conR3 ? <span className="dato">{f.caida_ultimo_round == null ? "—" : porcentajeAR(f.caida_ultimo_round)}</span> : <Rotulo>en validación</Rotulo>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Desplazable>
      )}
      <p className="mt-3 text-sm text-texto-2">{textos.contextoLectura}</p>
      {(!conGuardia || !conR3) && (
        <p className="mt-1 text-xs text-texto-3">
          {!conGuardia && textos.condicionadoC3} {!conR3 && textos.condicionadoC2}
        </p>
      )}
    </EstadoBloque>
  );
}

function VolumenPorRound({ p, producto }: { p: Perfil; producto?: Producto }) {
  const rondas = p.rounds_promedio ?? [];
  const max = Math.max(1, ...rondas.map((r) => r.golpes_por_minuto));
  return (
    <EstadoBloque estado={estadoDe("06.volumen_round", producto)} titulo="Volumen por round promedio" motivo={textos.condicionadoC2}>
      {rondas.length === 0 ? (
        <p className="text-sm text-texto-3">Sin sesiones con rounds declarados.</p>
      ) : (
        <ul className="flex flex-col gap-2">
          {rondas.map((r) => (
            <li key={r.round} className="grid grid-cols-[2.5rem_1fr_3.5rem] items-center gap-2 text-sm">
              <span className="dato">R{r.round}</span>
              <span aria-hidden="true" className="h-2 rounded-full bg-superficie-2"><span className="block h-full rounded-full bg-texto-2" style={{ width: `${(r.golpes_por_minuto / max) * 100}%` }} /></span>
              <span className="dato text-right">{decimal(r.golpes_por_minuto, 1)}/min</span>
            </li>
          ))}
        </ul>
      )}
    </EstadoBloque>
  );
}

function Combinaciones({ p, producto }: { p: Perfil; producto?: Producto }) {
  const c = p.combinaciones;
  const regla = producto?.dominio.combinacion;
  return (
    <EstadoBloque
      estado={estadoDe("06.combinaciones", producto)}
      titulo="Combinaciones"
      className="lg:col-span-3"
      nota={`Dos golpes o más, con cualquier mano, con menos de ${decimal(regla?.max_s ?? 0.5, 1)} s del inicio de un golpe al del siguiente. Una serie larga cuenta como una sola.`}
    >
      {!c ? (
        <p className="text-sm text-texto-3">Esta instancia no puede contar combinaciones.</p>
      ) : (
        <div className="flex flex-col gap-4">
          {c.total === 0 ? (
            <p className="text-sm text-texto-2">El sistema no encontró combinaciones completas en estas sesiones.</p>
          ) : (
            <dl className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <div><dt className="rotulo">Total</dt><dd className="font-display text-3xl font-semibold">{c.total}</dd></div>
              <div><dt className="rotulo">Por minuto visto</dt><dd className="font-display text-3xl font-semibold">{c.por_minuto == null ? "—" : decimal(c.por_minuto, 2)}</dd></div>
              {Object.entries(c.por_contexto).map(([ctx, n]) => (
                <div key={ctx}><dt className="rotulo">{NOMBRE_CONTEXTO[ctx] ?? ctx}</dt><dd className="font-display text-3xl font-semibold">{n}</dd></div>
              ))}
            </dl>
          )}
          {/* Medido: que secuencias de manos se repiten y de cuantos golpes son. */}
          {(c.secuencias_frecuentes?.length ?? 0) > 0 && (
            <div>
              <p className="text-sm text-texto-2">Secuencias de manos más frecuentes:</p>
              <ol className="mt-2 grid gap-2 md:grid-cols-3">
                {c.secuencias_frecuentes!.map((x) => (
                  <li key={x.brazos.join("-")} className="flex items-center justify-between">
                    <span className="font-mono text-sm uppercase tracking-[0.06em]">{x.brazos.join("-")}</span>
                    <span className="dato text-texto-2">{x.n}</span>
                  </li>
                ))}
              </ol>
              {c.por_largo && (
                <p className="mt-2 text-xs text-texto-3">
                  {Object.entries(c.por_largo).sort(([a], [b]) => a.localeCompare(b)).map(([k, n]) => `${n} de ${k} golpes`).join(" · ")}
                </p>
              )}
            </div>
          )}
          {/* El recall sale de la Fight-Card mas reciente; sin ninguna leida no hay numero que calificar. */}
          {c.recall_medido != null && (
            <p className="rounded-md border-l-2 border-estimado-borde bg-estimado-fondo px-3 py-2 text-sm text-texto-2">
              {textos.margenCombinaciones(c.recall_medido)}
            </p>
          )}
          {c.tipos_frecuentes.length > 0 && (
            <div className="rounded-sm border border-dashed border-estimado-borde p-3">
              <Rotulo className="text-estimado">estimación</Rotulo>
              <p className="mt-1 text-sm text-texto-2">Tipos más frecuentes dentro de esas combinaciones:</p>
              <ol className="mt-2 grid gap-2 md:grid-cols-3">
                {c.tipos_frecuentes.map((t) => (
                  <li key={t.tipos.join("-")} className="flex items-center justify-between">
                    <Estimado>{t.tipos.map((x) => nombreDeTipo(x, p.nomenclatura)).join(" · ")}</Estimado>
                    <span className="dato text-texto-2">{t.n}</span>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}
    </EstadoBloque>
  );
}
