"use client";

/*
 * 06 · Perfil tactico (04 §06). Ruta /peleador/?id=&contexto=.
 *
 * Se llama "perfil" y no "gemelo digital": no hay un modelo que simule al boxeador (04 §06).
 * Cada bloque lleva su estado (01 §4): la distribucion y las combinaciones son estimacion,
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
import { nombreDeCombinacion } from "@/lib/fightcard";
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
  const combos = p.combinaciones ?? [];
  const pausa = producto?.dominio.pausa_combinacion_s ?? 0.5;
  return (
    <EstadoBloque
      estado={estadoDe("06.combinaciones", producto)}
      titulo="Combinaciones frecuentes"
      className="lg:col-span-3"
      nota={`Golpes del mismo peleador separados por no más de ${decimal(pausa, 1)} s. ${p.boxeador.guardia ? "Numeración de gimnasio: 1 jab, 2 recto de atrás, 3 y 4 hook, 5 y 6 uppercut." : "Sin guardia cargada no se sabe qué mano es la de adelante: se nombran por familia."}`}
    >
      {combos.length === 0 ? (
        <p className="text-sm text-texto-3">Todavía no hay combinaciones con tipo estimado.</p>
      ) : (
        <ol className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
          {combos.map((c) => (
            <li key={JSON.stringify(c.secuencia)} className="flex items-center justify-between rounded-sm border border-dashed border-estimado-borde px-3 py-2">
              <Estimado className="font-display text-xl font-semibold">{nombreDeCombinacion(c, p.nomenclatura ?? {})}</Estimado>
              <span className="dato text-texto-2">{c.n} veces</span>
            </li>
          ))}
        </ol>
      )}
    </EstadoBloque>
  );
}
