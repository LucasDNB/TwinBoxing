"use client";

/*
 * 07 · Rival Fantasma (04 §07). Ruta /rival/?a=&b=.
 *
 * Lectura cualitativa entre dos perfiles ya construidos: no es un pronostico ni una
 * probabilidad de victoria. Solo metricas de sparring y combate de cada perfil. Las
 * lecturas salen de reglas fijas (lib/rival.ts) y cada una cita su metrica.
 */
import { Desplazable } from "@/components/boxtwin/Desplazable";
import { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Encabezado } from "@/components/boxtwin/Encabezado";
import { Esquina } from "@/components/boxtwin/Esquina";
import { EstadoBloque, Estimado } from "@/components/boxtwin/EstadoBloque";
import { ErrorConReintento, Esqueleto, Vacio } from "@/components/boxtwin/Estados";
import { Rotulo } from "@/components/boxtwin/Rotulo";
import { SelectNativo } from "@/components/boxtwin/Controles";
import { Siluetas } from "@/components/boxtwin/graficos/Siluetas";
import { ToggleGroup, ToggleGroupItem } from "@/components/animate-ui/components/radix/toggle-group";
import { useBoxeadores, usePerfil, useProducto } from "@/lib/api/consultas";
import type { Perfil, Producto } from "@/lib/api/esquemas";
import { estadoDe } from "@/lib/banderas";
import { decimal, porcentajeAR } from "@/lib/formato";
import { SIN_DIFERENCIAS, hayLecturas, lecturas, type FilaRival } from "@/lib/rival";
import { textos } from "@/lib/textos";

const DE_PELEA = ["sparring", "combate"];

/* Las metricas de sparring y combate de un perfil, promediadas por cantidad de sesiones. */
export function filaDe(p: Perfil): FilaRival {
  const filas = (p.por_contexto ?? []).filter((f) => DE_PELEA.includes(f.contexto));
  const peso = (f: (typeof filas)[number]) => f.sesiones ?? 1;
  const media = (k: "golpes_por_minuto" | "jab_fraccion" | "guardia_opuesta_arriba" | "caida_ultimo_round") => {
    const con = filas.filter((f) => f[k] != null);
    const w = con.reduce((a, f) => a + peso(f), 0);
    return w ? con.reduce((a, f) => a + (f[k] as number) * peso(f), 0) / w : null;
  };
  return {
    nombre: p.boxeador.nombre,
    altura_m: p.boxeador.altura_m ?? null,
    envergadura_m: p.boxeador.envergadura_m ?? null,
    guardia: p.boxeador.guardia,
    golpes_por_minuto: media("golpes_por_minuto"),
    jab_fraccion: media("jab_fraccion"),
    guardia_opuesta_arriba: media("guardia_opuesta_arriba"),
    caida_ultimo_round: media("caida_ultimo_round"),
  };
}

export function PantallaRival() {
  const router = useRouter();
  const q = useSearchParams();
  const a = q.get("a") ?? "";
  const b = q.get("b") ?? "";
  const boxeadores = useBoxeadores();
  const producto = useProducto();
  const pa = usePerfil(a || null);
  const pb = usePerfil(b || null);
  const [superpuestas, setSuperpuestas] = useState("lado");

  function elegir(lado: "a" | "b", id: string) {
    const u = new URLSearchParams(q.toString());
    if (id) u.set(lado, id);
    else u.delete(lado);
    router.replace(`/rival/?${u.toString()}`);
  }

  if (boxeadores.isPending) return <Esqueleto />;
  if (boxeadores.isError) return <ErrorConReintento error={boxeadores.error} alReintentar={() => boxeadores.refetch()} />;
  const lista = boxeadores.data;

  return (
    <>
      <Encabezado titulo="Rival Fantasma" rotulo="Comparar dos perfiles">
        <p className="text-texto-2">{textos.rivalAviso}</p>
      </Encabezado>

      {lista.length < 2 ? (
        <Vacio titulo="Hacen falta dos peleadores con perfil">Cargá peleadores y asignalos a sus sesiones.</Vacio>
      ) : (
        <div className="flex flex-col gap-4">
          <div className="grid gap-3 md:grid-cols-2">
            {(["a", "b"] as const).map((l) => (
              <div key={l} className="flex flex-col gap-1.5">
                <label htmlFor={`rival-${l}`} className="flex items-center gap-2 text-sm"><Esquina lado={l === "a" ? "A" : "B"} /> Perfil {l.toUpperCase()}</label>
                <SelectNativo id={`rival-${l}`} value={l === "a" ? a : b} onChange={(e) => elegir(l, e.target.value)}>
                  <option value="">Elegí un peleador</option>
                  {lista.map((x) => (
                    // CA-07.2: el mismo peleador no puede estar en los dos lados.
                    <option key={x.id} value={x.id} disabled={x.id === (l === "a" ? b : a)}>
                      {x.nombre} · {x.sesiones ?? 0} {x.sesiones === 1 ? "sesión" : "sesiones"}
                    </option>
                  ))}
                </SelectNativo>
              </div>
            ))}
          </div>

          {a && b && a === b && <p role="alert" className="text-sm text-error">Elegí dos peleadores distintos.</p>}

          {a && b && a !== b && (pa.isPending || pb.isPending) && <Esqueleto filas={3} />}
          {(pa.isError || pb.isError) && <ErrorConReintento error={pa.error ?? pb.error} alReintentar={() => { pa.refetch(); pb.refetch(); }} />}
          {a && b && a !== b && pa.data && pb.data && <Comparacion pa={pa.data} pb={pb.data} producto={producto.data} superpuestas={superpuestas} onSuperpuestas={setSuperpuestas} />}
        </div>
      )}
    </>
  );
}

function Comparacion({ pa, pb, producto, superpuestas, onSuperpuestas }: { pa: Perfil; pb: Perfil; producto?: Producto; superpuestas: string; onSuperpuestas: (v: string) => void }) {
  const fa = filaDe(pa);
  const fb = filaDe(pb);
  const banderas = { comparacion_rounds: estadoDe("07.frente_r3", producto) === "medido", guardia: estadoDe("07.frente_guardia", producto) === "medido" };
  const lec = lecturas(fa, fb, banderas);
  const conRounds = banderas.comparacion_rounds;
  const promedioRounds = (p: Perfil) => {
    const r = p.rounds_promedio ?? [];
    return r.length ? r.reduce((s, x) => s + x.golpes_por_minuto, 0) / r.length : null;
  };

  type Fila = { nombre: string; a: React.ReactNode; b: React.ReactNode; estado?: "apagado" };
  const filas: Fila[] = [
    { nombre: "Altura", a: fa.altura_m ? `${decimal(fa.altura_m, 2)} m` : "—", b: fb.altura_m ? `${decimal(fb.altura_m, 2)} m` : "—" },
    { nombre: "Envergadura", a: fa.envergadura_m ? `${decimal(fa.envergadura_m, 2)} m` : "—", b: fb.envergadura_m ? `${decimal(fb.envergadura_m, 2)} m` : "—" },
    { nombre: "Guardia", a: fa.guardia ?? "—", b: fb.guardia ?? "—" },
    conRounds
      ? { nombre: "Golpes por minuto por round (promedio)", a: decimal(promedioRounds(pa), 1), b: decimal(promedioRounds(pb), 1) }
      : { nombre: "Golpes por minuto", a: decimal(fa.golpes_por_minuto, 1), b: decimal(fb.golpes_por_minuto, 1) },
    { nombre: "Jab sobre el total", a: fa.jab_fraccion == null ? "—" : <Estimado>{porcentajeAR(fa.jab_fraccion)}</Estimado>, b: fb.jab_fraccion == null ? "—" : <Estimado>{porcentajeAR(fb.jab_fraccion)}</Estimado> },
    banderas.guardia
      ? { nombre: "Guardia opuesta arriba", a: porcentajeAR(fa.guardia_opuesta_arriba), b: porcentajeAR(fb.guardia_opuesta_arriba) }
      : { nombre: "Guardia opuesta arriba", a: null, b: null, estado: "apagado" },
    conRounds
      ? { nombre: "Caída en el último round", a: porcentajeAR(fa.caida_ultimo_round), b: porcentajeAR(fb.caida_ultimo_round) }
      : { nombre: "Caída en el último round", a: null, b: null, estado: "apagado" },
  ];

  return (
    <>
      <EstadoBloque
        estado={estadoDe("07.dimensiones", producto)}
        titulo="Dimensiones reales"
        accion={
          <ToggleGroup type="single" value={superpuestas} onValueChange={(v) => v && onSuperpuestas(v)} aria-label="Cómo dibujar las siluetas" className="gap-1">
            <ToggleGroupItem value="lado" className="min-h-11 border border-borde px-3">Lado a lado</ToggleGroupItem>
            <ToggleGroupItem value="encima" className="min-h-11 border border-borde px-3">Superpuestas</ToggleGroupItem>
          </ToggleGroup>
        }
      >
        <Siluetas a={fa} b={fb} superpuestas={superpuestas === "encima"} />
      </EstadoBloque>

      <EstadoBloque estado={estadoDe("07.frente", producto)} titulo="Frente a frente" nota="Solo sesiones de sparring y combate de cada perfil.">
        <Desplazable nombre="Tabla frente a frente">
          <table className="w-full min-w-[420px] text-sm">
            <caption className="sr-only">Frente a frente</caption>
            <thead>
              <tr>
                <th scope="col" className="rotulo py-2 text-left font-normal">Métrica</th>
                <th scope="col" className="py-2 text-right font-normal"><span className="inline-flex items-center gap-1"><Esquina lado="A" tamano={16} />{fa.nombre}</span></th>
                <th scope="col" className="py-2 text-right font-normal"><span className="inline-flex items-center gap-1"><Esquina lado="B" tamano={16} />{fb.nombre}</span></th>
              </tr>
            </thead>
            <tbody>
              {filas.map((f) => (
                <tr key={f.nombre} className="border-t border-borde-suave">
                  <th scope="row" className="py-2 pr-3 text-left font-normal">{f.nombre}</th>
                  {f.estado === "apagado" ? (
                    <td colSpan={2} className="py-2 text-right"><Rotulo>en validación</Rotulo></td>
                  ) : (
                    <>
                      <td className="dato py-2 text-right">{f.a}</td>
                      <td className="dato py-2 text-right">{f.b}</td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </Desplazable>
      </EstadoBloque>

      <EstadoBloque estado={estadoDe("07.lecturas", producto)} titulo="Oportunidades y amenazas" motivo={textos.enConstruccion.lecturas}>
        {!hayLecturas(lec) ? (
          <p className="text-sm text-texto-2">{SIN_DIFERENCIAS}</p>
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {(["A", "B"] as const).map((l) => (
              <div key={l}>
                <p className="flex items-center gap-2 font-medium"><Esquina lado={l} />{l === "A" ? fa.nombre : fb.nombre}</p>
                <ul className="mt-2 space-y-2 text-sm">
                  {lec[l].oportunidades.map((x) => (
                    <li key={x.texto} className="rounded-sm border-l-2 border-ok-borde pl-2"><Rotulo className="text-ok">oportunidad · {x.metrica}</Rotulo><span className="block">{x.texto}</span></li>
                  ))}
                  {lec[l].amenazas.map((x) => (
                    <li key={x.texto} className="rounded-sm border-l-2 border-error-borde pl-2"><Rotulo className="text-error">amenaza · {x.metrica}</Rotulo><span className="block">{x.texto}</span></li>
                  ))}
                  {lec[l].oportunidades.length + lec[l].amenazas.length === 0 && <li className="text-texto-3">Sin lecturas para este lado.</li>}
                </ul>
              </div>
            ))}
          </div>
        )}
      </EstadoBloque>

      <EstadoBloque estado={estadoDe("07.cruces", producto)} titulo="Dónde ataca uno y dónde recibe el otro" motivo={textos.nodispExposicion} />
    </>
  );
}
