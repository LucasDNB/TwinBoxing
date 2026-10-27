"use client";

/*
 * Los bloques del perfil que agregan los cambios del 14-10 (specs/cambios-14-10/01 §4):
 *
 *   W20  distribucion de golpes por mano y altura
 *   W21  volumen por round, con segmentos de 3 minutos y el rotulo de C2
 *   W22  mapa de exposicion: lo que le dirigen sus rivales, no lo que le entro (DC-3)
 *   W23  avatar 3D ilustrativo (DC-6)
 *   W24  distancia de trabajo y estilo, cargados a mano (DC-7)
 *
 * Los numeros salen todos de la API (B15 a B17); aca solo se dibujan con su estado.
 */
import Link from "next/link";
import { useState } from "react";
import { AJUSTES_CALOR, Avatar3D } from "@/components/boxtwin/avatar/Avatar3D";
import { LeyendaCalor } from "@/components/boxtwin/avatar/LeyendaCalor";
import { EstadoBloque, Estimado } from "@/components/boxtwin/EstadoBloque";
import { Rotulo } from "@/components/boxtwin/Rotulo";
import { NOMBRE_ZONA, SiluetaZonas } from "@/components/boxtwin/graficos/SiluetaZonas";
import { TIPOS_GOLPE, ZONAS, type Perfil, type Producto, type Zona } from "@/lib/api/esquemas";
import { estadoDe, visible } from "@/lib/banderas";
import { intensidades } from "@/lib/avatar/exposicion";
import { decimal, entero, nombreDeTipo, porcentajeAR } from "@/lib/formato";
import { medidasDe } from "@/lib/maniqui";
import { textos } from "@/lib/textos";
import { cn } from "@/lib/utils";

export const NOMBRE_DISTANCIA: Record<string, string> = { larga: "Larga", media: "Media", corta: "Corta" };
export const NOMBRE_ESTILO: Record<string, string> = { fajador: "Fajador", estilista: "Estilista" };

/* Con menos golpes dirigidos el mapa va sin colorear (ampliacion-13-10/04 §4). */
const MINIMO_MAPA = 50;

// -- W20 ---------------------------------------------------------------------------------

export function DistribucionManoAltura({ p, producto }: { p: Perfil; producto?: Producto }) {
  const a = p.ataque;
  const conAltura = visible("06.altura", producto) && !!a?.zonas;
  const sesgo = a?.sesgo_detector;
  const altura = (z: Zona) => a?.zonas?.[z]?.n ?? 0;
  const conDato = a?.con_dato ?? 0;
  const pct = (n: number, de: number) => (de ? porcentajeAR(n / de) : "—");
  const manos = a?.manos ?? { izq: 0, der: 0 };
  const totalManos = manos.izq + manos.der;

  return (
    <EstadoBloque
      estado={estadoDe("06.distribucion", producto)}
      titulo="Distribución de golpes"
      nota={conAltura ? `${textos.alturaMetodo} ${sesgo ? textos.sesgoAltura(sesgo.recall_cuerpo, sesgo.recall_cabeza) : ""}` : undefined}
    >
      <div className="flex flex-col gap-4">
        {/* Por mano: medido, sale del detector. */}
        <div data-fila="manos">
          <p className="mb-1 text-sm">Por mano</p>
          <div className="grid grid-cols-2 gap-3">
            {(["izq", "der"] as const).map((m) => (
              <div key={m} className="flex items-baseline justify-between rounded-sm bg-superficie-2 px-3 py-2">
                <span className="text-sm text-texto-2">{m === "izq" ? "Izquierda" : "Derecha"}</span>
                <span className="dato">{entero(manos[m])} · {pct(manos[m], totalManos)}</span>
              </div>
            ))}
          </div>
        </div>

        {conAltura && (
          <table className="w-full text-sm" aria-label="Golpes por mano y altura">
            <thead>
              <tr>
                <th scope="col" className="rotulo py-1 text-left font-normal"><span className="sr-only">Altura</span></th>
                <th scope="col" className="rotulo py-1 text-right font-normal">Izquierda</th>
                <th scope="col" className="rotulo py-1 text-right font-normal">Derecha</th>
                <th scope="col" className="rotulo py-1 text-right font-normal">Total</th>
              </tr>
            </thead>
            <tbody>
              {(["cabeza", "cuerpo"] as const).map((h) => {
                const izq = altura(`${h}_mano_izq` as Zona);
                const der = altura(`${h}_mano_der` as Zona);
                return (
                  <tr key={h} className="border-t border-borde-suave">
                    <th scope="row" className="py-2 text-left font-normal">{h === "cabeza" ? "Cabeza" : "Cuerpo"}</th>
                    <td className="py-2 text-right"><Estimado className="dato">{entero(izq)} · {pct(izq, conDato)}</Estimado></td>
                    <td className="py-2 text-right"><Estimado className="dato">{entero(der)} · {pct(der, conDato)}</Estimado></td>
                    <td className="py-2 text-right"><Estimado className="dato">{entero(izq + der)} · {pct(izq + der, conDato)}</Estimado></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        {conAltura && (a?.sin_dato ?? 0) > 0 && (
          <p className="text-xs text-texto-3">{textos.alturaSinDato(String(a!.sin_dato))}</p>
        )}

        <MezclaPorTipo p={p} />
      </div>
    </EstadoBloque>
  );
}

/* La mezcla por tipo que mostraba el bloque antes del 14-10, ahora debajo y mas chica. */
function MezclaPorTipo({ p }: { p: Perfil }) {
  const mezcla = p.totales.mezcla;
  const clasificados = TIPOS_GOLPE.reduce((a, t) => a + (mezcla[t] ?? 0), 0);
  const clasificador = p.sesiones.find((s) => s.clasificador)?.clasificador ?? null;
  const exactitud = p.sesiones.find((s) => s.exactitud_familia != null)?.exactitud_familia ?? null;
  return (
    <div className="rounded-sm border border-dashed border-estimado-borde p-3">
      <Rotulo className="text-estimado">estimación · tipo de golpe</Rotulo>
      {clasificados === 0 ? (
        <p className="mt-1 text-sm text-texto-3">Ningún golpe tiene tipo estimado todavía.</p>
      ) : (
        <ul className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1">
          {TIPOS_GOLPE.map((t) => (
            <li key={t} className="flex items-baseline justify-between text-sm">
              <span>{nombreDeTipo(t, p.nomenclatura)}</span>
              <Estimado className="dato">{porcentajeAR((mezcla[t] ?? 0) / clasificados)}</Estimado>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-2 text-xs text-texto-3">{textos.estimacionTipo(clasificador, exactitud)} {p.totales.golpes} detectados, {p.totales.sin_clasificar} sin clasificar.</p>
    </div>
  );
}

// -- W21 ---------------------------------------------------------------------------------

export function VolumenPorRound({ p, producto }: { p: Perfil; producto?: Producto }) {
  const rondas = p.rounds_promedio ?? [];
  const max = Math.max(1, ...rondas.map((r) => r.golpes_por_minuto));
  const c2Aprobado = producto?.criterios.C2.estado === "aprobado";
  const origenes = new Set(rondas.map((r) => r.origen ?? "declarados"));
  const prefijo = (o: string | undefined) => (o === "segmentos_180s" ? "S" : o === "mixto" ? "R/S" : "R");
  return (
    <EstadoBloque
      estado={estadoDe("06.volumen_round", producto)}
      titulo="Volumen por round promedio"
      motivo={textos.condicionadoC2}
      accion={!c2Aprobado ? <Rotulo className="text-estimado">{textos.roundsSinValidar}</Rotulo> : undefined}
    >
      {rondas.length === 0 ? (
        <p className="text-sm text-texto-3">Sin sesiones con video suficiente para cortar en rounds.</p>
      ) : (
        <>
          <ul className="flex flex-col gap-2">
            {rondas.map((r) => (
              <li key={r.round} className="grid grid-cols-[2.5rem_1fr_3.5rem] items-center gap-2 text-sm">
                <span className="dato">{prefijo(r.origen)}{r.round}</span>
                <span aria-hidden="true" className="h-2 rounded-full bg-superficie-2"><span className="block h-full rounded-full bg-texto-2" style={{ width: `${(r.golpes_por_minuto / max) * 100}%` }} /></span>
                <span className="dato text-right">{decimal(r.golpes_por_minuto, 1)}/min</span>
              </li>
            ))}
          </ul>
          {origenes.has("segmentos_180s") && !origenes.has("mixto") && !origenes.has("declarados") && (
            <p className="mt-2 text-xs text-texto-3">{textos.roundsSegmentos}</p>
          )}
          {(origenes.has("mixto") || (origenes.has("segmentos_180s") && origenes.has("declarados"))) && (
            <p className="mt-2 text-xs text-texto-3">{textos.roundsMixto}</p>
          )}
        </>
      )}
    </EstadoBloque>
  );
}

// -- W22 ---------------------------------------------------------------------------------

export function MapaExposicion({ p, producto, contexto, vos = false }: { p: Perfil; producto?: Producto; contexto: string; vos?: boolean }) {
  const e = p.exposicion;
  const estado = estadoDe("06.exposicion", producto);
  if (estado !== "estimacion") {
    return <EstadoBloque estado={estado} titulo="Mapa de exposición" motivo={textos.nodispExposicion} />;
  }
  const sinRival = contexto === "bolsa" || contexto === "sombra";
  return (
    <EstadoBloque
      estado="estimacion"
      titulo={vos ? "Lo que te tiraron tus compañeros" : "Hacia dónde le tiran"}
      nota={e?.sesgo_detector.corregido ? textos.exposicionCorregida(e.sesgo_detector.recall_cuerpo, e.sesgo_detector.recall_cabeza) : undefined}
    >
      <p className="mb-3 text-sm text-texto-2">{textos.exposicionSubtitulo}</p>
      {sinRival || !e ? (
        <p className="text-sm text-texto-3">{sinRival ? textos.exposicionSinRival : "Todavía no hay sesiones de sparring o combate."}</p>
      ) : (
        <ContenidoMapa e={e} vos={vos} />
      )}
    </EstadoBloque>
  );
}

function ContenidoMapa({ e, vos = false }: { e: NonNullable<Perfil["exposicion"]>; vos?: boolean }) {
  const conDato = e.con_dato ?? 0;
  const colorear = conDato >= MINIMO_MAPA;
  const fr = Object.fromEntries(ZONAS.map((z) => [z, e.zonas?.[z]?.fraccion_corregida ?? null])) as Record<Zona, number | null>;
  const dominante = e.total > 0 ? e.por_rival.find((r) => r.total / e.total > 0.7) : undefined;
  return (
    <div className="grid gap-4 md:grid-cols-[auto_1fr] md:items-start">
      <SiluetaZonas
        fracciones={fr}
        colorear={colorear}
        etiqueta={`Silueta con las cuatro zonas: ${ZONAS.map((z) => `${NOMBRE_ZONA[z]} ${fr[z] == null ? "sin dato" : porcentajeAR(fr[z])}`).join(", ")}`}
      />
      <div className="flex flex-col gap-3">
        {!colorear && <p className="text-sm text-estimado">{textos.exposicionPreliminar(conDato)}</p>}
        <table className="w-full text-sm">
          <caption className="sr-only">Golpes de sus rivales por zona</caption>
          <thead>
            <tr>
              <th scope="col" className="rotulo py-1 text-left font-normal">Zona</th>
              <th scope="col" className="rotulo py-1 text-right font-normal">Detectados</th>
              <th scope="col" className="rotulo py-1 text-right font-normal">Fracción</th>
            </tr>
          </thead>
          <tbody>
            {ZONAS.map((z) => (
              <tr key={z} className="border-t border-borde-suave">
                <th scope="row" className="py-1.5 text-left font-normal">{NOMBRE_ZONA[z]}</th>
                <td className="dato py-1.5 text-right">{entero(e.zonas?.[z]?.n ?? 0)}</td>
                <td className="py-1.5 text-right"><Estimado className="dato">{fr[z] == null ? "—" : porcentajeAR(fr[z])}</Estimado></td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-xs text-texto-3">
          {entero(e.total)} golpes de sus rivales en {e.sesiones} {e.sesiones === 1 ? "sesión" : "sesiones"}
          {(e.sin_dato ?? 0) > 0 && `; ${e.sin_dato} sin altura`}.
        </p>
        {e.por_rival.length > 0 && (
          <div>
            <p className="text-sm">Rivales que aportan</p>
            <ul className="mt-1 text-sm text-texto-2">
              {e.por_rival.map((r) => (
                <li key={r.boxeador_id ?? "sin"} className="flex justify-between"><span>{vos && !r.boxeador_id ? "sin perfil" : r.nombre}</span><span className="dato">{r.total}</span></li>
              ))}
            </ul>
            {dominante && <p className="mt-2 text-sm text-estimado">{textos.exposicionDominante(dominante.nombre)}</p>}
          </div>
        )}
      </div>
    </div>
  );
}

// -- W23 ---------------------------------------------------------------------------------

export function AvatarPerfil({ p, producto }: { p: Perfil; producto?: Producto }) {
  const b = p.boxeador;
  const m = medidasDe(b.altura_m, b.envergadura_m);
  const guardia = b.guardia === "zurda" || b.guardia === "ortodoxa" ? b.guardia : null;
  const medidas = m.aEscala
    ? `${decimal(m.altura, 2)} m de altura y ${decimal(m.envergadura, 2)} m de envergadura`
    : "proporciones estándar";
  // El calor es el mismo dato que el mapa de exposicion (W22): si ese bloque no se ve, no se pinta.
  const zonas = visible("06.exposicion", producto) ? p.exposicion?.zonas : null;
  const calor = intensidades(zonas);
  const [ajustes, setAjustes] = useState(AJUSTES_CALOR);
  return (
    <EstadoBloque
      estado={estadoDe("06.avatar", producto)}
      titulo="Avatar 3D"
      accion={<Rotulo className="text-estimado">{textos.avatarIlustrativo}</Rotulo>}
    >
      <Avatar3D
        figuras={[{ medidas: m, guardia, lado: "A", extension: { adelantada: 0.15, atrasada: 0 }, x: 0, mira: 1, calor }]}
        etiqueta={`Avatar en guardia ${guardia ?? "sin guardia cargada"}, ${medidas}${calor ? ", con el mapa de golpes dirigidos sobre el cuerpo" : ""}`}
        alternativa={`En guardia ${guardia ?? "(sin guardia cargada)"}, ${medidas}.`}
        alto={340}
        ajustes={ajustes}
      />
      {calor && zonas && (
        <div className="mt-3">
          <LeyendaCalor
            id="calor-perfil"
            ajustes={ajustes}
            onCambio={setAjustes}
            fracciones={Object.fromEntries(ZONAS.map((z) => [z, zonas[z]?.fraccion_corregida ?? null])) as Record<Zona, number | null>}
          />
        </div>
      )}
      {!m.aEscala && <p className="mt-2 text-xs text-texto-3">{textos.avatarSinMedidas}</p>}
      <p className="mt-1 text-xs text-texto-3">Arrastrá para girarlo.</p>
    </EstadoBloque>
  );
}

// -- W24 ---------------------------------------------------------------------------------

export function DistanciaEstilo({ p, producto }: { p: Perfil; producto?: Producto }) {
  const b = p.boxeador;
  const cargado = b.distancia || b.estilo;
  return (
    <EstadoBloque
      estado={estadoDe("06.distancia", producto)}
      titulo="Distancia de trabajo"
      accion={cargado ? <Rotulo>{textos.distanciaManual}</Rotulo> : undefined}
    >
      {!cargado ? (
        <p className="text-sm text-texto-3">
          {textos.distanciaSinCargar}{" "}
          <Link href="/peleadores/" className="underline underline-offset-4">Cargarlos en Peleadores</Link>
        </p>
      ) : (
        <dl className="grid grid-cols-2 gap-3">
          <div>
            <dt className="rotulo">Distancia</dt>
            <dd className={cn("font-display text-2xl font-semibold", !b.distancia && "text-texto-3")}>{b.distancia ? NOMBRE_DISTANCIA[b.distancia] : "—"}</dd>
          </div>
          <div>
            <dt className="rotulo">Estilo</dt>
            <dd className={cn("font-display text-2xl font-semibold", !b.estilo && "text-texto-3")}>{b.estilo ? NOMBRE_ESTILO[b.estilo] : "—"}</dd>
          </div>
        </dl>
      )}
    </EstadoBloque>
  );
}
