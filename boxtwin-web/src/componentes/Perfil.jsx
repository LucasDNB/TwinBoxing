/*
 * BoxTwin - El perfil de un boxeador: sus sesiones una al lado de la otra.
 *
 * POR QUE EXISTE
 *   Una sesion suelta contesta "cuantos golpes tiro hoy". La pregunta del entrenador es
 *   "esta mejorando", y esa necesita varias sesiones de la misma persona.
 *
 *   Cada numero va con lo que lo limita y en el mismo lugar, no al pie. El volumen es de
 *   golpes DETECTADOS y esta por debajo del real; la mezcla sale de un clasificador que no
 *   generaliza. Separar el numero de su reserva es la forma prolija de mentir.
 */

import { useEffect, useState } from 'react'
import { api } from '../api.js'

function Barra({ valor, maximo }) {
  const ancho = maximo > 0 ? Math.round((valor / maximo) * 100) : 0
  return (
    <div className="barra" aria-hidden="true">
      <div className="barra_relleno" style={{ width: `${ancho}%` }} />
    </div>
  )
}

export default function Perfil({ boxeadorId, alVolver }) {
  const [p, setP] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let vivo = true
    api
      .perfil(boxeadorId)
      .then((r) => vivo && setP(r))
      .catch((e) => vivo && setError(e.message))
    return () => {
      vivo = false
    }
  }, [boxeadorId])

  if (error) return <p className="error" role="alert">{error}</p>
  if (!p) return <p className="cargando">cargando el perfil…</p>

  const t = p.totales
  const maxGolpes = Math.max(1, ...p.evolucion.map((e) => e.golpes))
  const totalMezcla = Object.values(t.mezcla).reduce((s, n) => s + n, 0)

  return (
    <div className="perfil">
      <button type="button" className="enlace" onClick={alVolver}>← volver</button>

      <h2>{p.boxeador.nombre}</h2>
      {p.boxeador.guardia ? <p className="ayuda">guardia {p.boxeador.guardia}</p> : null}

      {t.sesiones === 0 ? (
        <p className="ayuda">
          Todavia no hay sesiones asignadas a este boxeador. Se asignan desde la Fight-Card
          de cada sesion, diciendo cual de los dos lados es el.
        </p>
      ) : (
        <>
          <section className="resumen">
            <div className="dato">
              <strong>{t.sesiones}</strong>
              <span>sesiones</span>
            </div>
            <div className="dato">
              <strong>{t.golpes}</strong>
              <span>golpes detectados</span>
            </div>
            {t.golpes_por_minuto_medio != null ? (
              <div className="dato">
                <strong>{t.golpes_por_minuto_medio}</strong>
                <span>golpes por minuto</span>
              </div>
            ) : null}
          </section>

          <section>
            <h3>Evolucion</h3>
            <ul className="evolucion">
              {p.evolucion.map((e) => (
                <li key={e.sesion_id}>
                  <span className="fecha">{e.fecha.slice(0, 10)}</span>
                  <Barra valor={e.golpes} maximo={maxGolpes} />
                  <span className="cifra">
                    {e.golpes}
                    {e.golpes_por_minuto != null ? ` · ${e.golpes_por_minuto}/min` : ''}
                  </span>
                </li>
              ))}
            </ul>
          </section>

          {totalMezcla > 0 ? (
            <section>
              <h3>Mezcla de golpes</h3>
              <ul className="mezcla">
                {Object.entries(t.mezcla)
                  .sort((x, y) => y[1] - x[1])
                  .map(([tipo, n]) => (
                    <li key={tipo}>
                      <span className="tipo estimado">{tipo}</span>
                      <Barra valor={n} maximo={totalMezcla} />
                      <span className="cifra">
                        {n} <em>{Math.round((n / totalMezcla) * 100)}%</em>
                      </span>
                    </li>
                  ))}
              </ul>
              {t.sin_clasificar > 0 ? (
                <p className="ayuda">
                  {t.sin_clasificar} golpes sin tipo estimado. No entran en este reparto, que
                  no es lo mismo que contarlos como cero.
                </p>
              ) : null}
            </section>
          ) : null}

          {Object.keys(t.brazos).length > 0 ? (
            <section>
              <h3>Lateralidad</h3>
              <ul className="mezcla">
                {Object.entries(t.brazos).map(([brazo, n]) => (
                  <li key={brazo}>
                    <span>{brazo}</span>
                    <Barra valor={n} maximo={t.golpes || 1} />
                    <span className="cifra">{n}</span>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}
        </>
      )}

      {p.avisos.map((a) => (
        <p className="margen" role="note" key={a}>{a}</p>
      ))}
    </div>
  )
}
