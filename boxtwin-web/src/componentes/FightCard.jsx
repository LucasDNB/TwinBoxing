/*
 * Paso 5 y 6: la Fight-Card, y tocar un evento para ver el golpe.
 *
 * Tres decisiones de esta pantalla, y ninguna es estetica:
 *
 * 1. EL MARGEN VA ARRIBA DE TODO Y NO EN UNA NOTA AL PIE. Con recall 0,484 el conteo esta
 *    por debajo del real, y un numero grande sin su margen se lee como un dato exacto. El
 *    texto sale del propio documento, no de una constante de la interfaz.
 *
 * 2. EL VIDEO ES LA FUNCION PRINCIPAL, no un extra. Lo que convierte un conteo dudoso en
 *    una herramienta de revision es que el entrenador pueda tocar un evento, ver el golpe y
 *    decidir el mismo. Por eso el reproductor queda pegado arriba mientras se recorre la
 *    linea de tiempo.
 *
 * 3. LO MEDIDO Y LO ESTIMADO SE VEN DISTINTO. El volumen y la guardia salen de medicion; el
 *    tipo de golpe sale de un clasificador que no generaliza a fuentes nuevas, y va con su
 *    confianza y con un rotulo. Mezclarlos en la misma tipografia seria mentir prolijo.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { api, urlExport } from '../api.js'
import Boxeadores from './Boxeadores.jsx'
import { nombreDeTipo, porcentaje, segundosATiempo, textoDeMargen } from '../formato.js'

const TIPOS = ['jab', 'cross', 'hook', 'uppercut']

export default function FightCard({ sesionId, fc, sesion, alCambiar, alVerPerfil }) {
  const video = useRef(null)
  const caja = useRef(null)
  const [ticket, setTicket] = useState(null)
  // El video procesado es la ultima etapa, asi que puede no estar cuando la Fight-Card ya
  // se muestra. Mientras no este se usa el original: es preferible un video sin marcas a un
  // hueco, porque la linea de tiempo ya sirve igual.
  const [hayProcesado, setHayProcesado] = useState(false)
  const [activo, setActivo] = useState(null)
  const [filtro, setFiltro] = useState('todos')
  const [corrigiendo, setCorrigiendo] = useState(null)
  // Desplegable: son decenas de eventos y ocupan mas que todo el resto junto. Arranca
  // cerrada y la eleccion se recuerda, porque quien la usa para revisar la abre una vez.
  const [abierta, setAbierta] = useState(() => {
    try {
      return localStorage.getItem('boxtwin.linea_abierta') === '1'
    } catch {
      return false
    }
  })

  const alternarLinea = () => {
    setAbierta((v) => {
      try {
        localStorage.setItem('boxtwin.linea_abierta', v ? '0' : '1')
      } catch {
        /* sin persistencia, dura lo que dure la pestana */
      }
      return !v
    })
  }
  const [error, setError] = useState(null)
  // El reproductor NO va fijo por default: pegado arriba se come media pantalla y tapa las
  // lecturas de abajo, que son las que el entrenador vino a leer. Queda a eleccion, y la
  // eleccion se recuerda.
  const [fijo, setFijo] = useState(() => {
    try {
      return localStorage.getItem('boxtwin.video_fijo') === '1'
    } catch {
      return false
    }
  })

  const alternarFijo = () => {
    setFijo((v) => {
      try {
        localStorage.setItem('boxtwin.video_fijo', v ? '0' : '1')
      } catch {
        /* sin persistencia, dura lo que dure la pestana */
      }
      return !v
    })
  }

  // Quien esta armando el video con marcas, si es que alguien lo esta armando.
  const render = (sesion?.trabajos || []).filter((t) => t.etapa === 'render').pop()
  const armando = render && ['en_cola', 'tomado'].includes(render.estado)
  const [pedido, setPedido] = useState(false)

  const armar = async () => {
    setError(null)
    try {
      await api.pedirRender(sesionId)
      setPedido(true)
    } catch (e) {
      setError(e.message)
    }
  }

  useEffect(() => {
    let vivo = true
    api
      .ticketVideo(sesionId)
      .then((r) => vivo && setTicket(r.ticket))
      .catch(() => vivo && setTicket(null))
    return () => {
      vivo = false
    }
  }, [sesionId])

  useEffect(() => {
    if (!ticket) return
    let vivo = true
    // HEAD y no GET: solo interesa si existe, y el archivo pesa.
    fetch(`/videos/${sesionId}/procesado?t=${encodeURIComponent(ticket)}`, { method: 'HEAD' })
      .then((r) => vivo && setHayProcesado(r.ok))
      .catch(() => vivo && setHayProcesado(false))
    return () => {
      vivo = false
    }
  }, [sesionId, ticket, fc])

  const eventos = useMemo(() => {
    const todos = []
    for (const p of ['A', 'B']) {
      for (const g of fc.peleadores[p]?.golpes || []) todos.push({ ...g, peleador: p })
    }
    return todos.sort((x, y) => x.t_inicio - y.t_inicio)
  }, [fc])

  const visibles = filtro === 'todos' ? eventos : eventos.filter((e) => e.peleador === filtro)

  const irA = (ev) => {
    setActivo(ev.id)
    const v = video.current
    if (!v) return
    // Medio segundo antes: el golpe empieza donde el anotador dice que arranca el
    // movimiento, que es antes de que se haga evidente. Caer justo ahi se siente tarde.
    v.currentTime = Math.max(0, ev.t_inicio - 0.5)
    v.play().catch(() => {})
    // Sin el reproductor fijo hay que traerlo a la vista, si no el salto no se ve.
    if (!fijo && caja.current) {
      const r = caja.current.getBoundingClientRect()
      if (r.bottom < 0 || r.top > window.innerHeight * 0.5) {
        caja.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
      }
    }
  }

  const corregir = async (ev, tipo) => {
    setError(null)
    try {
      await api.corregir(sesionId, ev.id, tipo)
      setCorrigiendo(null)
      alCambiar()
    } catch (err) {
      setError(err.message)
    }
  }

  const ident = fc.identidad || {}
  const sinAsignar = ident.sin_asignar ?? 0

  return (
    <div className="fightcard">
      <div className={`reproductor ${fijo ? 'fijo' : ''}`} ref={caja}>
        {ticket ? (
          <video
            ref={video}
            controls
            playsInline
            preload="metadata"
            src={
              hayProcesado
                ? `/videos/${sesionId}/procesado?t=${encodeURIComponent(ticket)}`
                : `/videos/${sesionId}/stream?t=${encodeURIComponent(ticket)}`
            }
          />
        ) : (
          <div className="sin_imagen alto">cargando el video…</div>
        )}
        <div className="barra_video">
          <button type="button" className="enlace chico" onClick={alternarFijo}>
            {fijo ? 'soltar el video' : 'fijar el video arriba'}
          </button>
          {hayProcesado && <span className="sutil">con los golpes marcados</span>}
        </div>

        {ticket && !hayProcesado ? (
          <p className="nota_video" role="status">
            {armando || pedido ? (
              <>
                El video con los golpes marcados se está armando. Mientras tanto se muestra
                el original, que ya sirve para saltar a cada evento.
              </>
            ) : render?.estado === 'fallo' ? (
              <>
                No se pudo armar el video con las marcas.{' '}
                <button type="button" className="enlace chico" onClick={armar}>
                  reintentar
                </button>
              </>
            ) : (
              <>
                Esta sesión no tiene el video con los golpes marcados.{' '}
                <button type="button" className="enlace chico" onClick={armar}>
                  armarlo ahora
                </button>
              </>
            )}
          </p>
        ) : null}
      </div>

      <Boxeadores sesionId={sesionId} alVerPerfil={alVerPerfil} />
      {(fc.avisos || []).map((a, i) => (
        <p className="aviso" key={i}>{a}</p>
      ))}

      <section className="tarjeta">
        <h2>Volumen detectado</h2>
        <table className="numeros">
          <thead>
            <tr><th></th><th>total</th><th>izq</th><th>der</th></tr>
          </thead>
          <tbody>
            {['A', 'B'].map((p) => {
              const t = fc.peleadores[p]?.total || { total: 0, izq: 0, der: 0 }
              return (
                <tr key={p}>
                  <th scope="row">Peleador {p}</th>
                  <td className="grande">{t.total}</td>
                  <td>{t.izq}</td>
                  <td>{t.der}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
        {/* RF6 y RF5 en una linea: el conteo no puede viajar sin su margen ni sin decir
            cuanto tiempo quedo sin mirar, pero eso cabe al pie de la tabla y no en un
            bloque de seis renglones arriba de todo. */}
        <p className="ayuda">{textoDeMargen(fc)}</p>
        <p className="ayuda">
          Identidad resuelta: A {porcentaje(ident.cobertura_A)}, B{' '}
          {porcentaje(ident.cobertura_B)}
          {sinAsignar > 0.005
            ? ` · ${porcentaje(sinAsignar)} del tiempo con alguien sin identificar`
            : ''}
          .
        </p>
      </section>

      {(fc.peleadores.A?.por_round || []).length > 0 && (
        <section className="tarjeta">
          <h2>Por round</h2>
          {['A', 'B'].map((p) => {
            const rondas = fc.peleadores[p]?.por_round || []
            const caida = fc.lectura?.caida_entre_rounds?.[p]
            const tope = Math.max(1, ...rondas.map((r) => r.total))
            // Con un solo round no hay nada que comparar y una barra sola al 100% no
            // informa: informa el numero. Las barras aparecen cuando hay dos o mas.
            if (rondas.length < 2) {
              const r = rondas[0]
              return (
                <div className="bloque_round" key={p}>
                  <h3>Peleador {p}</h3>
                  <p>
                    Round {r.round}: <strong>{r.total}</strong> golpes,{' '}
                    {r.por_minuto} por minuto.
                  </p>
                </div>
              )
            }
            return (
              <div className="bloque_round" key={p}>
                <h3>Peleador {p}</h3>
                <div className="barras">
                  {rondas.map((r) => (
                    <div className="barra_round" key={r.round}>
                      <div className="columna">
                        <div
                          className="relleno"
                          style={{ height: `${(r.total / tope) * 100}%` }}
                          title={`${r.total} golpes`}
                        />
                      </div>
                      <span className="etiqueta">R{r.round}</span>
                      <span className="valor">{r.total}</span>
                    </div>
                  ))}
                </div>
                {caida && (
                  <p className="ayuda">
                    {caida.variacion < 0 ? 'Bajó' : 'Subió'}{' '}
                    {Math.abs(caida.variacion).toFixed(1)} golpes por minuto entre el primer
                    round y el último ({caida.primer_round_por_minuto} →{' '}
                    {caida.ultimo_round_por_minuto}).
                  </p>
                )}
              </div>
            )
          })}
        </section>
      )}

      <section className="tarjeta">
        <div className="encabezado_linea">
          <h2>
            <button type="button" className="desplegable" onClick={alternarLinea}
                    aria-expanded={abierta}>
              <span className="flecha" aria-hidden="true">{abierta ? '▾' : '▸'}</span>
              Línea de tiempo
              <span className="sutil"> · {eventos.length}</span>
            </button>
          </h2>
          {abierta && (
            <div className="opciones">
              {['todos', 'A', 'B'].map((f) => (
                <button
                  key={f}
                  type="button"
                  className={`chip ${filtro === f ? 'activo' : ''}`}
                  onClick={() => setFiltro(f)}
                >
                  {f === 'todos' ? 'Los dos' : f}
                </button>
              ))}
            </div>
          )}
        </div>

        {error && <p className="error" role="alert">{error}</p>}
        {abierta && visibles.length === 0 && (
          <p className="sutil">No se detectó ningún golpe.</p>
        )}

        {abierta && (
        <ol className="eventos">
          {visibles.map((ev) => {
            const tipo = nombreDeTipo(ev.tipo, fc.nomenclatura)
            return (
              <li key={ev.id} className={activo === ev.id ? 'activo' : ''}>
                <button type="button" className="evento" onClick={() => irA(ev)}>
                  <span className="t">{segundosATiempo(ev.t_inicio)}</span>
                  <span className={`pel pel_${ev.peleador}`}>{ev.peleador}</span>
                  <span className="brazo">{ev.brazo}</span>
                  {tipo ? (
                    <span className={`tipo ${ev.corregido ? 'corregido' : 'estimado'}`}>
                      {tipo}
                      {ev.corregido ? (
                        <em> corregido</em>
                      ) : (
                        ev.confianza_tipo != null && <em> {porcentaje(ev.confianza_tipo)}</em>
                      )}
                    </span>
                  ) : (
                    <span className="tipo vacio">sin clasificar</span>
                  )}
                </button>
                <button
                  type="button"
                  className="enlace chico"
                  onClick={() => setCorrigiendo(corrigiendo === ev.id ? null : ev.id)}
                >
                  {corrigiendo === ev.id ? 'cancelar' : 'corregir'}
                </button>
                {corrigiendo === ev.id && (
                  <div className="correccion">
                    {TIPOS.map((t) => (
                      <button
                        key={t}
                        type="button"
                        className="chip"
                        onClick={() => corregir(ev, t)}
                      >
                        {nombreDeTipo(t, fc.nomenclatura)}
                      </button>
                    ))}
                  </div>
                )}
              </li>
            )
          })}
        </ol>
        )}
      </section>

      <section className="tarjeta">
        <h2>Exportar</h2>
        <div className="opciones">
          <a className="chip" href={urlExport(sesionId, 'csv')}>CSV</a>
          <a className="chip" href={urlExport(sesionId, 'pdf')} target="_blank" rel="noreferrer">
            Imprimible
          </a>
        </div>
        <p className="ayuda">
          El CSV trae una fila por golpe con su instante y su confianza. Para PDF, abrí el
          imprimible y elegí «Guardar como PDF».
        </p>
      </section>

      <section className="tarjeta tenue">
        <h2>Lo que esta Fight-Card no dice</h2>
        <ul>
          {(fc.no_incluye || []).map((x, i) => (
            <li key={i}>{x}</li>
          ))}
        </ul>
        <p className="ayuda">
          Detector {fc.detector?.checkpoint || '—'} · clasificador{' '}
          {fc.clasificador?.checkpoint || 'no corrido'} · contrato v{fc.version}
        </p>
      </section>
    </div>
  )
}
