/*
 * BoxTwin - El flujo entero, en un componente.
 *
 * POR QUE SIN ROUTER
 *   El flujo del producto son seis pasos y dos pantallas: la lista de sesiones y una
 *   sesion. Un router agregaria una dependencia y un concepto para manejar dos estados que
 *   ya estan en la URL con un hash. Cuando haya mas pantallas se agrega; hoy seria
 *   estructura sin contenido.
 *
 * EL SONDEO
 *   El estado se consulta cada pocos segundos y no por websocket. El procesamiento dura
 *   entre minutos y una hora, asi que la diferencia entre enterarse al instante y
 *   enterarse cuatro segundos despues no existe, y un websocket agrega reconexion,
 *   heartbeat y un modo de falla nuevo. El sondeo se detiene cuando la sesion llega a un
 *   estado final, que es lo que evita que una pestana abierta toda la noche golpee la API.
 */

import { useCallback, useEffect, useState } from 'react'
import { api, cerrarSesion, token, usuario as usuarioGuardado } from './api.js'
import Estado from './componentes/Estado.jsx'
import FightCard from './componentes/FightCard.jsx'
import Perfil from './componentes/Perfil.jsx'
import Ingreso from './componentes/Ingreso.jsx'
import Siembra from './componentes/Siembra.jsx'
import Subir from './componentes/Subir.jsx'
import { segundosATiempo } from './formato.js'

const FINALES = ['listo', 'fallo', 'espera_siembra']
const SONDEO_MS = 4000

function idDeLaUrl() {
  const h = window.location.hash.replace(/^#\/?/, '')
  return h.startsWith('sesion/') ? h.slice('sesion/'.length) : null
}

export default function App() {
  const [usuario, setUsuario] = useState(() => (token() ? usuarioGuardado() : ''))
  const [sesionId, setSesionId] = useState(idDeLaUrl)
  const [sesion, setSesion] = useState(null)
  const [fc, setFc] = useState(null)
  const [lista, setLista] = useState([])
  const [error, setError] = useState(null)
  // El perfil se abre encima de la sesion y no en otra pagina: se entra desde la
  // Fight-Card, se mira y se vuelve, sin perder donde estaba.
  const [perfilId, setPerfilId] = useState(null)

  useEffect(() => {
    const alCambiar = () => setSesionId(idDeLaUrl())
    window.addEventListener('hashchange', alCambiar)
    return () => window.removeEventListener('hashchange', alCambiar)
  }, [])

  const abrir = useCallback((id) => {
    window.location.hash = id ? `#/sesion/${id}` : '#/'
    setSesionId(id)
    setSesion(null)
    setFc(null)
  }, [])

  const salir = () => {
    cerrarSesion()
    setUsuario('')
    abrir(null)
  }

  // -- la lista
  const refrescarLista = useCallback(() => {
    if (!usuario) return
    api.sesiones().then(setLista).catch((e) => setError(e.message))
  }, [usuario])

  useEffect(() => {
    if (!sesionId) refrescarLista()
  }, [sesionId, refrescarLista])

  // -- una sesion, con sondeo mientras se procesa
  useEffect(() => {
    if (!usuario || !sesionId) return undefined
    let vivo = true
    let timer = null

    const consultar = async () => {
      try {
        const s = await api.estado(sesionId)
        if (!vivo) return
        setSesion(s)
        if (s.estado === 'listo') {
          try {
            setFc(await api.fightcard(sesionId))
          } catch {
            /* puede estar escribiendose todavia; el proximo sondeo la trae */
          }
        }
        if (!FINALES.includes(s.estado)) timer = setTimeout(consultar, SONDEO_MS)
      } catch (e) {
        if (!vivo) return
        setError(e.message)
        if (e.estado === 401) setUsuario('')
      }
    }
    consultar()
    return () => {
      vivo = false
      if (timer) clearTimeout(timer)
    }
  }, [usuario, sesionId])

  const recargarFightcard = useCallback(() => {
    api.fightcard(sesionId).then(setFc).catch((e) => setError(e.message))
  }, [sesionId])

  if (!usuario) return <Ingreso alEntrar={setUsuario} />

  return (
    <div className="app">
      <header>
        <button type="button" className="marca enlace" onClick={() => abrir(null)}>
          BoxTwin
        </button>
        <div className="derecha">
          <span className="sutil">{usuario}</span>
          <button type="button" className="enlace" onClick={salir}>salir</button>
        </div>
      </header>

      {error && (
        <p className="error" role="alert" onClick={() => setError(null)}>{error}</p>
      )}

      {perfilId ? (
        <main>
          <Perfil boxeadorId={perfilId} alVolver={() => setPerfilId(null)} />
        </main>
      ) : (
      <main>
        {!sesionId ? (
          <>
            <Subir alSubir={abrir} />
            <section className="tarjeta">
              <h2>Sesiones</h2>
              {lista.length === 0 && <p className="sutil">Todavía no subiste ninguna.</p>}
              <ul className="lista">
                {lista.map((s) => (
                  <li key={s.id}>
                    <button type="button" className="fila" onClick={() => abrir(s.id)}>
                      <strong>{s.nombre}</strong>
                      <span className={`estado ${s.estado}`}>{s.estado.replace('_', ' ')}</span>
                      <span className="sutil">
                        {s.duracion_s ? segundosATiempo(s.duracion_s) : ''}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          </>
        ) : !sesion ? (
          <p className="sutil">Cargando…</p>
        ) : (
          <>
            {sesion.estado !== 'listo' && <Estado sesion={sesion} />}
            {sesion.estado === 'espera_siembra' && (
              <Siembra
                sesion={sesion}
                alSembrar={() => setSesion({ ...sesion, estado: 'en_cola' })}
              />
            )}
            {sesion.estado === 'listo' && fc && (
              <FightCard
                sesionId={sesionId}
                fc={fc}
                alCambiar={recargarFightcard}
                alVerPerfil={setPerfilId}
              />
            )}
            {sesion.estado === 'listo' && !fc && <p className="sutil">Cargando la Fight-Card…</p>}
          </>
        )}
      </main>
      )}
    </div>
  )
}
