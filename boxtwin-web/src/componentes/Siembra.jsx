/*
 * Paso 4: la unica pregunta que el sistema le hace al usuario.
 *
 * "Cual de estos dos sos vos" parece trivial y es la diferencia medida entre 82,4% y 99,1%
 * de tracks bien asignados. Los dos recortes salen del MISMO cuadro del video, y eso esta
 * dicho en pantalla: es lo que le permite al que mira descartar que sean la misma persona
 * dos veces, que es el error que esta pantalla existe para evitar.
 *
 * Si el sistema no pudo proponer una pareja, se muestran mas candidatos y elige el usuario.
 */

import { useState } from 'react'
import { api } from '../api.js'

export default function Siembra({ sesion, alSembrar }) {
  const [a, setA] = useState(sesion.pareja_sugerida?.[0] ?? null)
  const [b, setB] = useState(sesion.pareja_sugerida?.[1] ?? null)
  const [error, setError] = useState(null)
  const [enviando, setEnviando] = useState(false)

  const candidatos = sesion.candidatos || []
  const haySugerencia = Boolean(sesion.pareja_sugerida)

  const elegir = (track) => {
    // Un toque pone A, el siguiente pone B, el tercero reemplaza A. Sin modo ni
    // instrucciones: en el gimnasio nadie lee.
    if (a === track) return setA(null)
    if (b === track) return setB(null)
    if (a == null) return setA(track)
    if (b == null) return setB(track)
    setA(track)
  }

  const enviar = async () => {
    setError(null)
    setEnviando(true)
    try {
      await api.sembrar(sesion.id, a, b)
      alSembrar()
    } catch (err) {
      setError(err.message)
      setEnviando(false)
    }
  }

  return (
    <div className="tarjeta">
      <h2>¿Quién es quién?</h2>
      <p className="sutil">
        {haySugerencia
          ? 'Los dos recortes salen del mismo instante del video, así que son dos personas distintas. Marcá cuál es cada uno.'
          : 'El sistema no pudo separar dos peleadores con seguridad. Elegí los dos entre estos candidatos.'}
      </p>

      <div className="candidatos">
        {candidatos.map((c) => {
          const rol = c.track === a ? 'A' : c.track === b ? 'B' : null
          return (
            <button
              type="button"
              key={c.track}
              className={`candidato ${rol ? 'elegido' : ''}`}
              onClick={() => elegir(c.track)}
              aria-pressed={Boolean(rol)}
            >
              {c.recorte_url ? (
                <img src={c.recorte_url} alt={`Candidato del track ${c.track}`} loading="lazy" />
              ) : (
                <div className="sin_imagen">sin recorte</div>
              )}
              {rol && <span className="rol">{rol}</span>}
              <span className="meta">
                track {c.track} · guante {Math.round((c.fraccion_guante ?? 0) * 100)}%
              </span>
            </button>
          )
        })}
      </div>

      {a != null && b != null && (
        <p className="ayuda">
          A es el track {a}, B es el track {b}. Si salieron al revés, se puede volver a
          elegir después sin subir el video de nuevo.
        </p>
      )}

      {error && <p className="error" role="alert">{error}</p>}

      <button
        className="principal"
        disabled={a == null || b == null || a === b || enviando}
        onClick={enviar}
      >
        {enviando ? 'Mandando…' : 'Confirmar y seguir'}
      </button>
    </div>
  )
}
