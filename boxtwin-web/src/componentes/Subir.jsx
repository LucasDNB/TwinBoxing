/*
 * Paso 2: subir el video y decir cuanto dura el round.
 *
 * El campo de round no es un detalle de configuracion: sin el, la Fight-Card se agrega por
 * sesion y se pierde la comparacion entre rounds, que es una de las dos lecturas que el
 * recall medido sostiene. Por eso esta arriba y no escondido en un desplegable.
 *
 * `capture` en el input abre la camara directo en el celular, que es el caso base del
 * producto: el entrenador filma y sube ahi mismo.
 */

import { useState } from 'react'
import { api } from '../api.js'

const PRESETS = [
  { etiqueta: '3 min', s: 180 },
  { etiqueta: '2 min', s: 120 },
  { etiqueta: '1 min', s: 60 },
  { etiqueta: 'Sin rounds', s: null },
]

export default function Subir({ alSubir }) {
  const [archivo, setArchivo] = useState(null)
  const [nombre, setNombre] = useState('')
  const [roundS, setRoundS] = useState(180)
  const [descansoS, setDescansoS] = useState(60)
  const [progreso, setProgreso] = useState(null)
  const [error, setError] = useState(null)

  const enviar = async (e) => {
    e.preventDefault()
    if (!archivo) return
    setError(null)
    setProgreso(0)
    try {
      const r = await api.subir(archivo, { nombre, roundS, descansoS }, setProgreso)
      alSubir(r.job_id)
    } catch (err) {
      setError(err.message)
      setProgreso(null)
    }
  }

  return (
    <form className="tarjeta" onSubmit={enviar}>
      <h2>Nueva sesión</h2>

      <label>
        Video del sparring
        <input
          type="file"
          accept="video/mp4,video/quicktime,.mp4,.mov"
          capture="environment"
          required
          onChange={(e) => {
            const f = e.target.files?.[0] || null
            setArchivo(f)
            if (f && !nombre) setNombre(f.name.replace(/\.[^.]+$/, ''))
          }}
        />
      </label>
      {archivo && (
        <p className="ayuda">
          {archivo.name} · {(archivo.size / 1024 ** 3).toFixed(2)} GB
        </p>
      )}

      <label>
        Nombre de la sesión
        <input
          type="text"
          placeholder="sparring del martes"
          value={nombre}
          onChange={(e) => setNombre(e.target.value)}
        />
      </label>

      <fieldset>
        <legend>Duración del round</legend>
        <div className="opciones">
          {PRESETS.map((p) => (
            <button
              type="button"
              key={p.etiqueta}
              className={`chip ${roundS === p.s ? 'activo' : ''}`}
              onClick={() => setRoundS(p.s)}
            >
              {p.etiqueta}
            </button>
          ))}
        </div>
        {roundS != null && (
          <label className="en_linea">
            Descanso (s)
            <input
              type="number"
              min="0"
              max="600"
              value={descansoS}
              onChange={(e) => setDescansoS(Number(e.target.value))}
            />
          </label>
        )}
        <p className="ayuda">
          Sin rounds declarados el resultado se agrega por sesión entera. El gong no se
          detecta: no se ve en el video.
        </p>
      </fieldset>

      {error && <p className="error" role="alert">{error}</p>}

      {progreso != null ? (
        <div className="progreso" role="status">
          <div className="barra" style={{ width: `${Math.round(progreso * 100)}%` }} />
          <span>{Math.round(progreso * 100)}% subido</span>
        </div>
      ) : (
        <button className="principal" disabled={!archivo}>
          Subir y procesar
        </button>
      )}
    </form>
  )
}
