/*
 * Paso 1 del flujo: entrar. Cuenta simple, email y clave.
 */

import { useState } from 'react'
import { api, guardarSesion } from '../api.js'

export default function Ingreso({ alEntrar }) {
  const [modo, setModo] = useState('login')
  const [email, setEmail] = useState('')
  const [clave, setClave] = useState('')
  const [invitacion, setInvitacion] = useState('')
  const [error, setError] = useState(null)
  const [esperando, setEsperando] = useState(false)

  const enviar = async (e) => {
    e.preventDefault()
    setError(null)
    setEsperando(true)
    try {
      const r =
        modo === 'login'
          ? await api.login(email, clave)
          : await api.registro(email, clave, invitacion)
      guardarSesion(r.token, r.usuario)
      alEntrar(r.usuario)
    } catch (err) {
      setError(err.message)
    } finally {
      setEsperando(false)
    }
  }

  return (
    <div className="centrado">
      <form className="tarjeta angosta" onSubmit={enviar}>
        <h1 className="marca">BoxTwin</h1>
        <p className="sutil">
          Análisis táctico de sparring filmado. Subís el video, marcás quién es quién, y ves
          cuánto trabajó cada uno.
        </p>

        <label>
          Email
          <input
            type="email"
            autoComplete="email"
            inputMode="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </label>

        <label>
          Contraseña
          <input
            type="password"
            autoComplete={modo === 'login' ? 'current-password' : 'new-password'}
            required
            minLength={8}
            value={clave}
            onChange={(e) => setClave(e.target.value)}
          />
        </label>
        {modo === 'registro' && <p className="ayuda">Mínimo 8 caracteres.</p>}

        {modo === 'registro' && (
          <label>
            Código de invitación
            <input
              type="text"
              autoComplete="off"
              value={invitacion}
              onChange={(e) => setInvitacion(e.target.value)}
            />
          </label>
        )}

        {error && <p className="error" role="alert">{error}</p>}

        <button className="principal" disabled={esperando}>
          {esperando ? 'Un momento…' : modo === 'login' ? 'Entrar' : 'Crear cuenta'}
        </button>

        <button
          type="button"
          className="enlace"
          onClick={() => {
            setModo(modo === 'login' ? 'registro' : 'login')
            setError(null)
          }}
        >
          {modo === 'login' ? 'No tengo cuenta' : 'Ya tengo cuenta'}
        </button>
      </form>
    </div>
  )
}
