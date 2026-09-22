/*
 * BoxTwin - La unica capa que habla con el servidor.
 *
 * POR QUE EXISTE
 *   Para que el token se ponga en un solo lugar y para que el manejo del 401 no este
 *   repetido en cada pantalla. Un componente que se olvida de mirar el 401 deja al usuario
 *   viendo una pantalla vacia sin saber que se le vencio la sesion.
 *
 *   Todas las rutas son relativas: en desarrollo las proxea vite y en produccion la API y
 *   el frontend salen por el mismo tunel. Asi no hay una URL de API que configurar ni que
 *   equivocar al desplegar.
 */

const CLAVE_TOKEN = 'boxtwin.token'
const CLAVE_USUARIO = 'boxtwin.usuario'

export function token() {
  try {
    return localStorage.getItem(CLAVE_TOKEN) || ''
  } catch {
    // Safari en modo privado tira al leer localStorage. Sin token la app pide login otra
    // vez, que es molesto pero no roto.
    return ''
  }
}

export function usuario() {
  try {
    return localStorage.getItem(CLAVE_USUARIO) || ''
  } catch {
    return ''
  }
}

export function guardarSesion(t, email) {
  try {
    localStorage.setItem(CLAVE_TOKEN, t)
    localStorage.setItem(CLAVE_USUARIO, email)
  } catch {
    /* sin persistencia, la sesion dura lo que dure la pestana */
  }
}

export function cerrarSesion() {
  try {
    localStorage.removeItem(CLAVE_TOKEN)
    localStorage.removeItem(CLAVE_USUARIO)
  } catch {
    /* nada que limpiar */
  }
}

export class ErrorApi extends Error {
  constructor(mensaje, estado) {
    super(mensaje)
    this.estado = estado
  }
}

async function pedir(ruta, opciones = {}) {
  const cabeceras = { ...(opciones.headers || {}) }
  const t = token()
  if (t) cabeceras.Authorization = `Bearer ${t}`
  if (opciones.body && !(opciones.body instanceof FormData)) {
    cabeceras['Content-Type'] = 'application/json'
  }

  const r = await fetch(ruta, { ...opciones, headers: cabeceras })
  if (r.status === 401) {
    cerrarSesion()
    throw new ErrorApi('se venció la sesión, entrá de nuevo', 401)
  }
  if (!r.ok) {
    let detalle = `error ${r.status}`
    try {
      const d = await r.json()
      detalle = typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail)
    } catch {
      /* el cuerpo no era json */
    }
    throw new ErrorApi(detalle, r.status)
  }
  if (r.status === 204) return null
  const tipo = r.headers.get('content-type') || ''
  return tipo.includes('json') ? r.json() : r.text()
}

export const api = {
  registro: (email, clave, invitacion = '') =>
    pedir('/auth/registro', {
      method: 'POST',
      body: JSON.stringify({ email, clave, invitacion }),
    }),

  login: (email, clave) =>
    pedir('/auth/login', { method: 'POST', body: JSON.stringify({ email, clave }) }),

  sesiones: () => pedir('/jobs'),

  estado: (id) => pedir(`/jobs/${id}`),

  sembrar: (id, track_a, track_b) =>
    pedir(`/jobs/${id}/siembra`, {
      method: 'POST',
      body: JSON.stringify({ track_a, track_b }),
    }),

  fightcard: (id) => pedir(`/fightcards/${id}`),

  corregir: (id, golpe, tipo) =>
    pedir(`/fightcards/${id}/golpes/${encodeURIComponent(golpe)}`, {
      method: 'PATCH',
      body: JSON.stringify({ tipo }),
    }),

  ticketVideo: (id) => pedir(`/videos/${id}/ticket`),

  pedirRender: (id) => pedir(`/jobs/${id}/render`, { method: 'POST' }),

  // La subida va con XMLHttpRequest y no con fetch por una sola razon: fetch no reporta
  // progreso de subida, y un video de celular puede tardar minutos. Una barra que no se
  // mueve durante tres minutos se lee como que se colgo.
  subir(archivo, { nombre, roundS, descansoS }, onProgreso) {
    return new Promise((resolver, rechazar) => {
      const datos = new FormData()
      datos.append('archivo', archivo)
      if (nombre) datos.append('nombre', nombre)
      if (roundS) datos.append('round_s', String(roundS))
      datos.append('descanso_s', String(descansoS ?? 60))

      const x = new XMLHttpRequest()
      x.open('POST', '/videos')
      const t = token()
      if (t) x.setRequestHeader('Authorization', `Bearer ${t}`)
      x.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgreso) onProgreso(e.loaded / e.total)
      }
      x.onload = () => {
        if (x.status >= 200 && x.status < 300) {
          resolver(JSON.parse(x.responseText))
        } else {
          let detalle = `error ${x.status}`
          try {
            detalle = JSON.parse(x.responseText).detail || detalle
          } catch {
            /* sin cuerpo util */
          }
          rechazar(new ErrorApi(detalle, x.status))
        }
      }
      x.onerror = () => rechazar(new ErrorApi('se cortó la conexión al subir', 0))
      x.send(datos)
    })
  },

  // Boxeadores y perfiles. El nombre es lo unico que hace que el peleador A de hoy y el de
  // la semana pasada sean la misma persona: A y B se asignan por posicion en pantalla.
  boxeadores: () => pedir('/boxeadores'),

  crearBoxeador: (nombre, guardia = null) =>
    pedir('/boxeadores', {
      method: 'POST',
      body: JSON.stringify({ nombre, guardia }),
    }),

  asignarBoxeadores: (id, boxeador_a, boxeador_b) =>
    pedir(`/sesiones/${id}/boxeadores`, {
      method: 'PUT',
      body: JSON.stringify({ boxeador_a, boxeador_b }),
    }),

  perfil: (id) => pedir(`/boxeadores/${id}/perfil`),
}

export const urlExport = (id, formato) => `/fightcards/${id}/export?formato=${formato}`
