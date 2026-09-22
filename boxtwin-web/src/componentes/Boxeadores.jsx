/*
 * BoxTwin - Asignar a quien corresponde cada lado de una sesion.
 *
 * POR QUE EXISTE
 *   Los nombres A y B se asignan por posicion en pantalla y no significan nada entre
 *   videos: el peleador A de hoy no es el A de la semana pasada. Sin alguien que lo diga,
 *   el sistema no puede juntar las sesiones de una misma persona, y sin eso no hay perfil
 *   ni evolucion, que es la pregunta que el entrenador tiene.
 *
 *   Es opcional y se puede dejar a medias. Sparring contra alguien que no esta cargado es
 *   el caso normal, y obligar a nombrar a los dos convertiria una anotacion util en un
 *   tramite antes de ver el resultado.
 */

import { useEffect, useState } from 'react'
import { api } from '../api.js'

export default function Boxeadores({ sesionId, asignados, alAsignar, alVerPerfil }) {
  const [lista, setLista] = useState([])
  const [a, setA] = useState(asignados?.boxeador_a || '')
  const [b, setB] = useState(asignados?.boxeador_b || '')
  const [nuevo, setNuevo] = useState('')
  const [error, setError] = useState(null)
  const [guardando, setGuardando] = useState(false)

  const recargar = () =>
    api
      .boxeadores()
      .then(setLista)
      .catch((e) => setError(e.message))

  useEffect(() => {
    recargar()
  }, [])

  const crear = async () => {
    const nombre = nuevo.trim()
    if (!nombre) return
    setError(null)
    try {
      const creado = await api.crearBoxeador(nombre)
      setNuevo('')
      await recargar()
      // Se deja elegido el que acaba de crear, que es lo que estaba por hacer.
      if (!a) setA(creado.id)
      else if (!b) setB(creado.id)
    } catch (e) {
      setError(e.message)
    }
  }

  const guardar = async () => {
    setError(null)
    setGuardando(true)
    try {
      await api.asignarBoxeadores(sesionId, a || null, b || null)
      alAsignar?.({ boxeador_a: a || null, boxeador_b: b || null })
    } catch (e) {
      setError(e.message)
    } finally {
      setGuardando(false)
    }
  }

  const opciones = (excluir) => (
    <>
      <option value="">sin asignar</option>
      {lista
        .filter((x) => x.id !== excluir)
        .map((x) => (
          <option key={x.id} value={x.id}>
            {x.nombre}
          </option>
        ))}
    </>
  )

  return (
    <section className="boxeadores">
      <h3>Quien es quien</h3>
      <p className="ayuda">
        A y B salen de la posicion en pantalla, asi que no significan lo mismo en otro video.
        Poniendo el nombre, las sesiones de una misma persona se juntan en un perfil.
      </p>

      <div className="asignacion">
        <label>
          <span className="lado lado_a">Peleador A</span>
          <select value={a} onChange={(e) => setA(e.target.value)}>{opciones(b)}</select>
          {a ? (
            <button type="button" className="enlace" onClick={() => alVerPerfil?.(a)}>
              ver perfil
            </button>
          ) : null}
        </label>

        <label>
          <span className="lado lado_b">Peleador B</span>
          <select value={b} onChange={(e) => setB(e.target.value)}>{opciones(a)}</select>
          {b ? (
            <button type="button" className="enlace" onClick={() => alVerPerfil?.(b)}>
              ver perfil
            </button>
          ) : null}
        </label>
      </div>

      <div className="alta">
        <input
          value={nuevo}
          placeholder="nombre de un boxeador nuevo"
          onChange={(e) => setNuevo(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && crear()}
        />
        <button type="button" onClick={crear} disabled={!nuevo.trim()}>
          agregar
        </button>
        <button type="button" className="primario" onClick={guardar} disabled={guardando}>
          {guardando ? 'guardando…' : 'guardar'}
        </button>
      </div>

      {error ? <p className="error" role="alert">{error}</p> : null}
    </section>
  )
}
