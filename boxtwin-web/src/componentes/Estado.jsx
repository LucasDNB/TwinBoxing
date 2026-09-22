/*
 * Paso 3: la espera.
 *
 * Es la pantalla que mas tiempo esta a la vista -el procesamiento tarda cerca de una vez y
 * media la duracion del video- y la que decide si el usuario cree que el sistema funciona.
 * Por eso muestra en que etapa esta y cuanto llevan las que ya terminaron, en vez de una
 * rueda que gira. Una rueda que gira durante cuarenta minutos se lee como que se colgo.
 */

import { segundosATiempo } from '../formato.js'

const PASOS = [
  { estado: 'en_cola', titulo: 'En cola', detalle: 'esperando que se libere la placa' },
  { estado: 'procesando', titulo: 'Detectando cuerpos', detalle: 'pose y seguimiento, cuadro por cuadro' },
  { estado: 'espera_siembra', titulo: 'Falta que digas quién es quién', detalle: '' },
  { estado: 'completando', titulo: 'Buscando los golpes', detalle: 'identidad y detector' },
  { estado: 'listo', titulo: 'Listo', detalle: '' },
]

export default function Estado({ sesion }) {
  const i = PASOS.findIndex((p) => p.estado === sesion.estado)
  const fallo = sesion.estado === 'fallo'

  return (
    <div className="tarjeta">
      <h2>{sesion.nombre}</h2>
      <p className="sutil">
        {sesion.video}
        {sesion.duracion_s ? ` · ${segundosATiempo(sesion.duracion_s)}` : ''}
      </p>

      {fallo ? (
        <div className="error" role="alert">
          <strong>Algo falló procesando esta sesión.</strong>
          <pre>{sesion.error}</pre>
        </div>
      ) : (
        <ol className="pasos">
          {PASOS.map((p, k) => (
            <li
              key={p.estado}
              className={k < i ? 'hecho' : k === i ? 'actual' : 'pendiente'}
            >
              <span className="punto" aria-hidden="true" />
              <div>
                <strong>{p.titulo}</strong>
                {p.detalle && <span className="sutil"> — {p.detalle}</span>}
              </div>
            </li>
          ))}
        </ol>
      )}

      {(sesion.etapas || []).length > 0 && (
        <details>
          <summary className="sutil">Tiempos de máquina</summary>
          <table className="numeros chica">
            <tbody>
              {sesion.etapas.map((e, k) => (
                <tr key={k}>
                  <th scope="row">{e.etapa}</th>
                  <td>{e.segundos.toFixed(1)} s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}

      {(sesion.avisos || []).map((a, k) => (
        <p className="aviso" key={k}>{a}</p>
      ))}
    </div>
  )
}
