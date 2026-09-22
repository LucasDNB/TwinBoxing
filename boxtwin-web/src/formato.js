/*
 * BoxTwin - Como se escriben los numeros y los nombres en pantalla.
 *
 * POR QUE EXISTE
 *   Dos cosas que no pueden quedar en cada componente.
 *
 *   La primera es la nomenclatura: RF11 pide uso rioplatense y el mapeo viene adentro de la
 *   Fight-Card, del servidor. El frontend no inventa nombres de golpe, los lee. Si maniana
 *   Lucas decide que el hook se llama de otra forma, cambia en un lugar y no en cinco
 *   componentes.
 *
 *   La segunda es el margen. Cada vez que se muestra un conteo hay que poder decir de donde
 *   sale y cuanto vale, y eso tiene que ser una funcion y no una frase que alguien copia y
 *   pega distinto en cada pantalla.
 */

export function segundosATiempo(s) {
  if (s == null || Number.isNaN(s)) return '—'
  const total = Math.max(0, Math.floor(s))
  const m = Math.floor(total / 60)
  const seg = total % 60
  return `${m}:${String(seg).padStart(2, '0')}`
}

export function nombreDeTipo(tipo, nomenclatura) {
  if (!tipo) return null
  return (nomenclatura && nomenclatura[tipo]) || tipo
}

export function porcentaje(x, decimales = 0) {
  if (x == null || Number.isNaN(x)) return '—'
  return `${(x * 100).toFixed(decimales)}%`
}

/*
 * El texto del margen. Sale del propio documento y no de una constante del frontend: si el
 * detector cambia, el numero de la pantalla cambia con el, y no queda uno viejo escrito en
 * el codigo de la interfaz.
 */
export function textoDeMargen(fc) {
  const d = (fc && fc.detector) || {}
  return (
    `Golpes detectados, no golpes lanzados. El detector encuentra alrededor del ` +
    `${porcentaje(d.recall_medido)} de los golpes reales y acierta en el ` +
    `${porcentaje(d.precision_medida)} de lo que marca, medido sobre material que no vio. ` +
    `El conteo está por debajo del real: lo que sirve es comparar adentro de la sesión.`
  )
}
