/*
 * Lo poco del frontend que tiene logica propia.
 *
 * El test que importa es el de nomenclatura: el frontend NO puede tener nombres de golpe
 * escritos adentro. Los lee del documento, porque el mapeo rioplatense es un criterio de
 * dominio que decide Lucas y tiene que poder cambiar en un solo lugar.
 */

import { describe, expect, it } from 'vitest'
import { nombreDeTipo, porcentaje, segundosATiempo, textoDeMargen } from './formato.js'

describe('segundosATiempo', () => {
  it('escribe minutos y segundos', () => {
    expect(segundosATiempo(0)).toBe('0:00')
    expect(segundosATiempo(75.4)).toBe('1:15')
    expect(segundosATiempo(3600)).toBe('60:00')
  })

  it('no inventa un tiempo cuando no hay dato', () => {
    expect(segundosATiempo(null)).toBe('—')
    expect(segundosATiempo(undefined)).toBe('—')
  })
})

describe('nombreDeTipo', () => {
  it('usa la nomenclatura del documento y no una propia', () => {
    const nom = { jab: 'jab', cross: 'gancho', hook: 'hook', uppercut: 'uppercut' }
    expect(nombreDeTipo('cross', nom)).toBe('gancho')
  })

  it('si el documento no trae el nombre, no traduce', () => {
    expect(nombreDeTipo('cross', {})).toBe('cross')
    expect(nombreDeTipo('cross', undefined)).toBe('cross')
  })

  it('sin tipo no muestra nada, que no es lo mismo que un tipo vacio', () => {
    expect(nombreDeTipo(null, {})).toBe(null)
  })
})

describe('textoDeMargen', () => {
  const fc = { detector: { precision_medida: 0.885, recall_medido: 0.484 } }

  it('sale del documento y no de una constante de la interfaz', () => {
    // Si el detector mejora, el numero de la pantalla mejora con el. Una constante en el
    // frontend quedaria vieja y nadie la miraria.
    const t = textoDeMargen(fc)
    expect(t).toContain('48%')
    expect(t).toContain('89%')
  })

  it('dice detectados y no lanzados', () => {
    expect(textoDeMargen(fc)).toContain('no golpes lanzados')
  })

  it('dice que el conteo esta por debajo del real', () => {
    expect(textoDeMargen(fc)).toContain('por debajo del real')
  })
})

describe('porcentaje', () => {
  it('redondea sin decimales por defecto', () => {
    expect(porcentaje(0.484)).toBe('48%')
  })

  it('no inventa un cero cuando no hay dato', () => {
    expect(porcentaje(null)).toBe('—')
  })
})
