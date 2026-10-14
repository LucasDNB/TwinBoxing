/*
 * Que estado tiene cada bloque del diseno (01 §4), segun /producto.
 *
 * POR QUE EXISTE
 *   La web nunca decide sola si un bloque condicionado se muestra: lo dice el servidor. Esta
 *   tabla es la de 01 §4 escrita una vez; cada pantalla pregunta `estadoDe(bloque, producto)`
 *   y dibuja lo que corresponde con <EstadoBloque>.
 */
import type { Producto } from "@/lib/api/esquemas";

// "en_construccion" (13-10, decision de Lucas): la funcion se ve en el prototipo pero todavia
// no se puede usar. Distinto de "no_disponible", que es una pieza que no existe.
export type EstadoBase = "medido" | "estimacion" | "no_disponible" | "en_construccion" | "fuera";
export type EstadoBloque = "medido" | "estimacion" | "apagado" | "no_disponible" | "en_construccion" | "fuera";
export type Bandera = keyof Producto["banderas"];

interface Fila {
  base: EstadoBase;
  bandera?: Bandera;
  /* Como se ve con la bandera apagada; por omision "apagado" (en validacion). */
  sinBandera?: "apagado" | "no_disponible";
}

export const BLOQUES = {
  "01.pendientes": { base: "medido" },
  "01.indicadores": { base: "medido" },
  "01.recientes": { base: "medido" },
  "01.tendencia": { base: "medido", bandera: "comparacion_rounds" },
  "01.plan": { base: "fuera" },
  "02.video": { base: "medido" },
  "02.checklist": { base: "medido" },
  "02.estimacion": { base: "medido" },
  "02.consumo": { base: "fuera" },
  "02.individuales": { base: "medido", bandera: "sesiones_individuales" },
  "03.personas": { base: "medido" },
  "03.propagacion": { base: "medido" },
  "03.tramos": { base: "medido" },
  "04.video": { base: "medido" },
  "04.linea": { base: "medido" },
  "04.tipo": { base: "estimacion" },
  "04.guardia": { base: "medido", bandera: "guardia" },
  "04.altura": { base: "estimacion", bandera: "objetivo", sinBandera: "no_disponible" },
  "05.detectados": { base: "medido" },
  "05.distribucion": { base: "estimacion" },
  "05.rounds": { base: "medido", bandera: "comparacion_rounds" },
  "05.guardia": { base: "medido", bandera: "guardia" },
  // 14-10: la altura se muestra como estimacion por decision de producto aunque C8 no paso;
  // la API enciende `objetivo` y la Fight-Card lleva lo medido en el rotulo. Apagada, sigue
  // como pieza no disponible y sin decir cabeza ni cuerpo (CA-14.2).
  "05.altura": { base: "estimacion", bandera: "objetivo", sinBandera: "no_disponible" },
  "05.contacto": { base: "no_disponible" },
  "05.exportar": { base: "medido" },
  "05.combinaciones": { base: "medido" },
  "06.datos": { base: "medido" },
  "06.distribucion": { base: "estimacion" },
  "06.combinaciones": { base: "medido" },
  "06.volumen_round": { base: "medido", bandera: "comparacion_rounds" },
  "06.contextos": { base: "medido" },
  "06.contextos_guardia": { base: "medido", bandera: "guardia" },
  "06.contextos_r3": { base: "medido", bandera: "comparacion_rounds" },
  "06.evolucion": { base: "medido" },
  "06.avatar": { base: "no_disponible" },
  "06.exposicion": { base: "no_disponible" },
  "06.distancia": { base: "no_disponible" },
  "07.seleccion": { base: "medido" },
  "07.dimensiones": { base: "medido" },
  "07.frente": { base: "medido" },
  "07.frente_guardia": { base: "medido", bandera: "guardia" },
  "07.frente_r3": { base: "medido", bandera: "comparacion_rounds" },
  "07.cruces": { base: "no_disponible" },
  "07.lecturas": { base: "en_construccion" },
  "08.subida": { base: "medido" },
  "09.fightcard": { base: "medido" },
} as const satisfies Record<string, Fila>;

export type Bloque = keyof typeof BLOQUES;

/*
 * Sin /producto todavia, un bloque condicionado queda apagado: es preferible no mostrar un
 * numero un segundo que mostrar uno que despues desaparece.
 */
export function estadoDe(bloque: Bloque, producto: Producto | null | undefined): EstadoBloque {
  const f: Fila = BLOQUES[bloque];
  if (f.base === "fuera" || f.base === "no_disponible" || f.base === "en_construccion") return f.base;
  if (f.bandera) {
    const encendida = producto?.banderas?.[f.bandera] ?? false;
    if (!encendida) return f.sinBandera ?? "apagado";
  }
  return f.base;
}

export function visible(bloque: Bloque, producto: Producto | null | undefined): boolean {
  const e = estadoDe(bloque, producto);
  return e === "medido" || e === "estimacion";
}
