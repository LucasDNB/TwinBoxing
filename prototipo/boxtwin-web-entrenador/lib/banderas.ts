/*
 * Que estado tiene cada bloque del diseno (01 §4), segun /producto.
 *
 * POR QUE EXISTE
 *   La web nunca decide sola si un bloque condicionado se muestra: lo dice el servidor. Esta
 *   tabla es la de 01 §4 escrita una vez; cada pantalla pregunta `estadoDe(bloque, producto)`
 *   y dibuja lo que corresponde con <EstadoBloque>.
 */
import type { Cuenta, Producto, Version } from "@/lib/api/esquemas";

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
  // W20: la division por altura dentro de la distribucion, con la misma bandera que la Fight-Card.
  "06.altura": { base: "estimacion", bandera: "objetivo", sinBandera: "no_disponible" },
  "06.combinaciones": { base: "medido" },
  "06.volumen_round": { base: "medido", bandera: "comparacion_rounds" },
  "06.contextos": { base: "medido" },
  "06.contextos_guardia": { base: "medido", bandera: "guardia" },
  "06.contextos_r3": { base: "medido", bandera: "comparacion_rounds" },
  "06.evolucion": { base: "medido" },
  // Cambios del 14-10: el avatar es ilustrativo (DC-6) y la distancia, cargada a mano (DC-7);
  // los dos se dibujan siempre y lo dicen. El mapa sale de la altura (DC-3, DC-4).
  "06.avatar": { base: "medido" },
  "06.exposicion": { base: "estimacion", bandera: "exposicion", sinBandera: "no_disponible" },
  "06.distancia": { base: "medido" },
  "07.seleccion": { base: "medido" },
  "07.dimensiones": { base: "medido" },
  "07.frente": { base: "medido" },
  "07.frente_guardia": { base: "medido", bandera: "guardia" },
  "07.frente_r3": { base: "medido", bandera: "comparacion_rounds" },
  // DC-8: "Oportunidades y amenazas" y el cruce de zonas son un solo bloque.
  "07.cruces": { base: "estimacion", bandera: "exposicion", sinBandera: "no_disponible" },
  "08.subida": { base: "medido" },
  "09.fightcard": { base: "medido" },
} as const satisfies Record<string, Fila>;

export type Bloque = keyof typeof BLOQUES;

/*
 * Sin /producto todavia, un bloque condicionado queda apagado: es preferible no mostrar un
 * numero un segundo que mostrar uno que despues desaparece.
 */
/*
 * Segundo eje (versiones 26-10, 03 §1): un bloque que no es de la version vuelve "fuera" y no
 * se dibuja. No es "no disponible": en esa version no existe. La matriz es la de 01 §4.
 */
const FUERA_POR_VERSION: Partial<Record<Version, readonly Bloque[]>> = {
  individual: ["07.seleccion", "07.dimensiones", "07.frente", "07.frente_guardia", "07.frente_r3", "07.cruces"],
};
/* D4 se revierte en parte: las horas vuelven, medidas, cuando la cuenta las trae (B20). */
const CON_HORAS: readonly Bloque[] = ["01.plan", "02.consumo"];

export function estadoDe(
  bloque: Bloque,
  producto: Producto | null | undefined,
  cuenta?: Cuenta | null,
): EstadoBloque {
  if (cuenta && FUERA_POR_VERSION[cuenta.version]?.includes(bloque)) return "fuera";
  if (CON_HORAS.includes(bloque)) return cuenta?.horas ? "medido" : "fuera";
  const f: Fila = BLOQUES[bloque];
  if (f.base === "fuera" || f.base === "no_disponible" || f.base === "en_construccion") return f.base;
  if (f.bandera) {
    const encendida = producto?.banderas?.[f.bandera] ?? false;
    if (!encendida) return f.sinBandera ?? "apagado";
  }
  return f.base;
}

export function visible(bloque: Bloque, producto: Producto | null | undefined, cuenta?: Cuenta | null): boolean {
  const e = estadoDe(bloque, producto, cuenta);
  return e === "medido" || e === "estimacion";
}
