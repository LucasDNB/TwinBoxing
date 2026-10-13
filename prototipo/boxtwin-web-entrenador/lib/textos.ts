/*
 * Los mensajes que se repiten (04, "Mensajes que se repiten").
 *
 * Se definen una vez. Los que llevan numeros los toman del documento o de /producto: si el
 * detector mejora, la pantalla mejora sola (03 §6). Lenguaje de gimnasio: "tramo" y no
 * "track", "marca" y no "sembra" (03 §8).
 */
import { decimal, porcentajeAR } from "@/lib/formato";

/* round(valor x 10) de cada 10 (03 §6). */
export function deCada10(x: number): number {
  return Math.min(10, Math.max(0, Math.round(x * 10)));
}

function seEscapa(recall: number): string {
  const faltan = 10 - deCada10(recall);
  if (faltan === 5) return "la mitad";
  if (faltan === 0) return "casi ninguno";
  return `${faltan} de cada 10`;
}

export const textos = {
  margenDetector(precision: number, recall: number) {
    return `Acierta ${deCada10(precision)} de cada 10 que marca y se le escapa ${seEscapa(recall)}. El volumen real es mayor.`;
  },
  /* B10 (ampliacion 01 §4). {r} es el recall de golpes, o el de ternas cuando C11 se mida. */
  margenCombinaciones(recall: number) {
    return `Para contar una combinación el sistema tiene que encontrar los tres golpes. Encuentra cerca de ${deCada10(recall)} de cada 10 golpes, así que se le escapan la mayoría de las combinaciones: el número sirve para comparar sesiones, no como total real.`;
  },
  combinacionesVacio: "El sistema no encontró combinaciones completas en esta sesión.",
  combinacionesVersionVieja:
    "Esta sesión se procesó antes de contar combinaciones; al reprocesarla aparecen.",
  /* B8b (ampliacion 02 §2): el margen medido en sparring no se traslada a otro contexto. */
  margenOtroContexto(medidoEn: string, contexto: string) {
    const en = (NOMBRE_CONTEXTO[medidoEn] ?? medidoEn).toLowerCase();
    const ctx = (NOMBRE_CONTEXTO[contexto] ?? contexto).toLowerCase();
    return `El margen del detector se midió en ${en}; en ${ctx} todavía no se midió. Los golpes que se ven son los que el sistema detectó.`;
  },
  individual: {
    titulo: "¿Cuál es el boxeador?",
    indicacion: "Tocá una vez al que está trabajando. El resto del video se sigue por el color de sus guantes.",
    espejo: "Hay otra figura con los mismos guantes al mismo tiempo. Si estás frente a un espejo, elegí al de verdad, no al reflejo.",
  },
  margenPanel(recall: number) {
    return `El sistema encuentra cerca de ${deCada10(recall)} de cada 10 golpes: el total real es mayor.`;
  },
  estimacionTipo(checkpoint: string | null | undefined, exactitud: number | null | undefined) {
    const ck = checkpoint && checkpoint.trim() ? checkpoint : "sin versión registrada";
    const ex = exactitud == null ? "sin medir" : porcentajeAR(exactitud);
    return `Tipo estimado por el modelo ${ck}. Exactitud sobre material nuevo: ${ex}.`;
  },
  condicionadoC2:
    "La comparación entre rounds se muestra cuando se confirme que el sistema detecta parejo en todos los rounds.",
  condicionadoC3:
    "La guardia se muestra cuando el indicador pase la validación contra marcado a mano.",
  nodispContacto:
    "Una cámara sola no ve si el golpe tocó. Para estimarlo hace falta una pieza que todavía no está.",
  nodispAltura: "Esta versión del modelo dice el tipo de golpe, no si fue a la cabeza o al cuerpo.",
  nodisp3d:
    "El giro de un golpe alrededor del cuerpo no se ve con una cámara de frente. El avatar necesita más de una cámara.",
  nodispExposicion:
    "Saber dónde recibe cada uno necesita saber si el golpe tocó, y una cámara sola no lo ve.",
  nodispDistancia:
    "La distancia de trabajo necesita que se definan las bandas larga, media y corta como criterio de dominio.",
  sinAsignar(fraccion: number) {
    return `El ${porcentajeAR(fraccion)} del video quedó sin saber quién es quién. Los golpes de esos tramos no se cuentan.`;
  },
  notaCorreccion(checkpoint: string | null | undefined) {
    const ck = checkpoint && checkpoint.trim() ? checkpoint : "sin versión registrada";
    return `El tipo de golpe es una estimación del modelo ${ck} y mejora con cada versión. Tu corrección se guarda para entrenar y no cambia el conteo de golpes detectados.`;
  },
  tramosVersionVieja:
    "Esta sesión se procesó antes de la revisión de tramos; reprocesala para revisarlos.",
  enConstruccion: {
    titulo: "En construcción",
    individuales:
      "Las sesiones de bolsa y sombra todavía se están validando. Por ahora se analizan sparring y combate.",
    lecturas:
      "Las lecturas de oportunidades y amenazas se están revisando como criterio de dominio. Mientras tanto, el frente a frente muestra los números medidos.",
  },
  rivalAviso:
    "Lectura cualitativa entre perfiles ya construidos. No es un pronóstico ni una probabilidad de victoria.",
  separacion(sep: number, piso: number) {
    return `separación de color ${decimal(sep, 2)} (mínimo ${decimal(piso, 2)})`;
  },
  consentimiento:
    "Las personas que aparecen en el video dieron su consentimiento por escrito y ninguna es menor de 18 años.",
  contextoLectura:
    "Cada contexto mide algo distinto: se compara cómo cambia él, no contra una línea de base común.",
  motivoTramo: {
    color_insuficiente: "guante tapado o poca luz",
    sin_coexistencia: "cruce o cuerpo a cuerpo",
    corto: "salió de cuadro",
    abstencion: "no se pudo decidir",
  } as Record<string, string>,
};

export const NOMBRE_CONTEXTO: Record<string, string> = {
  bolsa: "Bolsa", sombra: "Sombra", sparring: "Sparring", combate: "Combate",
};

export const DESCRIPCION_CONTEXTO: Record<string, string> = {
  bolsa: "Un peleador trabajando la bolsa.",
  sombra: "Un peleador sin rival ni bolsa.",
  sparring: "Dos peleadores, guantes de distinto color.",
  combate: "Pelea con rounds reglamentarios.",
};

export const NOMBRE_PRACTICA: Record<string, string> = {
  tecnica_libre: "Técnica libre", combinaciones: "Combinaciones", defensa: "Defensa",
};

export const NOMBRE_ESTADO: Record<string, string> = {
  subida: "Subiendo",
  en_cola: "En cola",
  procesando: "Procesando",
  espera_siembra: "Falta un paso",
  completando: "Completando",
  listo: "Listo",
  fallo: "Falló",
};

export const NOMBRE_ETAPA: Record<string, string> = {
  pose: "pose",
  guantes: "guantes",
  preproceso: "pose",
  evidencia: "guantes",
  identidad: "identidad",
  detector: "detector",
  guardia: "guardia",
  clasificar: "tipo de golpe",
  render: "video con esqueleto",
};

/* Los nombres del diseno que no pueden aparecer en el build (CA-01.2, 06 §6). */
export const NOMBRES_DE_DEMOSTRACION = ["Tomás Ríos", "Gimnasio Demo", "M. Gómez"];
