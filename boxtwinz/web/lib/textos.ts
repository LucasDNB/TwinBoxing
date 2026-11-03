/*
 * Los rotulos que la spec fija palabra por palabra (seccion 12). El test de textos
 * obligatorios (CZ7) busca estos mismos strings en las pantallas.
 */
export const TEXTOS = {
  golpes: "golpes detectados",
  perfilSimulado: "Perfil simulado",
  pagoSimulado: "Pago simulado — versión de demostración",
  estimacion: "estimación",
  roundsSinValidar: "Comparación entre rounds sin validar",
  enEvaluacion: "Reconocimiento en evaluación",
  consentimientoUso:
    "Tengo 18 años o más y cuento con la autorización de las personas que aparecen en los videos que cargo. No cargo videos con menores de edad.",
  consentimientoReent:
    "Autorizo a usar mis videos y valoraciones para reentrenar el modelo. Lo que ya se usó en un modelo entrenado no puede retirarse de ese modelo.",
  pieRanking: "Ranking de actividad: cuenta golpes detectados, no golpes que conectaron",
} as const;

export const MODALIDAD: Record<string, string> = { sombra: "Sombra", bolsa: "Bolsa", sparring: "Sparring" };
