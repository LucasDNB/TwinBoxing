/* Un perfil con la forma de B7, para las pruebas de 06 y 07. */
export function perfilDePrueba(id: string, nombre: string, extra: Record<string, unknown> = {}) {
  return {
    nomenclatura: { jab: "jab", cross: "gancho", hook: "hook", uppercut: "uppercut" },
    contexto: "todas",
    sesiones_por_contexto: { bolsa: 0, sombra: 0, sparring: 2, combate: 1 },
    rounds_total: 9,
    por_contexto: [
      { contexto: "sparring", sesiones: 2, golpes_por_minuto: 12.8, jab_fraccion: 0.37, guardia_opuesta_arriba: null, caida_ultimo_round: null },
      { contexto: "combate", sesiones: 1, golpes_por_minuto: 15.1, jab_fraccion: 0.41, guardia_opuesta_arriba: null, caida_ultimo_round: null },
    ],
    rounds_promedio: null,
    combinaciones: {
      patron: "cualquiera", total: 23, por_minuto: 0.41, por_contexto: { sparring: 15, combate: 8 },
      por_largo: { "2": 14, "3": 7, "4+": 2 },
      secuencias_frecuentes: [{ brazos: ["izq", "der"], n: 9 }, { brazos: ["der", "izq"], n: 6 }],
      tipos_frecuentes: [{ tipos: ["jab", "cross", "hook"], n: 9, estimado: true }],
      sesiones_con_combinaciones: 3,
      recall_medido: 0.484,
    },
    boxeador: { id, nombre, guardia: "ortodoxa", notas: null, categoria: "Welter", altura_m: 1.75, envergadura_m: 1.8 },
    sesiones: [
      { sesion_id: "s1", nombre: "a", fecha: "2026-09-21T10:00:00", lado: "A", golpes: 40, minutos: 3, golpes_por_minuto: 13.3, mezcla: { jab: 10, cross: 8 }, sin_clasificar: 2, detector: "det-v1", clasificador: "clf-v1", exactitud_familia: null, contexto: "sparring" },
    ],
    totales: { sesiones: 3, golpes: 120, mezcla: { jab: 37, cross: 30, hook: 20, uppercut: 13 }, brazos: {}, sin_clasificar: 20, golpes_por_minuto_medio: 13.4 },
    evolucion: [
      { fecha: "2026-09-21T10:00:00", sesion_id: "s1", golpes: 40, golpes_por_minuto: 13.3, detector: "det-v1", contexto: "sparring" },
      { fecha: "2026-09-28T10:00:00", sesion_id: "s2", golpes: 38, golpes_por_minuto: 12.3, detector: "det-v1", contexto: "sparring" },
      { fecha: "2026-10-05T10:00:00", sesion_id: "s3", golpes: 42, golpes_por_minuto: 15.1, detector: "det-v2", contexto: "combate" },
    ],
    avisos: [],
    ...extra,
  };
}

const SESGO = { recall_cabeza: 0.512, recall_cuerpo: 0.348, corregido: true };

/* Un bloque de zonas (B16) con conteos dados, en el orden cabeza izq, cabeza der, cuerpo izq, cuerpo der. */
export function zonasDePrueba(n: [number, number, number, number], extra: Record<string, unknown> = {}) {
  const nombres = ["cabeza_mano_izq", "cabeza_mano_der", "cuerpo_mano_izq", "cuerpo_mano_der"] as const;
  const conDato = n.reduce((a, x) => a + x, 0);
  const pesos = n.map((x, i) => x / (i < 2 ? SESGO.recall_cabeza : SESGO.recall_cuerpo));
  const suma = pesos.reduce((a, x) => a + x, 0);
  const zonas = Object.fromEntries(nombres.map((z, i) => [z, {
    n: n[i], fraccion: conDato ? n[i] / conDato : null, fraccion_corregida: suma ? pesos[i] / suma : null,
  }]));
  return {
    zonas, manos: { izq: n[0] + n[2] + 3, der: n[1] + n[3] + 2 },
    alturas: { cabeza: n[0] + n[1], cuerpo: n[2] + n[3], sin_dato: 5 },
    total: conDato + 5, sin_dato: 5, con_dato: conDato, sesgo_detector: SESGO, ...extra,
  };
}

/* La respuesta de GET /rival con dos lecturas de zona coincidente, una por sentido. */
export function rivalDePrueba() {
  const zonas = ["cabeza_mano_izq", "cabeza_mano_der", "cuerpo_mano_izq", "cuerpo_mano_der"] as const;
  const fr = [0.4, 0.3, 0.1, 0.2];
  const cruces = (["A", "B"] as const).flatMap((at) => zonas.map((z, i) => ({
    atacante: at, zona: z, ataque_fraccion: fr[i], exposicion_fraccion: fr[3 - i], n_ataque: 40 - i, n_exposicion: 30 + i,
  })));
  return {
    cruces,
    lecturas: [
      { tipo: "oportunidad", para: "A", atacante: "A", regla: "zona_coincidente", zona: "cabeza_mano_izq", texto: "Ana Paz suele ir a la cabeza con la izquierda (40%), y ahí es donde a Bea Ruiz suelen buscarlo (35%)." },
      { tipo: "amenaza", para: "B", atacante: "A", regla: "zona_coincidente", zona: "cabeza_mano_izq", texto: "Ana Paz suele ir a la cabeza con la izquierda (40%), y ahí es donde a Bea Ruiz suelen buscarlo (35%)." },
      { tipo: "oportunidad", para: "B", atacante: "B", regla: "cuerpo", zona: "cuerpo", texto: "A Ana Paz no suelen trabajarlo al cuerpo (12%) y Bea Ruiz lo hace (30%)." },
      { tipo: "amenaza", para: "A", atacante: "B", regla: "cuerpo", zona: "cuerpo", texto: "A Ana Paz no suelen trabajarlo al cuerpo (12%) y Bea Ruiz lo hace (30%)." },
    ],
    base: { A: { ataque: 120, exposicion: 90 }, B: { ataque: 80, exposicion: 70 } },
    faltan: null,
    sesgo_detector: SESGO,
  };
}
