/*
 * Los esquemas zod de cada respuesta de la API (02 §5).
 *
 * POR QUE EXISTE
 *   Para que un componente nunca reciba un documento que no entiende. Si la Fight-Card no
 *   valida, la pantalla muestra el error con la version recibida y no dibuja a medias.
 *
 *   La Fight-Card acepta 0.1 y 0.2 (05 §4). Los campos que agrega la 0.2 son opcionales en
 *   el esquema y `normalizarFightCard` (lib/fightcard.ts) los completa con su valor neutro.
 *   Una version que no es ninguna de las dos se rechaza: no se adivina un contrato nuevo.
 */
import { z } from "zod";

const num = z.number();
const numN = z.number().nullable();
const strN = z.string().nullable();

export const ESTADOS_SESION = [
  "subida", "en_cola", "procesando", "espera_siembra", "completando", "listo", "fallo",
] as const;
export const EstadoSesion = z.enum(ESTADOS_SESION);
export type EstadoSesion = z.infer<typeof EstadoSesion>;

export const CONTEXTOS = ["bolsa", "sombra", "sparring", "combate"] as const;
export const Contexto = z.enum(CONTEXTOS);
export type Contexto = z.infer<typeof Contexto>;

export const PRACTICAS = ["tecnica_libre", "combinaciones", "defensa"] as const;

export const Producto = z.object({
  registro: z.enum(["abierto", "invitacion", "cerrado"]),
  max_bytes: num,
  max_minutos: num,
  formatos: z.array(z.string()),
  factor_procesamiento: z.object({
    min: num, mediana: num, max: num, medido_sobre: z.string(),
  }),
  criterios: z.object({
    C2: z.object({ estado: z.enum(["pendiente", "aprobado", "rechazado"]), fecha: strN }),
    C3: z.object({ estado: z.enum(["pendiente", "aprobado", "rechazado"]), fecha: strN }),
  }),
  banderas: z.object({
    comparacion_rounds: z.boolean(),
    guardia: z.boolean(),
    sesiones_individuales: z.boolean(),
    postura_2d: z.boolean(),
  }),
  dominio: z.object({ pausa_combinacion_s: num, perfil_preliminar_sesiones: num }),
});
export type Producto = z.infer<typeof Producto>;

export const Token = z.object({ token: z.string(), tipo: z.string(), usuario: z.string() });

export const Progreso = z.object({
  etapa: z.string(),
  hecho: num.optional(),
  total: num.optional(),
  fraccion: numN.optional(),
  segundos: num.optional(),
});
export type Progreso = z.infer<typeof Progreso>;

export const BoxeadorCorto = z.object({ id: z.string(), nombre: z.string() }).nullable();

export const ColorGuante = z.object({ hex: z.string(), nombre: z.string() }).nullable();

export const Candidato = z.object({
  track: num,
  cuadro: num.optional(),
  alto: num.optional(),
  recorte_url: strN.optional(),
  coexiste_con: numN.optional(),
  cuadros_de_coexistencia: num.optional(),
  avisos: z.array(z.string()).optional(),
  // B2. Opcionales: sin B2 la siembra funciona con el recorte solo (01 §7, riesgo 2).
  color_guante: ColorGuante.optional(),
  t_desde: numN.optional(),
  t_hasta: numN.optional(),
  alto_relativo: numN.optional(),
  sugerido: z.boolean().optional(),
});
export type Candidato = z.infer<typeof Candidato>;

export const Sesion = z.object({
  id: z.string(),
  nombre: z.string(),
  estado: EstadoSesion,
  error: strN,
  video: z.string(),
  duracion_s: numN,
  fps: numN,
  rounds: z.object({
    round_s: numN,
    descanso_s: numN,
    cantidad: numN.optional(),
  }),
  siembra: z.object({ track_a: num, track_b: numN }).nullable(),
  creada: strN,
  actualizada: strN,
  // B1
  contexto: Contexto.optional(),
  practica: strN.optional(),
  boxeadores: z.object({ A: BoxeadorCorto, B: BoxeadorCorto }).optional(),
  consentimiento: z.boolean().optional(),
  // GET /jobs/{id}, y GET /jobs para las que estan en curso (B5)
  progreso: Progreso.nullable().optional(),
  candidatos: z.array(Candidato).optional(),
  avisos: z.array(z.string()).optional(),
  pareja_sugerida: z.array(num).nullable().optional(),
  trabajos: z
    .array(z.object({ etapa: z.string(), estado: z.string(), intento: num, error: strN }))
    .optional(),
});
export type Sesion = z.infer<typeof Sesion>;
export const ListaSesiones = z.array(Sesion);

export const Boxeador = z.object({
  id: z.string(),
  nombre: z.string(),
  guardia: z.enum(["ortodoxa", "zurda"]).nullable(),
  notas: strN,
  creado: z.string(),
  // B6
  categoria: strN.optional(),
  altura_m: numN.optional(),
  envergadura_m: numN.optional(),
  sesiones: num.optional(),
  ultima_sesion: strN.optional(),
});
export type Boxeador = z.infer<typeof Boxeador>;
export const ListaBoxeadores = z.array(Boxeador);

// -- Fight-Card ----------------------------------------------------------------

export const TIPOS_GOLPE = ["jab", "cross", "hook", "uppercut"] as const;
export type TipoGolpe = (typeof TIPOS_GOLPE)[number];
export type TipoCorreccion = TipoGolpe | "no_golpe";

export const Golpe = z.object({
  id: z.string(),
  t_inicio: num,
  t_fin: num,
  cuadro_inicio: num.optional(),
  cuadro_fin: num.optional(),
  brazo: z.string(),
  score: num.optional(),
  tipo: strN,
  confianza_tipo: numN,
  tipo_clasificador: strN.optional(),
  corregido: z
    .object({
      tipo: z.string(),
      tipo_original: strN.optional(),
      checkpoint_original: strN.optional(),
      cuando: strN.optional(),
      por: strN.optional(),
    })
    .nullable(),
  descartado: z.boolean().optional(),
});
export type Golpe = z.infer<typeof Golpe>;

const Conteo = z.object({ total: num, izq: num, der: num });

export const Round = z.object({
  round: num, inicio_s: num, fin_s: num, total: num, izq: num, der: num, por_minuto: num,
});
export type Round = z.infer<typeof Round>;

export const EventoGuardia = z.object({
  golpe: z.string(),
  peleador: z.string(),
  brazo: z.string(),
  t: num,
  retorno_ms: numN.optional(),
  retorno_lento: z.boolean().nullable().optional(),
  fraccion_opuesta_afuera: numN.optional(),
  mano_opuesta_caida: z.boolean().nullable().optional(),
  medible: z.boolean().optional(),
  motivo: strN.optional(),
});

export const Peleador = z.object({
  golpes: z.array(Golpe),
  por_round: z.array(Round),
  total: Conteo,
  guardia: z.array(EventoGuardia).optional(),
  descartados: num.optional(),
});
export type Peleador = z.infer<typeof Peleador>;

export const FightCard = z.object({
  version: z.enum(["0.1", "0.2"]),
  video: z.object({
    nombre: strN.optional(),
    duracion_s: num,
    fps: num,
    rounds: z.array(z.object({ round: num, inicio_s: num, fin_s: num })),
  }),
  identidad: z.object({
    cobertura_A: num.optional(),
    cobertura_B: num.optional(),
    sin_asignar: num.optional(),
    cuadros: num.optional(),
    separacion_de_perfiles: numN.optional(),
    piso_de_separacion: numN.optional(),
    tracks_sin_asignar: numN.optional(),
    // 0.2 (B3)
    tramos_total: numN.optional(),
    tramos_asignados_auto: numN.optional(),
    tramos_asignados_manual: numN.optional(),
    separacion_color: numN.optional(),
  }),
  detector: z.object({
    checkpoint: z.string(),
    precision_medida: num,
    recall_medido: num,
    medido_sobre: z.string().optional(),
    umbral: numN.optional(),
  }),
  clasificador: z.object({
    checkpoint: z.string().nullable(),
    exactitud_familia_fuente_no_vista: numN,
    estimado: z.boolean().optional(),
  }),
  peleadores: z.object({ A: Peleador, B: Peleador.optional() }),
  nomenclatura: z.record(z.string(), z.string()),
  no_incluye: z.array(z.string()),
  avisos: z.array(z.string()).optional(),
  // Lo agrega GET /fightcards/{id} desde la base; no vive en el archivo (05 §4).
  sesion: z
    .object({
      contexto: Contexto.nullable().optional(),
      practica: strN.optional(),
      boxeadores: z.object({
        A: z.object({ id: z.string(), nombre: z.string(), guardia: strN.optional(), categoria: strN.optional() }).nullable(),
        B: z.object({ id: z.string(), nombre: z.string(), guardia: strN.optional(), categoria: strN.optional() }).nullable(),
      }),
    })
    .optional(),
});
export type FightCardCruda = z.infer<typeof FightCard>;

// -- Perfil (B7 extiende lo que ya devuelve) ------------------------------------

export const EntradaPerfil = z.object({
  sesion_id: z.string(),
  nombre: z.string(),
  fecha: z.string(),
  lado: z.string(),
  golpes: num,
  minutos: numN,
  golpes_por_minuto: numN,
  mezcla: z.record(z.string(), num),
  sin_clasificar: num,
  detector: strN,
  clasificador: strN,
  exactitud_familia: numN.optional(),
  contexto: z.string().nullable().optional(),
});

export const PasoCombinacion = z.object({ familia: z.string(), lado: z.enum(["adelante", "atras"]).nullable() });

export const Perfil = z.object({
  boxeador: z.object({
    id: z.string(), nombre: z.string(), guardia: strN, notas: strN,
    categoria: strN.optional(), altura_m: numN.optional(), envergadura_m: numN.optional(),
  }),
  sesiones: z.array(EntradaPerfil),
  totales: z.object({
    sesiones: num, golpes: num, mezcla: z.record(z.string(), num),
    brazos: z.record(z.string(), num), sin_clasificar: num, golpes_por_minuto_medio: numN,
  }),
  evolucion: z.array(z.object({
    fecha: z.string(), sesion_id: z.string(), golpes: num, golpes_por_minuto: numN,
    detector: strN.optional(),
  })),
  avisos: z.array(z.string()),
  // B7
  nomenclatura: z.record(z.string(), z.string()).optional(),
  contexto: z.string().optional(),
  sesiones_por_contexto: z.record(z.string(), num).optional(),
  rounds_total: num.optional(),
  por_contexto: z.array(z.object({
    contexto: z.string(),
    sesiones: num.optional(),
    golpes_por_minuto: numN,
    jab_fraccion: numN,
    guardia_opuesta_arriba: numN,
    caida_ultimo_round: numN,
  })).optional(),
  rounds_promedio: z.array(z.object({ round: num, golpes_por_minuto: num })).nullable().optional(),
  combinaciones: z.array(z.object({
    secuencia: z.array(PasoCombinacion), n: num, estimado: z.boolean(),
  })).optional(),
});
export type Perfil = z.infer<typeof Perfil>;

// -- Panel (B5) -------------------------------------------------------------------

export const Panel = z.object({
  peleadores_activos: num,
  peleadores_preliminares: num,
  sesiones_mes: z.object({
    total: num, listo: num, en_curso: num, espera_siembra: num, fallo: num,
  }),
  rounds_mes: z.object({ total: num }).catchall(num),
  golpes_detectados_mes: num,
  recall_detector: numN,
  pendientes: z.array(z.object({ sesion_id: z.string(), estado: z.string(), titulo: z.string() })),
});
export type Panel = z.infer<typeof Panel>;

// -- Tramos (B3) ------------------------------------------------------------------

export const MOTIVOS_TRAMO = ["color_insuficiente", "sin_coexistencia", "corto", "abstencion"] as const;

export const Tramos = z.object({
  cobertura: z.object({ A: num, B: num, sin_asignar: num }),
  separacion_color: numN,
  tramos: z.array(z.object({
    track: num,
    t_desde: num,
    t_hasta: num,
    motivo: z.enum(MOTIVOS_TRAMO),
    recorte_url: strN,
    asignado: z.enum(["A", "B", "ninguno"]).nullable(),
  })),
});
export type Tramos = z.infer<typeof Tramos>;
