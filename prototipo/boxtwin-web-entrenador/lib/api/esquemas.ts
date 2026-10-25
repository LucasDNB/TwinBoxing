/*
 * Los esquemas zod de cada respuesta de la API (02 §5).
 *
 * POR QUE EXISTE
 *   Para que un componente nunca reciba un documento que no entiende. Si la Fight-Card no
 *   valida, la pantalla muestra el error con la version recibida y no dibuja a medias.
 *
 *   La Fight-Card acepta 0.1, 0.2 y 0.3 (05 §4, ampliacion-13-10). Los campos que agregan
 *   la 0.2 y la 0.3 son opcionales en el esquema y `normalizarFightCard` (lib/fightcard.ts)
 *   completa los de 0.2 con su valor neutro; los de 0.3 se calculan solo en el worker, asi que
 *   en una version vieja quedan ausentes y la pantalla lo dice. Una version que no es ninguna
 *   de las tres se rechaza: no se adivina un contrato nuevo.
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
    // Ampliacion del 13-10 (08 §2)
    C8: z.object({ estado: z.enum(["pendiente", "aprobado", "rechazado"]), fecha: strN }).optional(),
    C9: z.object({ estado: z.enum(["pendiente", "aprobado", "rechazado"]), fecha: strN }).optional(),
    C10: z.object({ estado: z.enum(["pendiente", "aprobado", "rechazado"]), fecha: strN }).optional(),
  }),
  banderas: z.object({
    comparacion_rounds: z.boolean(),
    guardia: z.boolean(),
    sesiones_individuales: z.boolean(),
    postura_2d: z.boolean(),
    // 08 §3. Opcionales: una API anterior a la ampliacion no las manda y quedan apagadas.
    objetivo: z.boolean().optional(),
    exposicion: z.boolean().optional(),
    avatar_3d: z.boolean().optional(),
  }),
  dominio: z.object({
    // B10: la regla vive en el anotador; null si la instancia no lo tiene instalado. Opcional:
    // una API anterior a la ampliacion manda `pausa_combinacion_s` y la web tiene que seguir
    // andando hasta que se reinicie (la web y la API se despliegan por separado).
    combinacion: z.object({ patron: z.string(), separacion: z.string(), max_s: num, min_golpes: num.optional() }).nullable().optional(),
    perfil_preliminar_sesiones: num,
  }),
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
  // W19: candidatos en cuadro a la vez con guantes parecidos (el reflejo de un espejo).
  mismo_color_a_la_vez: z.array(num).optional(),
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
  // B17 (cambios-14-10): cargados a mano.
  distancia: z.enum(["larga", "media", "corta"]).nullable().optional(),
  estilo: z.enum(["fajador", "estilista"]).nullable().optional(),
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
  // 0.3 (B11): la altura. C8 no paso (0,726); desde el 14-10 se muestra igual por decision de
  // producto, como estimacion, siempre que la bandera `objetivo` este encendida (CA-14.2).
  objetivo: z.enum(["cabeza", "cuerpo"]).nullable().optional(),
  objetivo_confianza: numN.optional(),
  objetivo_motivo: strN.optional(),
  objetivo_corregido: z.object({ objetivo: z.enum(["cabeza", "cuerpo"]), objetivo_original: strN.optional() }).passthrough().nullable().optional(),
  // Tipo v2 (16-10): probabilidades de recto, hook y uppercut, y el rol que sale de brazo y guardia.
  p_familia: z.array(num).length(3).optional(),
  rol: z.enum(["lead", "rear"]).optional(),
});
export type Golpe = z.infer<typeof Golpe>;

// 0.3 (B11): cabeza + cuerpo + objetivo_sin_dato = total, cuando la sesion paso por enriquecer.
const Altura = { cabeza: num.optional(), cuerpo: num.optional(), objetivo_sin_dato: num.optional() };

const Conteo = z.object({ total: num, izq: num, der: num, ...Altura });

export const Round = z.object({
  round: num, inicio_s: num, fin_s: num, total: num, izq: num, der: num, por_minuto: num, ...Altura,
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

// 0.3 (B10, regla del 14-10): cualquier secuencia de 2 o mas golpes con menos de 0,5 s entre
// cada uno. Tiempo y brazo son medidos; los tipos, estimacion. Una Fight-Card contada antes con
// "izq-der-izq" se sigue leyendo (trae ternas y no trae `brazos`).
export const Combinaciones = z.object({
  patron: z.enum(["cualquiera", "izq-der-izq"]),
  separacion: z.string(),
  max_s: num,
  min_golpes: num.optional(),
  lista: z.array(z.object({
    golpes: z.array(z.string()).min(2),
    t_inicio: num,
    t_fin: num,
    brazos: z.array(z.enum(["izq", "der"]).nullable()).optional(),
    tipos: z.array(strN).min(2),
    round: numN,
  })),
  total: num,
  por_round: z.array(z.object({ round: num, total: num })),
});
export type Combinaciones = z.infer<typeof Combinaciones>;

export const Peleador = z.object({
  golpes: z.array(Golpe),
  por_round: z.array(Round),
  total: Conteo,
  guardia: z.array(EventoGuardia).optional(),
  descartados: num.optional(),
  combinaciones: Combinaciones.optional(),
});
export type Peleador = z.infer<typeof Peleador>;

export const FightCard = z.object({
  version: z.enum(["0.1", "0.2", "0.3"]),
  video: z.object({
    nombre: strN.optional(),
    duracion_s: num,
    fps: num,
    rounds: z.array(z.object({ round: num, inicio_s: num, fin_s: num })),
    // Cambios del 14-10 (DC-5): rounds declarados o segmentos de 3 minutos.
    rounds_origen: z.enum(["declarados", "segmentos_180s"]).nullable().optional(),
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
    // B8b: en que contexto se midio el margen. Una sesion de otro contexto no lo hereda.
    medido_en: Contexto.optional(),
    umbral: numN.optional(),
    // C11 (ampliacion 01 §5): el recall de ternas, cuando se mida.
    combinaciones: z.object({ recall_medido: numN.optional() }).optional(),
  }),
  clasificador: z.object({
    checkpoint: z.string().nullable(),
    exactitud_familia_fuente_no_vista: numN,
    estimado: z.boolean().optional(),
    // Tipo v2 (16-10). Sin `metodo`, la Fight-Card es de PoseConv3D.
    // geometrico-e7: modo académico (25-10), E4 más el 3D de HMR2; trae la leyenda de licencia.
    metodo: z.enum(["poseconv3d", "geometrico-v1", "geometrico-e7"]).optional(),
    licencia: strN.optional(),
    referencia: strN.optional(),
    guardia_supuesta: z.object({ A: z.boolean().optional(), B: z.boolean().optional() }).optional(),
  }),
  peleadores: z.object({ A: Peleador, B: Peleador.optional() }),
  // 0.3 (B11): con que regla se calculo la altura y lo que dio C8.
  metodos: z.object({
    objetivo: z.object({
      regla: z.string(),
      tau: num,
      medido: z.object({
        exactitud_balanceada: num,
        recall_cuerpo: num.optional(),
        golpes: num.optional(),
        pasa: z.boolean().optional(),
      }).nullable().optional(),
    }).optional(),
  }).optional(),
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
}).superRefine((fc, ctx) => {
  // Igual que el schema compartido: una 0.3 trae las combinaciones de cada peleador.
  if (fc.version !== "0.3") return;
  for (const [lado, p] of Object.entries(fc.peleadores)) {
    if (p && !p.combinaciones) ctx.addIssue({ code: "custom", path: ["peleadores", lado, "combinaciones"], message: "una 0.3 trae combinaciones" });
  }
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

// -- Zonas (B16, cambios-14-10) ------------------------------------------------------
// Cabeza o cuerpo, por la mano con que vino el golpe (DC-2). Las fracciones corregidas
// compensan que el detector encuentra menos golpes al cuerpo (DC-4).
export const ZONAS = ["cabeza_mano_izq", "cabeza_mano_der", "cuerpo_mano_izq", "cuerpo_mano_der"] as const;
export type Zona = (typeof ZONAS)[number];

const SesgoDetector = z.object({ recall_cabeza: num, recall_cuerpo: num, corregido: z.boolean() });

const BloqueZonas = z.object({
  zonas: z.record(z.enum(ZONAS), z.object({ n: num, fraccion: numN, fraccion_corregida: numN })).nullable(),
  manos: z.object({ izq: num, der: num }),
  alturas: z.object({ cabeza: num, cuerpo: num, sin_dato: num }).nullable(),
  total: num,
  sin_dato: numN,
  con_dato: numN,
  sesgo_detector: SesgoDetector,
});
export type BloqueZonas = z.infer<typeof BloqueZonas>;

// GET /rival (B18): ataque contra exposicion. La regla vive en la API.
export const Rival = z.object({
  cruces: z.array(z.object({
    atacante: z.enum(["A", "B"]), zona: z.enum(ZONAS),
    ataque_fraccion: num, exposicion_fraccion: num, n_ataque: num, n_exposicion: num,
  })),
  lecturas: z.array(z.object({
    tipo: z.enum(["oportunidad", "amenaza"]), para: z.enum(["A", "B"]), atacante: z.enum(["A", "B"]),
    regla: z.string(), zona: z.string(), texto: z.string(),
  })),
  // DC-10: `excluidas` son las sesiones contra el otro que no entran en la exposicion.
  base: z.object({
    A: z.object({ ataque: num, exposicion: num, excluidas: num.optional() }),
    B: z.object({ ataque: num, exposicion: num, excluidas: num.optional() }),
  }),
  faltan: z.object({ minimo: num, base: z.any() }).nullable(),
  sesgo_detector: SesgoDetector,
});
export type Rival = z.infer<typeof Rival>;

export const Perfil = z.object({
  boxeador: z.object({
    id: z.string(), nombre: z.string(), guardia: strN, notas: strN,
    categoria: strN.optional(), altura_m: numN.optional(), envergadura_m: numN.optional(),
    distancia: z.enum(["larga", "media", "corta"]).nullable().optional(),
    estilo: z.enum(["fajador", "estilista"]).nullable().optional(),
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
  rounds_promedio: z.array(z.object({
    round: num, golpes_por_minuto: num,
    origen: z.enum(["declarados", "segmentos_180s", "mixto"]).optional(),
  })).nullable().optional(),
  // B16 (cambios-14-10): mano por altura de los golpes propios, y lo que le dirigen.
  ataque: BloqueZonas.extend({ tipo: z.string().optional() }).nullable().optional(),
  exposicion: BloqueZonas.extend({
    tipo: z.literal("dirigidos"),
    sesiones: num,
    por_rival: z.array(z.object({ boxeador_id: strN, nombre: z.string(), total: num })),
  }).nullable().optional(),
  // B10: las combinaciones. null si la instancia no tiene el anotador (no se inventa la regla).
  // Una API anterior manda la lista por tipo de B7: se lee como null, no como error.
  combinaciones: z.preprocess((v) => (Array.isArray(v) ? null : v), z.object({
    patron: z.string(),
    total: num,
    por_minuto: numN,
    por_contexto: z.record(z.string(), num),
    // Regla del 14-10: como se reparten por largo y que secuencias de brazos se repiten.
    por_largo: z.record(z.string(), num).optional(),
    secuencias_frecuentes: z.array(z.object({ brazos: z.array(z.string()), n: num })).optional(),
    tipos_frecuentes: z.array(z.object({ tipos: z.array(z.string()), n: num, estimado: z.boolean() })),
    sesiones_con_combinaciones: num,
    recall_medido: numN.optional(),
  }).nullable().optional()),
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
  // B8b: en bolsa y sombra no hay B.
  cobertura: z.object({ A: num, B: num.optional(), sin_asignar: num }),
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
