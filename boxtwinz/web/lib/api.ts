/*
 * Cliente de la API de BoxTwinZ. Mismo origen y cookie httpOnly: no hay token en el codigo.
 * Los errores llegan con un codigo estable (plan_requerido, cupo_agotado, ...) y la interfaz
 * decide por el codigo, nunca por el texto.
 */
export class ErrorApi extends Error {
  constructor(public status: number, public codigo: string, mensaje: string) {
    super(mensaje);
  }
}

async function leerError(r: Response): Promise<ErrorApi> {
  try {
    const d = await r.json();
    const det = d?.detail;
    if (det && typeof det === "object" && "codigo" in det) return new ErrorApi(r.status, det.codigo, det.mensaje);
    if (Array.isArray(det)) return new ErrorApi(r.status, "invalido", "Revisá los datos del formulario");
    return new ErrorApi(r.status, "error", String(det ?? r.statusText));
  } catch {
    return new ErrorApi(r.status, "error", r.statusText || "Error de red");
  }
}

export async function api<T>(ruta: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const { json, ...resto } = init;
  const r = await fetch(`/api${ruta}`, {
    credentials: "same-origin",
    ...resto,
    headers: json !== undefined ? { "Content-Type": "application/json", ...resto.headers } : resto.headers,
    body: json !== undefined ? JSON.stringify(json) : resto.body,
  });
  if (!r.ok) throw await leerError(r);
  if (r.status === 204) return undefined as T;
  return (await r.json()) as T;
}

export type Yo = {
  id: string; email: string; nombre: string; plan: "free" | "premium"; rol: "usuario" | "admin";
  guardia: "ortodoxa" | "zurda" | null; consentimiento_reentrenamiento: boolean; pago_simulado: string | null;
};

export type Perfil = {
  id: string; ranura: number; nombre: string; estilo: string; guardia: string | null; publico: boolean;
  customizado: boolean; rotulo: string | null; arquetipo: { id: string; clave: string; nombre: string } | null;
};

export type SesionResumen = {
  id: string; perfil_id: string; modalidad: "sombra" | "bolsa" | "sparring"; estado: string;
  duracion_s: number | null; creado: string; rotulo: string | null; error: string | null;
};

export type Metrica = {
  golpes: number; volumen: number; pct_izquierda: number | null; pct_mano_adelantada: number | null;
  combinaciones_min: number; pct_golpes_en_combinacion: number | null; sosten_ritmo: number | null;
  iniciativa: number | null; curva?: number[];
};
export type Segmento = Metrica & { segmento: number; inicio_s: number; fin_s: number };
export type BloqueMetricas = { total: Metrica | null; segmentos: Segmento[] };
export type GolpeR = {
  id: string; quien: "usuario" | "rival"; mano: "izquierda" | "derecha"; t_inicio: number; t_fin: number;
  confianza: number; altura_estimada: string | null;
};
export type Consejos = { consejos: { clave: string; texto: string }[]; pie: string };
export type Exposicion = {
  rival_izquierda: number; rival_derecha: number; lado_izquierdo_usuario: number; lado_derecho_usuario: number; aclaracion: string;
};
export type SesionDetalle = SesionResumen & {
  rival_perfil_id: string | null; rounds_declarados: number | null; calidad_pose: number | null;
  apta_ranking: boolean; error_etapa: string | null; vista_resultado: boolean;
};
export type Resultado = {
  sesion: SesionDetalle; golpes: GolpeR[]; metricas: { usuario?: BloqueMetricas; rival?: BloqueMetricas };
  exposicion: Exposicion | null; altura: { rotulo: string; cabeza: number; cuerpo: number } | null;
  consejos: Consejos; valorada: boolean; valoracion_obligatoria: boolean;
  rotulos: { golpes: string; evaluacion: string | null; rounds: string; guardia_rival: string | null };
};
export type Cupo = { mes: string; limite_min: number; usados_min: number; restantes_min: number };
