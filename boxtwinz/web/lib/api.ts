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
