/*
 * La unica capa que habla con el servidor.
 *
 * POR QUE EXISTE
 *   Para que el token se ponga en un solo lugar y el 401 no se maneje en cada pantalla.
 *   Un 401 en cualquier pedido borra el token y lleva a /ingresar/?siguiente=<ruta> (02 §6).
 *
 *   Toda respuesta pasa por un esquema zod antes de llegar a un componente (02 §5). Si no
 *   valida, el error dice que version llego y la pantalla no dibuja un documento a medias.
 *
 *   Las rutas son relativas: en desarrollo las reescribe next dev y en produccion la API y
 *   la web comparten origen. No hay URL de API que configurar.
 */
import type { z } from "zod";
import { cerrarSesion, leerToken } from "@/lib/sesion-usuario";

export class ErrorApi extends Error {
  estado: number;
  constructor(mensaje: string, estado: number) {
    super(mensaje);
    this.name = "ErrorApi";
    this.estado = estado;
  }
}

export class ErrorDeContrato extends Error {
  version: string | null;
  constructor(mensaje: string, version: string | null) {
    super(mensaje);
    this.name = "ErrorDeContrato";
    this.version = version;
  }
}

/* Lo que pasa ante un 401. Se puede reemplazar en las pruebas. */
let alVencerse: () => void = () => {
  if (typeof window === "undefined") return;
  const aca = window.location.pathname + window.location.search;
  if (window.location.pathname.startsWith("/ingresar")) return;
  window.location.assign(`/ingresar/?siguiente=${encodeURIComponent(aca)}`);
};

export function alVencerseLaSesion(f: () => void) {
  alVencerse = f;
}

async function detalleDeError(r: Response): Promise<string> {
  try {
    const d = await r.json();
    if (typeof d.detail === "string") return d.detail;
    if (Array.isArray(d.detail)) {
      return d.detail.map((x: { msg?: string }) => x.msg ?? "").filter(Boolean).join("; ");
    }
    return JSON.stringify(d.detail);
  } catch {
    return `error ${r.status}`;
  }
}

export async function pedir(ruta: string, opciones: RequestInit = {}): Promise<unknown> {
  const cabeceras: Record<string, string> = {
    ...((opciones.headers as Record<string, string>) || {}),
  };
  const t = leerToken();
  if (t) cabeceras.Authorization = `Bearer ${t}`;
  if (opciones.body && !(opciones.body instanceof FormData)) {
    cabeceras["Content-Type"] = "application/json";
  }
  let r: Response;
  try {
    r = await fetch(ruta, { ...opciones, headers: cabeceras });
  } catch {
    throw new ErrorApi("no hay conexión con el servidor", 0);
  }
  if (r.status === 401) {
    const detalle = await detalleDeError(r);
    // En el ingreso un 401 es "email o clave incorrectos", no una sesion vencida.
    if (!ruta.startsWith("/auth/")) {
      cerrarSesion();
      alVencerse();
      throw new ErrorApi("se venció la sesión, entrá de nuevo", 401);
    }
    throw new ErrorApi(detalle, 401);
  }
  if (!r.ok) throw new ErrorApi(await detalleDeError(r), r.status);
  if (r.status === 204) return null;
  const tipo = r.headers.get("content-type") || "";
  return tipo.includes("json") ? r.json() : r.text();
}

/* Un pedido cuya respuesta se valida con un esquema. */
export async function pedirValidado<T extends z.ZodType>(
  esquema: T,
  ruta: string,
  opciones?: RequestInit,
): Promise<z.infer<T>> {
  const datos = await pedir(ruta, opciones);
  const r = esquema.safeParse(datos);
  if (!r.success) {
    const version =
      datos && typeof datos === "object" && "version" in datos
        ? String((datos as { version: unknown }).version)
        : null;
    const donde = r.error.issues
      .slice(0, 3)
      .map((i) => `${i.path.join(".") || "(raíz)"}: ${i.message}`)
      .join("; ");
    throw new ErrorDeContrato(
      `La respuesta de ${ruta} no tiene la forma esperada${version ? ` (versión ${version})` : ""}: ${donde}`,
      version,
    );
  }
  return r.data;
}

/* HEAD que solo pregunta si existe, sin bajar el archivo. */
export async function existe(ruta: string): Promise<boolean> {
  try {
    const r = await fetch(ruta, { method: "HEAD" });
    return r.ok;
  } catch {
    return false;
  }
}

export interface Subida {
  promesa: Promise<{ job_id: string; estado: string; bytes: number }>;
  cancelar: () => void;
}

/*
 * La subida va con XMLHttpRequest y no con fetch: fetch no informa el progreso de subida en
 * los navegadores de celular, y un video de varios minutos con una barra quieta se lee como
 * que se colgo. Se puede cancelar. No es reanudable (02 §7).
 */
export function subirConProgreso(
  datos: FormData,
  alAvanzar: (enviados: number, total: number) => void,
): Subida {
  const x = new XMLHttpRequest();
  const promesa = new Promise<{ job_id: string; estado: string; bytes: number }>(
    (resolver, rechazar) => {
      x.open("POST", "/videos");
      const t = leerToken();
      if (t) x.setRequestHeader("Authorization", `Bearer ${t}`);
      x.upload.onprogress = (e) => {
        if (e.lengthComputable) alAvanzar(e.loaded, e.total);
      };
      x.onload = () => {
        if (x.status >= 200 && x.status < 300) {
          resolver(JSON.parse(x.responseText));
          return;
        }
        if (x.status === 401) {
          cerrarSesion();
          alVencerse();
        }
        let detalle = `error ${x.status}`;
        try {
          const d = JSON.parse(x.responseText).detail;
          detalle = typeof d === "string" ? d : detalle;
        } catch {
          /* sin cuerpo util */
        }
        rechazar(new ErrorApi(detalle, x.status));
      };
      x.onerror = () =>
        rechazar(new ErrorApi("se cortó la conexión al subir: hay que subir de nuevo", 0));
      x.onabort = () => rechazar(new ErrorApi("subida cancelada", -1));
      x.send(datos);
    },
  );
  return { promesa, cancelar: () => x.abort() };
}
