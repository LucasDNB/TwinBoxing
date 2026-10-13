/*
 * Lo que la web decide sobre un video antes de subirlo (02 §7, 04 §02).
 *
 * POR QUE EXISTE
 *   Para no descubrir un 413 despues de subir un gigabyte por datos moviles. Formato, peso y
 *   duracion se validan en el cliente con los mismos limites que la API, que los publica en
 *   /producto. Los fps no: el navegador no los expone de forma confiable y los informa el
 *   servidor despues de procesar.
 */
import type { Producto } from "@/lib/api/esquemas";
import { decimal, megabytes } from "@/lib/formato";

export interface Metadatos {
  duracion_s: number;
  ancho: number;
  alto: number;
}

export function extension(nombre: string): string {
  const i = nombre.lastIndexOf(".");
  return i >= 0 ? nombre.slice(i).toLowerCase() : "";
}

/* El motivo por el que no se puede subir, o null (CA-02.1). */
export function validarArchivo(
  archivo: { name: string; size: number },
  producto: Pick<Producto, "formatos" | "max_bytes">,
): string | null {
  const ext = extension(archivo.name);
  if (!producto.formatos.includes(ext)) {
    return `El formato ${ext || "sin extensión"} no se acepta. Se aceptan ${producto.formatos.join(", ")}.`;
  }
  if (archivo.size > producto.max_bytes) {
    return `El video pesa ${megabytes(archivo.size)} y el máximo es ${megabytes(producto.max_bytes)}.`;
  }
  return null;
}

export function validarDuracion(duracion_s: number, max_minutos: number): string | null {
  if (duracion_s > max_minutos * 60) {
    return `El video dura ${decimal(duracion_s / 60, 0)} minutos y el máximo es ${decimal(max_minutos, 0)}.`;
  }
  return null;
}

/* Aviso, no bloqueo (04 §02, paso 1). */
export function avisoDeCalidad(m: Metadatos): string | null {
  if (m.alto > m.ancho || m.alto < 720) {
    return "El sistema rinde mejor con el celular horizontal y en 1080p.";
  }
  return null;
}

/*
 * "Entre X y Y min": duracion x factor medido, redondeado a minutos (D2). Reemplaza el
 * "<= 20 min" del diseno, que no salia de ninguna medicion.
 */
export function estimacion(duracion_s: number, factor: { min: number; max: number }) {
  const min = Math.max(1, Math.round((duracion_s * factor.min) / 60));
  const max = Math.max(min, Math.round((duracion_s * factor.max) / 60));
  return { min, max, texto: `Entre ${min} y ${max} min` };
}

/* "3:00" -> 180. Acepta "3" como minutos. Null si no se entiende. */
export function mmssASegundos(t: string): number | null {
  const s = t.trim();
  if (/^\d+$/.test(s)) return Number(s) * 60;
  const m = /^(\d+):([0-5]\d)$/.exec(s);
  if (!m) return null;
  return Number(m[1]) * 60 + Number(m[2]);
}

/* Lee duracion y resolucion sin decodificar el video entero. */
export function leerMetadatos(archivo: File): Promise<Metadatos> {
  return new Promise((resolver, rechazar) => {
    const url = URL.createObjectURL(archivo);
    const v = document.createElement("video");
    v.preload = "metadata";
    v.muted = true;
    const fin = () => URL.revokeObjectURL(url);
    v.onloadedmetadata = () => {
      const d = { duracion_s: v.duration, ancho: v.videoWidth, alto: v.videoHeight };
      fin();
      resolver(d);
    };
    v.onerror = () => {
      fin();
      rechazar(new Error("el navegador no pudo leer el video"));
    };
    v.src = url;
  });
}

export const CONDICIONES_FILMACION = [
  { id: "fija", texto: "Cámara fija, apoyada o en trípode" },
  { id: "cuerpo", texto: "Los dos peleadores de cuerpo entero en cuadro" },
  { id: "guantes", texto: "Guantes de distinto color en cada peleador" },
  { id: "luz", texto: "Luz de frente, no a contraluz" },
] as const;
