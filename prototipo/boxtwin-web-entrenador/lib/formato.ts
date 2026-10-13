/*
 * Como se escriben los numeros, los tiempos y los nombres en pantalla.
 *
 * Porta boxtwin-web/src/formato.js con sus pruebas sin cambiar los casos (02 §3, 06 §2), y
 * suma lo que la web nueva necesita: coma decimal es-AR, fechas cortas y bytes.
 *
 * La nomenclatura de golpes viene adentro de la Fight-Card, del servidor: la web no inventa
 * nombres de golpe, los lee (RF11, D7).
 */

export function segundosATiempo(s: number | null | undefined): string {
  if (s == null || Number.isNaN(s)) return "—";
  const total = Math.max(0, Math.floor(s));
  const m = Math.floor(total / 60);
  const seg = total % 60;
  return `${m}:${String(seg).padStart(2, "0")}`;
}

/* 05:47, con minutos de dos cifras, como el diseno en la linea de tiempo. */
export function mmss(s: number | null | undefined): string {
  if (s == null || Number.isNaN(s)) return "—";
  const total = Math.max(0, Math.floor(s));
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

/* "1 minuto 47", para lectores de pantalla (03 §8). */
export function tiempoHablado(s: number): string {
  const total = Math.max(0, Math.floor(s));
  const m = Math.floor(total / 60);
  const seg = total % 60;
  if (m === 0) return `${seg} segundos`;
  return `${m} ${m === 1 ? "minuto" : "minutos"} ${seg}`;
}

export function nombreDeTipo(
  tipo: string | null | undefined,
  nomenclatura: Record<string, string> | null | undefined,
): string | null {
  if (!tipo) return null;
  if (tipo === "no_golpe") return "no fue un golpe";
  return (nomenclatura && nomenclatura[tipo]) || tipo;
}

export function porcentaje(x: number | null | undefined, decimales = 0): string {
  if (x == null || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(decimales)}%`;
}

const formatos = new Map<number, Intl.NumberFormat>();

/* Numero con coma decimal (es-AR). */
export function decimal(x: number | null | undefined, decimales = 1): string {
  if (x == null || Number.isNaN(x)) return "—";
  let f = formatos.get(decimales);
  if (!f) {
    f = new Intl.NumberFormat("es-AR", {
      minimumFractionDigits: decimales,
      maximumFractionDigits: decimales,
    });
    formatos.set(decimales, f);
  }
  return f.format(x);
}

export function entero(x: number | null | undefined): string {
  if (x == null || Number.isNaN(x)) return "—";
  return new Intl.NumberFormat("es-AR", { maximumFractionDigits: 0 }).format(x);
}

/* Porcentaje con coma decimal y pegado, como el diseno: 0,184 -> "18,4%". */
export function porcentajeAR(x: number | null | undefined, decimales = 0): string {
  if (x == null || Number.isNaN(x)) return "—";
  return `${decimal(x * 100, decimales)}%`;
}

export function megabytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${decimal(bytes / 1024 ** 3, 2)} GB`;
  return `${decimal(bytes / 1024 ** 2, bytes >= 100 * 1024 ** 2 ? 0 : 1)} MB`;
}

/* 22/09 */
export function fechaCorta(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return `${String(d.getDate()).padStart(2, "0")}/${String(d.getMonth() + 1).padStart(2, "0")}`;
}

/* "lunes 13 de octubre" */
export function fechaLarga(d: Date): string {
  return new Intl.DateTimeFormat("es-AR", { weekday: "long", day: "numeric", month: "long" }).format(d);
}

/* "T. Ríos" */
export function nombreCorto(nombre: string): string {
  const partes = nombre.trim().split(/\s+/);
  if (partes.length < 2) return nombre.trim();
  return `${partes[0][0]}. ${partes.slice(1).join(" ")}`;
}

/* El texto del margen como lo tenia la web vieja. Se conserva para el export y sus pruebas. */
export function textoDeMargen(fc: { detector?: { recall_medido?: number; precision_medida?: number } } | null) {
  const d = (fc && fc.detector) || {};
  return (
    `Golpes detectados: el detector encuentra ~${porcentaje(d.recall_medido)} de los ` +
    `reales y acierta ~${porcentaje(d.precision_medida)}, sobre material que no vio. ` +
    `El conteo está por debajo del real; sirve para comparar adentro de la sesión.`
  );
}
