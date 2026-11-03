/* Formatos de numeros en es-AR: coma decimal, porcentajes sin decimales. */
export const num = (v: number | null | undefined, d = 1) =>
  v == null ? "—" : v.toLocaleString("es-AR", { minimumFractionDigits: d, maximumFractionDigits: d });
export const pct = (v: number | null | undefined) => (v == null ? "—" : `${Math.round(v * 100)}%`);
export const tiempo = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
