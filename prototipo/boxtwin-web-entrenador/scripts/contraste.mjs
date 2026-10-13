/*
 * POR QUE EXISTE
 *   03 §2 pide que ningun par texto/fondo usado quede bajo 4,5:1 (3:1 para texto de 24 px o
 *   mas). Los tokens se leen de app/globals.css, asi que si alguien cambia un color el
 *   chequeo mide el color nuevo y no una copia.
 *
 * QUE HACE
 *   Lee los tokens de @theme, calcula el contraste WCAG 2.1 de cada par de PARES y sale con
 *   codigo 1 si alguno no llega a su minimo.
 *
 * USO
 *   npm run contraste
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const aqui = path.dirname(fileURLToPath(import.meta.url));

export function leerTokens(css) {
  const tokens = {};
  for (const m of css.matchAll(/--color-([a-z0-9-]+):\s*(#[0-9A-Fa-f]{6})/g)) {
    tokens[m[1]] = m[2];
  }
  return tokens;
}

function lineal(c) {
  const v = c / 255;
  return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
}

export function luminancia(hex) {
  const n = parseInt(hex.slice(1), 16);
  return 0.2126 * lineal((n >> 16) & 255) + 0.7152 * lineal((n >> 8) & 255) + 0.0722 * lineal(n & 255);
}

export function contraste(a, b) {
  const [x, y] = [luminancia(a), luminancia(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
}

// [texto, fondo, minimo, donde se usa]
export const PARES = [
  ...["texto", "texto-2", "texto-3", "tenue"].flatMap((t) =>
    ["fondo", "nav", "superficie", "superficie-2", "elevado"].map((f) => [t, f, 4.5, "texto general"]),
  ),
  ["fondo", "texto", 4.5, "boton primario"],
  ["estimado", "estimado-fondo", 4.5, "bloque de estimacion"],
  ["estimado", "superficie", 4.5, "numero estimado sobre tarjeta"],
  ["estimado", "fondo", 4.5, "numero estimado sobre fondo"],
  ["ok", "ok-fondo", 4.5, "corregido / listo"],
  ["ok", "superficie", 4.5, "tipo corregido sobre tarjeta"],
  ["curso", "curso-fondo", 4.5, "en curso"],
  ["error", "error-fondo", 4.5, "fallo"],
  ["error", "superficie", 4.5, "error sobre tarjeta"],
  ["a-trazo", "superficie", 3, "marca A (grafico, 24 px o mas)"],
  ["b-trazo", "superficie", 3, "marca B (grafico, 24 px o mas)"],
  // La letra de la marca A/B va en el color de trazo sobre fondo oscuro, no en claro sobre
  // el relleno: texto sobre --color-a da 4,11:1 y sobre --color-b 3,92:1, bajo el minimo.
  // Medido en W1 el 13-10; por eso la Esquina dibuja contorno y letra, no relleno. Tampoco
  // va sobre --color-elevado: la A da 4,18:1 ahi.
  ...["fondo", "superficie", "superficie-2"].flatMap((f) => [
    ["a-trazo", f, 4.5, "letra A de la marca"],
    ["b-trazo", f, 4.5, "letra B de la marca"],
  ]),
];

export function revisar(tokens) {
  return PARES.map(([t, f, minimo, uso]) => {
    const r = contraste(tokens[t], tokens[f]);
    return { texto: t, fondo: f, minimo, uso, razon: Math.round(r * 100) / 100, ok: r >= minimo };
  });
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  const css = readFileSync(path.join(aqui, "..", "app", "globals.css"), "utf8");
  const filas = revisar(leerTokens(css));
  for (const f of filas) {
    console.log(`${f.ok ? "ok " : "MAL"} ${f.razon.toFixed(2).padStart(5)}:1  ${f.texto} / ${f.fondo}  (min ${f.minimo}, ${f.uso})`);
  }
  const malos = filas.filter((f) => !f.ok);
  if (malos.length) {
    console.error(`${malos.length} par(es) bajo el minimo`);
    process.exit(1);
  }
}
