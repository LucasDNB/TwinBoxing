/*
 * POR QUE EXISTE
 *   03 §9 fija un presupuesto: JavaScript inicial por ruta <= 380 KB comprimido, para un
 *   celular de gama media en 4G lenta. Se mide sobre el build exportado, que es lo que baja
 *   el celular, y no sobre una estimacion del bundler.
 *
 * QUE HACE
 *   Para cada pagina de out/, suma el tamano gzip de los <script src> que el HTML pide al
 *   cargar y sale con codigo 1 si alguna ruta pasa el limite.
 *
 * USO
 *   npm run build && node scripts/presupuesto.mjs
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { gzipSync } from "node:zlib";
import path from "node:path";
import { fileURLToPath } from "node:url";

const OUT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "out");
// 03 §9, modificado el 13-10: de 200 a 380 KB (el piso medido del stack es 192 KB).
const LIMITE_KB = 380;

function* paginas(d) {
  for (const n of readdirSync(d)) {
    const p = path.join(d, n);
    if (statSync(p).isDirectory()) {
      if (n !== "_next") yield* paginas(p);
    } else if (n === "index.html") yield p;
  }
}

const gz = new Map();
function kb(src) {
  if (!gz.has(src)) gz.set(src, gzipSync(readFileSync(path.join(OUT, src))).length / 1024);
  return gz.get(src);
}

let mal = 0;
for (const html of paginas(OUT)) {
  const t = readFileSync(html, "utf8");
  const srcs = [...new Set([...t.matchAll(/<script[^>]+src="\/([^"]+\.js)"/g)].map((m) => m[1]))];
  const total = srcs.reduce((a, s) => a + kb(s), 0);
  const ruta = "/" + path.relative(OUT, path.dirname(html)).replace(/\\/g, "/");
  const ok = total <= LIMITE_KB;
  if (!ok) mal += 1;
  console.log(`${ok ? "ok " : "MAL"} ${total.toFixed(1).padStart(6)} KB  ${ruta === "/" ? "/" : ruta + "/"}`);
}
if (mal) {
  console.error(`${mal} ruta(s) pasan los ${LIMITE_KB} KB`);
  process.exit(1);
}
