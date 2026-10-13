/*
 * POR QUE EXISTE
 *   CA-01.2 y 06 §6, paso 3: ningun numero ni nombre de demostracion del diseno puede llegar
 *   al build. El diseno del 29-09 usa "Tomás Ríos", "Gimnasio Demo" y "M. Gómez"; si alguno
 *   aparece en out/, alguien copio datos de mentira a una pantalla.
 *
 * QUE HACE
 *   Recorre out/ y sale con codigo 1 si encuentra alguno de esos nombres.
 *
 * USO
 *   npm run build && npm run sin-demo
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const raiz = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "out");
const NOMBRES = ["Tomás Ríos", "Gimnasio Demo", "M. Gómez", "Tom\\u00e1s R\\u00edos"];

function* archivos(d) {
  for (const n of readdirSync(d)) {
    const p = path.join(d, n);
    if (statSync(p).isDirectory()) yield* archivos(p);
    else if (/\.(html|js|txt|json|css)$/.test(n)) yield p;
  }
}

let hallados = 0;
for (const f of archivos(raiz)) {
  const t = readFileSync(f, "utf8");
  for (const n of NOMBRES) {
    if (t.includes(n)) {
      console.error(`${path.relative(raiz, f)}: contiene "${n}"`);
      hallados += 1;
    }
  }
}
if (hallados) process.exit(1);
console.log("out/ sin datos de demostración");
