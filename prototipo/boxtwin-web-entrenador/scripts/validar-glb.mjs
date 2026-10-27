/*
 * POR QUE EXISTE
 *   El avatar del perfil carga public/models/boxer_mpfb2.glb, exportado a mano desde Blender
 *   con MPFB2. Si el export cambia un nombre de hueso o se olvida de decimar, la web no falla:
 *   cae al maniqui de primitivas o le baja 20 MB al celular sin que nadie se entere. Esto
 *   revisa el contrato antes de copiar el archivo.
 *
 * QUE HACE
 *   Lee el chunk JSON del GLB (sin dependencias, sin decodificar geometria) y revisa:
 *   las dos mallas por nombre, el skin con los huesos que usa lib/avatar/solverRig.ts, los
 *   canales PBR del cuerpo, los triangulos y el peso. Errores bloquean; avisos no.
 *
 * USO
 *   node scripts/validar-glb.mjs public/models/boxer_mpfb2.glb
 */
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Los mismos que HUESOS en lib/avatar/solverRig.ts (la prueba compara las dos listas).
export const HUESOS = [
  "pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "head",
  "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
  "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
  "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r",
];
export const CUERPO = "Boxer_Body_Mesh";
export const GUANTES = "Boxer_Gloves_Mesh";
export const TRIANGULOS = { min: 15000, max: 35000, tope: 60000 };
// En 4G lenta, 6 MB son unos 15 s: el avatar aparece tarde pero aparece.
export const MEGAS = { aviso: 6, tope: 16 };

export function leerJson(buf) {
  if (buf.length < 20 || buf.readUInt32LE(0) !== 0x46546c67) throw new Error("no es un GLB (magic)");
  if (buf.readUInt32LE(4) !== 2) throw new Error(`GLB version ${buf.readUInt32LE(4)}, se espera 2`);
  const largo = buf.readUInt32LE(12);
  if (buf.readUInt32LE(16) !== 0x4e4f534a) throw new Error("el primer chunk no es JSON");
  return JSON.parse(buf.subarray(20, 20 + largo).toString("utf8"));
}

const coincide = (nombre, buscado) => {
  const n = (nombre ?? "").replace(/\./g, "_");
  return n === buscado || n.endsWith(`_${buscado}`) || n.endsWith(`:${buscado}`);
};

export function validarGlb(buf) {
  const errores = [], avisos = [];
  let g;
  try {
    g = leerJson(buf);
  } catch (e) {
    return { errores: [e.message], avisos, triangulos: 0, megas: buf.length / 2 ** 20 };
  }
  const nodos = g.nodes ?? [], mallas = g.meshes ?? [], accesores = g.accessors ?? [];

  // Una malla se encuentra por el nombre del nodo o por el de la malla.
  const mallaDe = (nombre) => {
    const n = nodos.find((x) => x.mesh != null && (x.name === nombre || mallas[x.mesh]?.name === nombre));
    return n ? { nodo: n, malla: mallas[n.mesh] } : null;
  };
  const cuerpo = mallaDe(CUERPO), guantes = mallaDe(GUANTES);
  if (!cuerpo) errores.push(`falta la malla ${CUERPO}`);
  if (!guantes) avisos.push(`falta la malla ${GUANTES}: los guantes no toman el color del lado`);
  if (cuerpo && cuerpo.nodo.skin == null) errores.push(`${CUERPO} no tiene skin: no se puede posar`);

  const articulaciones = new Set((g.skins ?? []).flatMap((s) => s.joints.map((j) => nodos[j]?.name)));
  const faltan = HUESOS.filter((h) => ![...articulaciones].some((n) => coincide(n, h)));
  if (!g.skins?.length) errores.push("el GLB no trae skins");
  else if (faltan.length) errores.push(`faltan huesos en el skin: ${faltan.join(", ")}`);

  if (cuerpo) {
    const mat = g.materials?.[cuerpo.malla.primitives?.[0]?.material];
    const pbr = mat?.pbrMetallicRoughness ?? {};
    if (!pbr.baseColorTexture) avisos.push("el cuerpo no trae BaseColor");
    if (!mat?.normalTexture) avisos.push("el cuerpo no trae NormalMap");
    if (!pbr.metallicRoughnessTexture) avisos.push("el cuerpo no trae mapa de rugosidad");
  }

  let triangulos = 0;
  for (const m of mallas) {
    for (const p of m.primitives ?? []) {
      if ((p.mode ?? 4) !== 4) continue;
      const a = accesores[p.indices ?? p.attributes?.POSITION];
      triangulos += Math.floor((a?.count ?? 0) / 3);
    }
  }
  if (triangulos > TRIANGULOS.tope) errores.push(`${triangulos} triangulos, tope ${TRIANGULOS.tope}: decimar`);
  else if (triangulos < TRIANGULOS.min || triangulos > TRIANGULOS.max) {
    avisos.push(`${triangulos} triangulos, fuera de ${TRIANGULOS.min}-${TRIANGULOS.max}`);
  }

  const megas = buf.length / 2 ** 20;
  if (megas > MEGAS.tope) errores.push(`${megas.toFixed(1)} MB, tope ${MEGAS.tope}`);
  else if (megas > MEGAS.aviso) avisos.push(`${megas.toFixed(1)} MB: pesado para el celular (aviso desde ${MEGAS.aviso})`);

  return { errores, avisos, triangulos, megas };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  const archivo = process.argv[2] ?? path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "public", "models", "boxer_mpfb2.glb");
  const r = validarGlb(readFileSync(archivo));
  console.log(`${archivo}: ${r.triangulos} triangulos, ${r.megas.toFixed(2)} MB`);
  for (const a of r.avisos) console.log(`aviso  ${a}`);
  for (const e of r.errores) console.error(`ERROR  ${e}`);
  process.exit(r.errores.length ? 1 : 0);
}
