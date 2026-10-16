/*
 * Una Fight-Card del tipo v2 armada sobre la 0.3 real: los campos opcionales de
 * docs/specs/2026-10-16-fightcard-tipo-v2 (metodo, referencia, guardia_supuesta, p_familia, rol).
 */
import fc03 from "./sesion-real/fightcard-0.3.json";

export function fcV2(supuesta: { A?: boolean; B?: boolean } = { A: false, B: true }) {
  const fc = structuredClone(fc03) as Record<string, any>;
  fc.clasificador = {
    ...fc.clasificador, checkpoint: "clasificador_geo_v1", exactitud_familia_fuente_no_vista: 0.7268,
    metodo: "geometrico-v1", referencia: "docs/experiments/2026-10-16-clasificador-geometrico.md",
    guardia_supuesta: supuesta,
  };
  for (const p of Object.values(fc.peleadores) as any[]) {
    for (const g of p.golpes) {
      g.p_familia = [0.7, 0.2, 0.1];
      g.rol = g.brazo === "izq" ? "lead" : "rear";
    }
  }
  return fc;
}
