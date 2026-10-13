/*
 * El contrato de la Fight-Card del lado de la web (06 §3). El mismo JSON Schema que valida
 * lo que produce el worker (boxtwin-annotator/tests/test_mvp_fightcard_contrato.py) valida
 * aca los documentos de una sesion real, y zod tiene que aceptar exactamente lo mismo.
 */
import { describe, expect, it } from "vitest";
import Ajv2020 from "ajv/dist/2020";
import schema from "../../../contrato/fightcard-0.3.schema.json";
import fc01 from "../fixtures/sesion-real/fightcard-0.1.json";
import fc02 from "../fixtures/sesion-real/fightcard-0.2.json";
import fc03 from "../fixtures/sesion-real/fightcard-0.3.json";
import { FightCard } from "@/lib/api/esquemas";

const ajv = new Ajv2020({ allErrors: true, strict: false });
const validar = ajv.compile(schema);

describe("contrato 0.3 compartido con el worker", () => {
  for (const [nombre, doc] of [["0.1", fc01], ["0.2", fc02], ["0.3", fc03]] as const) {
    it(`la Fight-Card ${nombre} real cumple el schema y zod la acepta`, () => {
      expect(validar(doc), JSON.stringify(validar.errors)).toBe(true);
      expect(FightCard.safeParse(doc).success).toBe(true);
    });
  }

  it("una version desconocida falla en los dos", () => {
    const d = { ...fc03, version: "0.4" };
    expect(validar(d)).toBe(false);
    expect(FightCard.safeParse(d).success).toBe(false);
  });

  it("una 0.3 sin combinaciones falla en los dos", () => {
    const d = { ...fc02, version: "0.3" };
    expect(validar(d)).toBe(false);
    expect(FightCard.safeParse(d).success).toBe(false);
  });

  it("un campo obligatorio faltante falla en los dos", () => {
    const d = structuredClone(fc02) as Record<string, unknown>;
    delete d.detector;
    expect(validar(d)).toBe(false);
    expect(FightCard.safeParse(d).success).toBe(false);
  });

  it("un documento con conectados no cumple el contrato", () => {
    expect(validar({ ...fc02, conectados: 4 })).toBe(false);
  });
});
