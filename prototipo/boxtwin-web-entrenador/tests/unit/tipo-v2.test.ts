/*
 * Tipo de golpe v2 (docs/specs/2026-10-16-fightcard-tipo-v2, 02-diseno.md §5): el contrato
 * admite los campos nuevos como opcionales, en el schema compartido y en zod, y el aviso de
 * guardia supuesta sale solo para los lados con golpes.
 */
import { describe, expect, it } from "vitest";
import Ajv2020 from "ajv/dist/2020";
import schema from "../../../contrato/fightcard-0.3.schema.json";
import fc03 from "../fixtures/sesion-real/fightcard-0.3.json";
import { fcV2 } from "../fixtures/fc-v2";
import { FightCard } from "@/lib/api/esquemas";
import { ladosConGuardiaSupuesta, normalizarFightCard } from "@/lib/fightcard";
import { textos } from "@/lib/textos";

const validar = new Ajv2020({ allErrors: true, strict: false }).compile(schema);

describe("tipo v2 en el contrato", () => {
  it("una Fight-Card v2 cumple el schema y zod la acepta, con los campos", () => {
    const d = fcV2();
    expect(validar(d), JSON.stringify(validar.errors)).toBe(true);
    const r = FightCard.safeParse(d);
    expect(r.success).toBe(true);
    if (r.success) {
      expect(r.data.clasificador.metodo).toBe("geometrico-v1");
      expect(r.data.clasificador.guardia_supuesta).toEqual({ A: false, B: true });
      expect(r.data.peleadores.A.golpes[0].p_familia).toHaveLength(3);
    }
  });

  it("una v1 (sin los campos) sigue valiendo en los dos", () => {
    expect(validar(fc03)).toBe(true);
    expect(FightCard.safeParse(fc03).success).toBe(true);
  });

  it("un metodo desconocido o un rol fuera del enum fallan en los dos", () => {
    const a = fcV2();
    a.clasificador.metodo = "otro";
    expect(validar(a)).toBe(false);
    expect(FightCard.safeParse(a).success).toBe(false);
    const b = fcV2();
    b.peleadores.A.golpes[0].rol = "adelantado";
    expect(validar(b)).toBe(false);
    expect(FightCard.safeParse(b).success).toBe(false);
  });
});

describe("aviso de guardia supuesta", () => {
  const norm = (d: unknown) => normalizarFightCard(FightCard.parse(d));

  it("sale para los lados supuestos con golpes", () => {
    expect(ladosConGuardiaSupuesta(norm(fcV2({ A: false, B: true })))).toEqual(["B"]);
    expect(ladosConGuardiaSupuesta(norm(fcV2({ A: true, B: true })))).toEqual(["A", "B"]);
    expect(ladosConGuardiaSupuesta(norm(fcV2({ A: false, B: false })))).toEqual([]);
  });

  it("no sale en una v1 ni para un lado sin golpes", () => {
    expect(ladosConGuardiaSupuesta(norm(fc03))).toEqual([]);
    const d = fcV2({ A: true, B: true });
    d.peleadores.B.golpes = [];
    expect(ladosConGuardiaSupuesta(norm(d))).toEqual(["A"]);
  });

  it("el texto es el del requisito", () => {
    expect(textos.avisoGuardiaSupuesta("Mono")).toBe(
      "Jab y cross suponen guardia ortodoxa. Cargá la guardia de Mono para corregirlo.");
  });
});

/*
 * Modo académico E7 (docs/specs/2026-10-25-e7-modo-academico, M5): el método nuevo y su
 * leyenda de licencia valen en el schema y en zod, y la nota del tipo lo dice.
 */
describe("modo académico E7", () => {
  const LIC = "modo academico: 3D de 4D-Humans con SMPL, licencia de investigacion no comercial";

  it("geometrico-e7 con licencia cumple el schema y zod la acepta", () => {
    const d = fcV2();
    d.clasificador.metodo = "geometrico-e7";
    (d.clasificador as Record<string, unknown>).licencia = LIC;
    expect(validar(d), JSON.stringify(validar.errors)).toBe(true);
    const r = FightCard.safeParse(d);
    expect(r.success).toBe(true);
    if (r.success) expect(r.data.clasificador.licencia).toBe(LIC);
  });

  it("la nota del modo académico nombra la licencia y E4 no lleva nota", () => {
    expect(textos.modoAcademico("geometrico-e7", LIC)).toContain("4D-Humans");
    expect(textos.modoAcademico("geometrico-e7", null)).toContain("modo académico");
    expect(textos.modoAcademico("geometrico-v1", LIC)).toBe("");
    expect(textos.modoAcademico(undefined, undefined)).toBe("");
  });
});
