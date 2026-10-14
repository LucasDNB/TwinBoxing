import { describe, expect, it } from "vitest";
import fc01 from "../fixtures/sesion-real/fightcard-0.1.json";
import fc02 from "../fixtures/sesion-real/fightcard-0.2.json";
import fc03 from "../fixtures/sesion-real/fightcard-0.3.json";
import { FightCard as Esquema } from "@/lib/api/esquemas";
import {
  combinacionesDe, golpesEnCombinacion, guardiaPorGolpe, lecturaRounds, normalizarFightCard,
  recallDeCombinaciones, resumen, sumaDeRounds,
} from "@/lib/fightcard";

const cargar = (d: unknown) => normalizarFightCard(Esquema.parse(d));

describe("contrato 0.1 y 0.2", () => {
  it("las dos versiones validan", () => {
    expect(Esquema.safeParse(fc01).success).toBe(true);
    expect(Esquema.safeParse(fc02).success).toBe(true);
  });

  it("una version desconocida no valida", () => {
    expect(Esquema.safeParse({ ...fc01, version: "0.4" }).success).toBe(false);
  });

  it("a una 0.1 se le completan descartado, descartados y los campos de tramos", () => {
    const fc = cargar(fc01);
    expect(fc.sinRevisionDeTramos).toBe(true);
    expect(fc.peleadores.A.descartados).toBe(0);
    expect(fc.peleadores.A.golpes.every((g) => g.descartado === false)).toBe(true);
    expect(fc.identidad.tramos_total).toBeNull();
    expect(fc.identidad.separacion_color).toBe(0.354);
  });
});

describe("resumen por sesion y por round (CA-05.1)", () => {
  it("la suma de los rounds es el total del documento", () => {
    const fc = cargar(fc01);
    for (const l of ["A", "B"] as const) {
      const p = fc.peleadores[l]!;
      expect(sumaDeRounds(p)).toBe(p.total.total);
      expect(resumen(fc, l)!.detectados).toBe(p.total.total);
    }
  });

  it("no fue un golpe no cambia los detectados y suma descartados (B4)", () => {
    const fc = cargar(fc02);
    const r = resumen(fc, "A")!;
    expect(r.detectados).toBe(51);
    expect(r.descartados).toBe(1);
    const enTipos = Object.values(r.tipos).reduce((a, b) => a + b, 0);
    expect(enTipos + r.sinClasificar).toBe(50);
  });

  it("sin clasificador todo queda sin clasificar", () => {
    const r = resumen(cargar(fc01), "B")!;
    expect(r.sinClasificar).toBe(32);
    expect(r.tipos.jab).toBe(0);
  });

  it("un round que no existe no inventa un resumen", () => {
    expect(resumen(cargar(fc01), "A", 7)).toBeNull();
  });

  it("los golpes por minuto salen de la duracion", () => {
    expect(resumen(cargar(fc01), "A")!.porMinuto).toBeCloseTo(17.49, 1);
  });
});

describe("lectura por round", () => {
  const r = (round: number, total: number) => ({ round, inicio_s: 0, fin_s: 1, total, izq: 0, der: 0, por_minuto: 0 });
  it("dice la caida en una frase", () => {
    expect(lecturaRounds([r(1, 50), r(2, 44), r(3, 31)], "Juan Pérez")).toBe("De R2 a R3, J. Pérez cae de 44 a 31, −30%");
  });
  it("con un round no hay lectura", () => {
    expect(lecturaRounds([r(1, 50)], "X")).toBeNull();
  });
});

describe("guardia por golpe", () => {
  it("cruza el id de la guardia (brazo en ingles) con el del golpe", () => {
    const fc = cargar(fc01);
    const m = guardiaPorGolpe(fc.peleadores.A);
    expect(m.get("A-izq-75")?.mano_opuesta_caida).toBe(true);
  });
});

describe("combinaciones izq-der-izq (W13)", () => {
  it("se leen del documento 0.3, por sesion y por round", () => {
    const fc = cargar(fc03);
    expect(combinacionesDe(fc, "A")).toBe(3);
    expect(combinacionesDe(fc, "B")).toBe(1);
    expect(combinacionesDe(fc, "A", 1)).toBe(3);
    expect(golpesEnCombinacion(fc, "A").size).toBe(9);
  });

  it("una 0.2 no las trae y la web no las calcula", () => {
    const fc = cargar(fc02);
    expect(combinacionesDe(fc, "A")).toBeNull();
    expect(golpesEnCombinacion(fc, "A").size).toBe(0);
  });

  it("el margen usa el recall de ternas cuando C11 esta medido", () => {
    const fc = cargar(fc03);
    expect(recallDeCombinaciones(fc)).toBe(fc.detector.recall_medido);
    const conC11 = cargar({ ...fc03, detector: { ...fc03.detector, combinaciones: { recall_medido: 0.2 } } });
    expect(recallDeCombinaciones(conC11)).toBe(0.2);
  });
});
