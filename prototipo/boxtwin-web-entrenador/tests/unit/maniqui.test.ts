/*
 * La geometria del avatar ilustrativo (cambios-14-10, W23 y W25; DC-6, DC-9).
 */
import { describe, expect, it } from "vitest";
import { ALTURA_ESTANDAR, _interno, alcance, esqueleto, medidasDe } from "@/lib/maniqui";

describe("maniqui", () => {
  it("a escala: los dos brazos y los hombros cierran la envergadura", () => {
    const m = medidasDe(1.75, 1.8);
    expect(m.aEscala).toBe(true);
    expect(m.hombros + 2 * (m.brazo + m.antebrazo + m.punio)).toBeCloseTo(1.8, 6);
  });

  it("sin medidas: proporciones estandar y lo dice", () => {
    const m = medidasDe(null, null);
    expect(m.aEscala).toBe(false);
    expect(m.altura).toBe(ALTURA_ESTANDAR);
    expect(m.envergadura).toBe(ALTURA_ESTANDAR);
    expect(medidasDe(1.7, null).aEscala).toBe(false);
  });

  it("el alcance crece con la extension y extendido es el brazo entero", () => {
    const m = medidasDe(1.75, 1.8);
    const valores = [0, 0.25, 0.5, 0.75, 1].map((e) => alcance(m, e));
    for (let i = 1; i < valores.length; i++) expect(valores[i]).toBeGreaterThan(valores[i - 1]);
    expect(valores[4]).toBeCloseTo(m.brazo + m.antebrazo + m.punio, 6);
  });

  it("el codo cumple los dos largos en guardia y extendido", () => {
    const m = medidasDe(1.75, 1.8);
    for (const ext of [0, 0.5, 1]) {
      const e = esqueleto(m, "ortodoxa", { adelantada: ext, atrasada: 0 });
      for (const l of ["izq", "der"] as const) {
        expect(_interno.largo(_interno.resta(e.codo[l], e.hombro[l]))).toBeCloseTo(m.brazo, 3);
        expect(_interno.largo(_interno.resta(e.muneca[l], e.codo[l]))).toBeCloseTo(m.antebrazo, 3);
      }
    }
  });

  it("ortodoxo con la izquierda adelante; zurdo espejado", () => {
    const m = medidasDe(1.75, 1.8);
    const o = esqueleto(m, "ortodoxa", { adelantada: 0, atrasada: 0 });
    const z = esqueleto(m, "zurda", { adelantada: 0, atrasada: 0 });
    expect(o.tobillo.izq[0]).toBeGreaterThan(o.tobillo.der[0]);
    expect(z.tobillo.der[0]).toBeGreaterThan(z.tobillo.izq[0]);
  });

  it("extender un brazo no mueve el otro", () => {
    const m = medidasDe(1.75, 1.8);
    const a = esqueleto(m, "ortodoxa", { adelantada: 0, atrasada: 0 });
    const b = esqueleto(m, "ortodoxa", { adelantada: 1, atrasada: 0 });
    expect(b.muneca.der).toEqual(a.muneca.der);
    expect(b.muneca.izq[0]).toBeGreaterThan(a.muneca.izq[0]);
  });
});
