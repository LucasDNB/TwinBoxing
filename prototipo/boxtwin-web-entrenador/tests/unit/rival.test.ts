import { describe, expect, it } from "vitest";
import { hayLecturas, lecturas, type FilaRival } from "@/lib/rival";

const base: FilaRival = {
  nombre: "x", altura_m: 1.75, envergadura_m: 1.8, guardia: "ortodoxa", golpes_por_minuto: 12,
  jab_fraccion: 0.4, guardia_opuesta_arriba: 0.6, caida_ultimo_round: -0.05,
};
const apagadas = { comparacion_rounds: false, guardia: false };
const encendidas = { comparacion_rounds: true, guardia: true };

describe("reglas del Rival Fantasma (04 §07)", () => {
  it("envergadura de 3 cm o mas: oportunidad para el de mayor envergadura", () => {
    const l = lecturas({ ...base, envergadura_m: 1.85 }, base, apagadas);
    expect(l.A.oportunidades[0]).toEqual({ texto: "Tiene 5 cm más de envergadura: puede trabajar a distancia con el jab.", metrica: "envergadura" });
    expect(lecturas({ ...base, envergadura_m: 1.82 }, base, apagadas).A.oportunidades).toHaveLength(0);
  });

  it("ortodoxo contra zurdo: amenaza para los dos", () => {
    const l = lecturas(base, { ...base, guardia: "zurda" }, apagadas);
    expect(l.A.amenazas[0].metrica).toBe("guardia");
    expect(l.B.amenazas[0].metrica).toBe("guardia");
  });

  it("ritmo de 15% o mas: oportunidad para el de mayor ritmo", () => {
    const l = lecturas(base, { ...base, golpes_por_minuto: 14 }, apagadas);
    expect(l.B.oportunidades[0].metrica).toBe("golpes por minuto");
    expect(lecturas(base, { ...base, golpes_por_minuto: 13 }, apagadas).B.oportunidades).toHaveLength(0);
  });

  it("CA-07.1: con C2 y C3 apagados ninguna lectura cita caida en R3 ni guardia opuesta", () => {
    const a = { ...base, caida_ultimo_round: -0.4, guardia_opuesta_arriba: 0.2 };
    const l = lecturas(a, base, apagadas);
    const todas = [...l.A.oportunidades, ...l.A.amenazas, ...l.B.oportunidades, ...l.B.amenazas];
    expect(todas.some((x) => /último round|otra mano/.test(x.texto))).toBe(false);
  });

  it("con C2 y C3, caida y guardia opuesta entran", () => {
    const a = { ...base, caida_ultimo_round: -0.2, guardia_opuesta_arriba: 0.45 };
    const l = lecturas(a, base, encendidas);
    expect(l.A.amenazas.map((x) => x.metrica)).toEqual(["caída en el último round", "guardia opuesta arriba"]);
    expect(l.B.oportunidades[0].texto).toBe("El rival cae 20% en el último round: el cierre es suyo.");
  });

  it("sin diferencias marcadas no hay lecturas", () => {
    expect(hayLecturas(lecturas(base, base, encendidas))).toBe(false);
  });
});
