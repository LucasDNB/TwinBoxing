import { describe, expect, it } from "vitest";
import fc01 from "../fixtures/sesion-real/fightcard-0.1.json";
import fc02 from "../fixtures/sesion-real/fightcard-0.2.json";
import { FightCard as Esquema, type Golpe } from "@/lib/api/esquemas";
import {
  combinaciones, guardiaPorGolpe, ladoDelBrazo, lecturaRounds, nombreDeCombinacion,
  normalizarFightCard, numeroDeGimnasio, resumen, sumaDeRounds,
} from "@/lib/fightcard";

const cargar = (d: unknown) => normalizarFightCard(Esquema.parse(d));

describe("contrato 0.1 y 0.2", () => {
  it("las dos versiones validan", () => {
    expect(Esquema.safeParse(fc01).success).toBe(true);
    expect(Esquema.safeParse(fc02).success).toBe(true);
  });

  it("una version desconocida no valida", () => {
    expect(Esquema.safeParse({ ...fc01, version: "0.3" }).success).toBe(false);
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

describe("combinaciones", () => {
  const g = (t: number, tipo: string | null, brazo = "izq", extra: Partial<Golpe> = {}): Golpe => ({
    id: `A-${brazo}-${t}`, t_inicio: t, t_fin: t + 0.2, brazo, tipo, confianza_tipo: 0.8,
    corregido: null, descartado: false, ...extra,
  });

  it("lado desde guardia y brazo; sin guardia no se inventa", () => {
    expect(ladoDelBrazo("izq", "ortodoxa")).toBe("adelante");
    expect(ladoDelBrazo("der", "ortodoxa")).toBe("atras");
    expect(ladoDelBrazo("izq", "zurda")).toBe("atras");
    expect(ladoDelBrazo("izq", null)).toBeNull();
  });

  it("numeracion de gimnasio", () => {
    expect(numeroDeGimnasio("jab", "adelante")).toBe(1);
    expect(numeroDeGimnasio("cross", "atras")).toBe(2);
    expect(numeroDeGimnasio("hook", "adelante")).toBe(3);
    expect(numeroDeGimnasio("hook", "atras")).toBe(4);
    expect(numeroDeGimnasio("uppercut", "atras")).toBe(6);
    expect(numeroDeGimnasio("hook", null)).toBeNull();
  });

  it("respeta la pausa, cuenta corregidos con su tipo y saca descartados", () => {
    const golpes = [
      g(1, "jab"), g(1.4, "cross", "der"),           // 1-2 (pausa 0,2)
      g(5, "jab"), g(5.5, "hook", "der", { corregido: { tipo: "cross" } }), // 1-2 corregido
      g(9, "jab"), g(9.9, "cross", "der"),          // pausa 0,7: no es combinacion
      g(12, "jab"), g(12.3, "cross", "der", { descartado: true }), // descartado no cuenta
    ];
    const c = combinaciones(golpes, 0.5, "ortodoxa");
    expect(c[0].n).toBe(2);
    expect(nombreDeCombinacion(c[0], {})).toBe("1-2");
  });

  it("sin guardia se nombra con la familia", () => {
    const c = combinaciones([g(1, "jab"), g(1.3, "cross", "der")], 0.5, null);
    expect(nombreDeCombinacion(c[0], { jab: "jab", cross: "gancho" })).toBe("jab + gancho");
  });
});
