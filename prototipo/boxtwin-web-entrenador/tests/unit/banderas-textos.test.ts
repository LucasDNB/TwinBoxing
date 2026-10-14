import { describe, expect, it } from "vitest";
import { BLOQUES, estadoDe, type Bloque } from "@/lib/banderas";
import { deCada10, textos } from "@/lib/textos";
import type { Producto } from "@/lib/api/esquemas";

const producto = (b: Partial<Producto["banderas"]> = {}): Producto => ({
  registro: "invitacion", max_bytes: 1, max_minutos: 30, formatos: [".mp4"],
  factor_procesamiento: { min: 1.134, mediana: 1.447, max: 2.624, medido_sobre: "x" },
  criterios: { C2: { estado: "pendiente", fecha: null }, C3: { estado: "pendiente", fecha: null } },
  banderas: { comparacion_rounds: false, guardia: false, sesiones_individuales: false, postura_2d: false, ...b },
  dominio: { combinacion: { patron: "izq-der-izq", separacion: "inicio_a_inicio", max_s: 0.5 }, perfil_preliminar_sesiones: 5 },
});

describe("estadoDe: cada fila de 01 §4", () => {
  const apagado = producto();
  const encendido = producto({ comparacion_rounds: true, guardia: true, sesiones_individuales: true });

  const esperado: Record<string, [string, string]> = {
    "01.tendencia": ["apagado", "medido"],
    "04.guardia": ["apagado", "medido"],
    "05.rounds": ["apagado", "medido"],
    "05.guardia": ["apagado", "medido"],
    "06.volumen_round": ["apagado", "medido"],
    "07.frente_guardia": ["apagado", "medido"],
    "07.frente_r3": ["apagado", "medido"],
    "04.tipo": ["estimacion", "estimacion"],
    "05.distribucion": ["estimacion", "estimacion"],
    "06.combinaciones": ["medido", "medido"],
    "05.combinaciones": ["medido", "medido"],
    "05.contacto": ["no_disponible", "no_disponible"],
    "05.altura": ["no_disponible", "no_disponible"],
    "06.avatar": ["no_disponible", "no_disponible"],
    "07.cruces": ["no_disponible", "no_disponible"],
    "01.plan": ["fuera", "fuera"],
    "02.consumo": ["fuera", "fuera"],
    "05.detectados": ["medido", "medido"],
    "07.lecturas": ["en_construccion", "en_construccion"],
  };

  for (const [bloque, [sin, con]] of Object.entries(esperado)) {
    it(`${bloque}: ${sin} / ${con}`, () => {
      expect(estadoDe(bloque as Bloque, apagado)).toBe(sin);
      expect(estadoDe(bloque as Bloque, encendido)).toBe(con);
    });
  }

  it("sin /producto todavia, lo condicionado queda apagado", () => {
    expect(estadoDe("05.rounds", null)).toBe("apagado");
  });

  it("ninguna fila queda sin estado", () => {
    for (const b of Object.keys(BLOQUES) as Bloque[]) {
      expect(["medido", "estimacion", "apagado", "no_disponible", "en_construccion", "fuera"]).toContain(estadoDe(b, apagado));
    }
  });
});

describe("textos del margen (03 §6)", () => {
  it("round(valor x 10) de cada 10", () => {
    expect(deCada10(0.885)).toBe(9);
    expect(deCada10(0.484)).toBe(5);
  });

  it("el margen dice lo del ejemplo del spec con los numeros medidos", () => {
    expect(textos.margenDetector(0.885, 0.484)).toBe(
      "Acierta 9 de cada 10 que marca y se le escapa la mitad. El volumen real es mayor.",
    );
  });

  it("cambia si cambia el recall del documento (CA-05.3)", () => {
    expect(textos.margenDetector(0.885, 0.7)).toContain("se le escapa 3 de cada 10");
  });

  it("panel", () => {
    expect(textos.margenPanel(0.484)).toBe("El sistema encuentra cerca de 5 de cada 10 golpes: el total real es mayor.");
  });

  it("sin exactitud medida lo dice", () => {
    expect(textos.estimacionTipo("", null)).toContain("sin medir");
  });
});
