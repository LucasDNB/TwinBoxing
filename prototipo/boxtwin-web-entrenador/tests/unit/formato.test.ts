/*
 * Las pruebas de boxtwin-web/src/formato.test.js, portadas sin cambiar los casos (06 §2),
 * mas las de lo que agrega la web nueva.
 */
import { describe, expect, it } from "vitest";
import {
  decimal, fechaCorta, megabytes, mmss, nombreCorto, nombreDeTipo, porcentaje, porcentajeAR,
  segundosATiempo, textoDeMargen, tiempoHablado,
} from "@/lib/formato";

describe("segundosATiempo", () => {
  it("escribe minutos y segundos", () => {
    expect(segundosATiempo(0)).toBe("0:00");
    expect(segundosATiempo(75.4)).toBe("1:15");
    expect(segundosATiempo(3600)).toBe("60:00");
  });

  it("no inventa un tiempo cuando no hay dato", () => {
    expect(segundosATiempo(null)).toBe("—");
    expect(segundosATiempo(undefined)).toBe("—");
  });
});

describe("nombreDeTipo", () => {
  it("usa la nomenclatura del documento y no una propia", () => {
    const nom = { jab: "jab", cross: "gancho", hook: "hook", uppercut: "uppercut" };
    expect(nombreDeTipo("cross", nom)).toBe("gancho");
  });

  it("si el documento no trae el nombre, no traduce", () => {
    expect(nombreDeTipo("cross", {})).toBe("cross");
    expect(nombreDeTipo("cross", undefined)).toBe("cross");
  });

  it("sin tipo no muestra nada, que no es lo mismo que un tipo vacio", () => {
    expect(nombreDeTipo(null, {})).toBe(null);
  });
});

describe("textoDeMargen", () => {
  const fc = { detector: { precision_medida: 0.885, recall_medido: 0.484 } };

  it("sale del documento y no de una constante de la interfaz", () => {
    const t = textoDeMargen(fc);
    expect(t).toContain("48%");
    expect(t).toContain("89%");
  });

  it("dice DETECTADOS, que es lo que RF6 exige y no es lo mismo que lanzados", () => {
    expect(textoDeMargen(fc)).toContain("Golpes detectados");
  });

  it("dice que el conteo esta por debajo del real", () => {
    expect(textoDeMargen(fc)).toContain("por debajo del real");
  });
});

describe("porcentaje", () => {
  it("redondea sin decimales por defecto", () => {
    expect(porcentaje(0.484)).toBe("48%");
  });

  it("no inventa un cero cuando no hay dato", () => {
    expect(porcentaje(null)).toBe("—");
  });
});

describe("lo nuevo de la web entrenador", () => {
  it("coma decimal es-AR", () => {
    expect(decimal(1.375, 2)).toBe("1,38");
    expect(porcentajeAR(0.184, 1)).toBe("18,4%");
  });

  it("tiempos de linea de tiempo y para lector de pantalla", () => {
    expect(mmss(347)).toBe("05:47");
    expect(tiempoHablado(107)).toBe("1 minuto 47");
    expect(tiempoHablado(12)).toBe("12 segundos");
  });

  it("fechas y nombres cortos", () => {
    expect(fechaCorta("2026-09-21T15:00:00")).toBe("21/09");
    expect(nombreCorto("Tomás Ríos")).toBe("T. Ríos");
    expect(nombreCorto("Ríos")).toBe("Ríos");
  });

  it("megabytes", () => {
    expect(megabytes(52 * 1024 ** 2)).toBe("52,0 MB");
    expect(megabytes(3 * 1024 ** 3)).toBe("3,00 GB");
  });

  it("no fue un golpe tiene nombre propio", () => {
    expect(nombreDeTipo("no_golpe", {})).toBe("no fue un golpe");
  });
});
