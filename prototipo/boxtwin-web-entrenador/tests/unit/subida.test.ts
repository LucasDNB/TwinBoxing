import { describe, expect, it } from "vitest";
import { avisoDeCalidad, estimacion, mmssASegundos, validarArchivo, validarDuracion } from "@/lib/subida";

const producto = { formatos: [".mp4", ".mov", ".m4v"], max_bytes: 6 * 1024 ** 3 };

describe("validacion antes de subir (CA-02.1)", () => {
  it("un .avi se rechaza con el motivo", () => {
    expect(validarArchivo({ name: "spar.avi", size: 10 }, producto)).toContain("El formato .avi no se acepta");
  });
  it("un archivo mayor que max_bytes se rechaza con el motivo", () => {
    expect(validarArchivo({ name: "spar.mp4", size: 7 * 1024 ** 3 }, producto)).toContain("el máximo es 6,00 GB");
  });
  it("mayusculas en la extension valen", () => {
    expect(validarArchivo({ name: "SPAR.MOV", size: 10 }, producto)).toBeNull();
  });
  it("duracion maxima", () => {
    expect(validarDuracion(31 * 60, 30)).toContain("el máximo es 30");
    expect(validarDuracion(29 * 60, 30)).toBeNull();
  });
});

describe("estimacion de tiempo (D2)", () => {
  it("duracion por el factor medido, redondeado a minutos", () => {
    // 10 min de video: 11,34 a 26,24 min
    expect(estimacion(600, { min: 1.134, max: 2.624 })).toEqual({ min: 11, max: 26, texto: "Entre 11 y 26 min" });
  });
  it("un video corto no da cero minutos", () => {
    expect(estimacion(10, { min: 1.134, max: 2.624 }).min).toBe(1);
  });
});

describe("calidad y tiempos", () => {
  it("vertical o bajo 720 avisa", () => {
    expect(avisoDeCalidad({ duracion_s: 1, ancho: 1080, alto: 1920 })).not.toBeNull();
    expect(avisoDeCalidad({ duracion_s: 1, ancho: 1280, alto: 640 })).not.toBeNull();
    expect(avisoDeCalidad({ duracion_s: 1, ancho: 1920, alto: 1080 })).toBeNull();
  });
  it("mm:ss", () => {
    expect(mmssASegundos("3:00")).toBe(180);
    expect(mmssASegundos("1:30")).toBe(90);
    expect(mmssASegundos("2")).toBe(120);
    expect(mmssASegundos("1:75")).toBeNull();
  });
});
