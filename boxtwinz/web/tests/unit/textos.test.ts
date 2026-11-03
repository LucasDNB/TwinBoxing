/* Los rotulos obligatorios (spec 12) son los mismos en la web y en la API (textos.py). */
import { readFileSync } from "node:fs";
import path from "node:path";
import { TEXTOS } from "@/lib/textos";

const py = readFileSync(path.resolve(__dirname, "../../../api/boxtwinz_api/textos.py"), "utf8").replace(/"\s*\n\s*"/g, "");

describe("textos obligatorios", () => {
  it.each([
    ["perfilSimulado", "Perfil simulado"],
    ["pagoSimulado", "Pago simulado — versión de demostración"],
    ["roundsSinValidar", "Comparación entre rounds sin validar"],
    ["enEvaluacion", "Reconocimiento en evaluación"],
  ] as const)("%s", (clave, texto) => {
    expect(TEXTOS[clave]).toBe(texto);
    expect(py).toContain(texto);
  });

  it("consentimientos iguales en web y API", () => {
    expect(py).toContain(TEXTOS.consentimientoUso);
    expect(py).toContain(TEXTOS.consentimientoReent);
  });

  it("dice golpes detectados, nunca lanzados", () => {
    expect(TEXTOS.golpes).toBe("golpes detectados");
  });
});
