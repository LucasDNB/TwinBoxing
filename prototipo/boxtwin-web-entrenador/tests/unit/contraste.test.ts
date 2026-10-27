import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
// @ts-expect-error modulo js sin tipos
import { contraste, leerTokens, revisar } from "../../scripts/contraste.mjs";

describe("contraste de los tokens (03 §2)", () => {
  const css = readFileSync(path.resolve(__dirname, "../../app/globals.css"), "utf8");
  const tokens = leerTokens(css);

  it("calcula el contraste WCAG conocido", () => {
    expect(contraste("#000000", "#FFFFFF")).toBeCloseTo(21, 1);
  });

  it("el rotulo tenue da 7,1:1 sobre superficie, como dice el spec", () => {
    expect(contraste(tokens["tenue"], tokens["superficie"])).toBeGreaterThanOrEqual(7.0);
  });

  it("ningun par usado queda bajo su minimo", () => {
    const malos = revisar(tokens).filter((f: { ok: boolean }) => !f.ok);
    expect(malos).toEqual([]);
  });
});
