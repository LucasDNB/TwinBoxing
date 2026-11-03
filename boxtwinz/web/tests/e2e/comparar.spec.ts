/* Z8: historial del perfil y comparacion de dos sesiones de la misma modalidad. */
import { expect, test } from "@playwright/test";
import { registrar, subir, valorar } from "./util";

test("comparar dos sesiones de bolsa", async ({ page }) => {
  await registrar(page);
  for (let i = 0; i < 2; i++) {
    await subir(page, "Bolsa");
    await expect(page.getByRole("table", { name: "Métricas de la sesión" })).toBeVisible({ timeout: 30_000 });
    await valorar(page);
  }
  await page.goto("/inicio/");
  await page.getByRole("link", { name: /Tu perfil/ }).click();
  await page.getByRole("tab", { name: "Bolsa" }).click();
  const checks = page.getByRole("checkbox", { name: /Comparar la sesión/ });
  await expect(checks).toHaveCount(2);
  await checks.nth(0).check();
  await checks.nth(1).check();
  await page.getByRole("button", { name: "Comparar dos sesiones" }).click();
  await expect(page.getByRole("table", { name: "Lado a lado" })).toBeVisible();
  await expect(page.getByText("Comparación entre rounds sin validar")).toBeVisible();
});

test("el perfil simulado se rotula", async ({ page }) => {
  await registrar(page);
  await expect(page.getByText("Perfil simulado").first()).toBeVisible();
});
