/* Z18: el recorrido de demo con las cuentas sembradas, cronometrado contra los 5 minutos. */
import { expect, test } from "@playwright/test";

test("recorrido de demo en menos de 5 minutos", async ({ page }) => {
  const t0 = Date.now();
  await page.goto("/");
  await page.getByLabel("Email").fill("demo-premium@boxtwinz.ar");
  await page.getByLabel("Contraseña").fill("clave-de-demo");
  await page.getByRole("button", { name: "Ingresar" }).click();
  await expect(page).toHaveURL(/\/inicio\/$/);

  await page.getByRole("link", { name: /^Sparring \d/ }).first().click();
  await expect(page.getByRole("table", { name: "Métricas de la sesión" })).toBeVisible();
  await expect(page.getByText("Golpes del rival por lado")).toBeVisible();

  await page.goto("/inicio/");
  await page.getByRole("link", { name: /Tu perfil/ }).click();
  await page.getByRole("tab", { name: "Sombra" }).click();
  await page.getByRole("checkbox", { name: /Comparar la sesión/ }).nth(0).check();
  await page.getByRole("checkbox", { name: /Comparar la sesión/ }).nth(1).check();
  await page.getByRole("button", { name: "Comparar dos sesiones" }).click();
  await expect(page.getByRole("table", { name: "Lado a lado" })).toBeVisible();

  await page.goto("/rival/");
  await page.getByLabel("Rival").selectOption({ label: "Sparring de demo" });
  await expect(page.getByText("Por dónde llegan sus golpes")).toBeVisible();

  await page.goto("/ranking/");
  await expect(page.getByRole("link", { name: /Demo Premium/ })).toBeVisible();
  await page.goto("/amigos/");
  await page.getByRole("link", { name: "Demo Free" }).click();
  await expect(page.getByRole("tablist", { name: "Modalidad" })).toBeVisible();
  await page.goto("/ajustes/");
  await expect(page.getByText("Pago simulado — versión de demostración")).toBeVisible();

  expect(Date.now() - t0).toBeLessThan(5 * 60_000);
});
