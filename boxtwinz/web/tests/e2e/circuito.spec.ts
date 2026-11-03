/* Z5 y Z7: subir un sparring, sembrar, ver el resultado; la valoracion es obligatoria. */
import { expect, test } from "@playwright/test";
import { registrar, subir, valorar } from "./util";

test("sparring: subir, sembrar, resultado y valoracion obligatoria", async ({ page }) => {
  await registrar(page);
  await subir(page, "Sparring", "2");
  await page.getByRole("link", { name: "Marcar quién es quién" }).click({ timeout: 30_000 });
  await page.getByRole("button", { name: /Persona 3/ }).click();
  await page.getByRole("button", { name: /Persona 7/ }).click();
  await page.getByRole("button", { name: "Contar golpes" }).click();

  await expect(page.getByRole("table", { name: "Métricas de la sesión" })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("Golpes detectados", { exact: true })).toBeVisible();
  await expect(page.getByText("Golpes del rival por lado")).toBeVisible();
  await expect(page.getByText("Comparación entre rounds sin validar")).toBeVisible();
  await expect(page.getByText("estimación").first()).toBeVisible();

  // Salir sin valorar abre el dialogo y no navega.
  await page.getByRole("link", { name: "Inicio" }).click();
  const dlg = page.getByRole("dialog");
  await expect(dlg).toBeVisible();
  await expect(page).toHaveURL(/\/sesion\//);
  await dlg.getByRole("button", { name: "5 estrellas" }).click();
  await dlg.getByRole("button", { name: "Enviar valoración" }).click();
  await expect(page).toHaveURL(/\/inicio\/$/);
});

test("sombra: sin siembra, rotulo de evaluacion y correccion", async ({ page }) => {
  await registrar(page);
  await subir(page, "Sombra");
  await expect(page.getByText("Reconocimiento en evaluación")).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: /Golpe de izquierda en/ }).first().click();
  await page.getByRole("button", { name: "Era la otra mano" }).click();
  await expect(page.getByRole("status")).toContainText("la corrección queda");
  await valorar(page);
  await expect(page.getByRole("link", { name: "Volver al inicio" })).toBeVisible();
});
