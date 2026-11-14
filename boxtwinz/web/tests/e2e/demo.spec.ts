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

test("probar demo: contraseña, selector, crear usuario rápido y amigos", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Probar demo" }).click();
  await page.getByLabel("Contraseña de la demo").fill("otra-clave");
  await page.getByRole("button", { name: "Entrar a la demo" }).click();
  await expect(page.getByText("La contraseña de la demo no es correcta")).toBeVisible();
  await page.getByLabel("Contraseña de la demo").fill("USAL@2026");
  await page.getByRole("button", { name: "Entrar a la demo" }).click();
  await expect(page).toHaveURL(/\/demo\/$/);

  const estandar = page.getByRole("list", { name: "Usuarios estándar" });
  await expect(estandar.getByRole("button")).toHaveText([/Fajador/, /Estilista/, /Zurdo/]);
  await expect(estandar.getByRole("button", { name: "Entrar como Zurdo" })).toContainText("No definido · Zurda");

  const nombre = `Rápido ${Date.now() % 100000}`;
  await page.getByLabel("Nombre").fill(nombre);
  await page.getByLabel("Estilo").selectOption({ label: "No definido" });
  await page.getByLabel("Zurda").check();
  await page.getByLabel("Peso (kg)").fill("72.5");
  await page.getByRole("button", { name: "Crear y entrar" }).click();
  await expect(page).toHaveURL(/\/inicio\/$/);

  await page.goto("/amigos/");
  for (const n of ["Fajador", "Estilista", "Zurdo"]) await expect(page.getByRole("link", { name: n, exact: true })).toBeVisible();
  await page.goto("/ranking/");
  await expect(page.getByRole("link", { name: /Fajador/ })).toBeVisible();

  await page.getByRole("link", { name: "Cambiar usuario" }).click();
  await expect(page.getByRole("list", { name: "Usuarios creados" }).getByRole("button", { name: `Entrar como ${nombre}` })).toBeVisible();
  await page.getByRole("button", { name: "Entrar como Fajador" }).click();
  await expect(page).toHaveURL(/\/inicio\/$/);
  await page.goto("/amigos/");
  await expect(page.getByRole("link", { name: nombre, exact: true })).toBeVisible();
});
