/*
 * Z16: axe (WCAG 2.1 AA) en cada pantalla, a 375 px de ancho sin scroll horizontal (CZ8), y
 * los textos obligatorios de la spec 12 (CZ7).
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { emailNuevo, registrar, subir, valorar } from "./util";

test.use({ viewport: { width: 375, height: 812 } });

async function sinViolaciones(page: Page, nombre: string) {
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
  expect(r.violations.map((v) => `${nombre}: ${v.id} (${v.nodes.length})`)).toEqual([]);
  const ancho = await page.evaluate(() => document.documentElement.scrollWidth);
  expect(ancho, `${nombre} desborda a 375 px`).toBeLessThanOrEqual(375);
}

test("pantallas publicas", async ({ page }) => {
  await page.goto("/");
  await sinViolaciones(page, "bienvenida");
  await page.goto("/registro/");
  await sinViolaciones(page, "registro");
});

test("pantallas con sesion y textos obligatorios", async ({ page }) => {
  await registrar(page, emailNuevo("a11y"));
  await sinViolaciones(page, "inicio");
  await expect(page.getByText("Perfil simulado").first()).toBeVisible();

  const id = await subir(page, "Bolsa");
  await expect(page.getByRole("table", { name: "Métricas de la sesión" })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("golpes detectados").first()).toBeVisible();
  await expect(page.getByText("Reconocimiento en evaluación")).toBeVisible();
  await expect(page.getByText("estimación").first()).toBeVisible();
  await sinViolaciones(page, "resultado");
  await valorar(page);
  await expect(page.getByText(/golpes lanzados/i)).toHaveCount(0);

  for (const [ruta, nombre] of [["/subir/", "subir"], ["/ranking/", "ranking"], ["/amigos/", "amigos"], ["/rival/", "rival"], ["/ajustes/", "ajustes"]]) {
    await page.goto(ruta);
    await page.getByRole("heading", { level: 1 }).waitFor();
    await sinViolaciones(page, nombre);
  }
  await expect(page.getByText("Pago simulado — versión de demostración")).toBeVisible();

  await page.goto("/inicio/");
  await page.getByRole("link", { name: /Tu perfil/ }).click();
  await page.getByRole("tab", { name: "Bolsa" }).click();
  await sinViolaciones(page, "perfil");
  await page.goto(`/comparar/?a=${id}&b=${id}`);
  await expect(page.getByText("Comparación entre rounds sin validar")).toBeVisible();
  await sinViolaciones(page, "comparar");
});

test("panel de administracion", async ({ page }) => {
  await registrar(page, "admin@boxtwinz.ar").catch(async () => {
    await page.goto("/");
    await page.getByLabel("Email").fill("admin@boxtwinz.ar");
    await page.getByLabel("Contraseña").fill("una-clave-larga");
    await page.getByRole("button", { name: "Ingresar" }).click();
    await expect(page).toHaveURL(/\/inicio\/$/);
  });
  await page.goto("/admin/");
  await expect(page.getByRole("tab", { name: "Uso" })).toBeVisible();
  await sinViolaciones(page, "admin");
  await page.getByRole("tab", { name: "Capacidad" }).click();
  await expect(page.getByRole("img", { name: /Utilización diaria/ })).toBeVisible();
  await page.getByRole("tab", { name: "Moderación" }).click();
  await expect(page.getByRole("button", { name: "Empezar tanda" })).toBeVisible();
});
