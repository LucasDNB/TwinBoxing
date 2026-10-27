/*
 * De punta a punta de las versiones (specs/versiones-26-10, 05 §4), contra el servidor con la
 * version de prueba encendida y la base sembrada por demo.py (levantar.sh con E2E_DEMO=1).
 *
 * E1 · Pixel 7: "Individual", subir bolsa, marcarse, Fight-Card y Mi progreso con la sesion.
 * E2 · Escritorio: "Gimnasio · titular", Equipo, cargar dos atletas con su consentimiento.
 * E3 · Pixel 7: "Gimnasio · entrenador", sparring con los atletas de E2, Fight-Card, perfil;
 *      "Salir" y entrar como "Entrenador" no muestra nada del gimnasio ni de la cuenta real.
 *
 * Van en serie: comparten las cuentas de prueba, como dos visitantes que tocan el mismo boton.
 */
import { devices, expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import fs from "node:fs";
import path from "node:path";

const VIDEO = path.join(__dirname, ".e2e", "sparring.mp4");
const ATLETAS = ["Atleta E2 uno", "Atleta E2 dos"];
// Pixel 7 sin defaultBrowserType: el proyecto ya es Chromium, y cambiarlo por grupo pide otro worker.
const { defaultBrowserType: _navegador, ...PIXEL7 } = devices["Pixel 7"];

test.describe.configure({ mode: "serial" });

async function sinViolacionesSerias(page: Page, donde: string) {
  await page.waitForFunction(() => document.getAnimations().every((a) => a.playState !== "running"), null, { timeout: 5_000 }).catch(() => {});
  await page.waitForTimeout(400);
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
  const serias = r.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(serias.map((v) => `${v.id}: ${v.help} -> ${v.nodes.map((n) => n.html.slice(0, 160)).join(" | ")}`), `axe en ${donde}`).toEqual([]);
}

async function entrarComo(page: Page, boton: RegExp) {
  await page.goto("/ingresar/");
  await expect(page.getByText("Versión de prueba")).toBeVisible();
  // CA-00.7: ningun campo de texto.
  await expect(page.locator("input")).toHaveCount(0);
  await sinViolacionesSerias(page, "ingreso de prueba");
  await page.getByRole("button", { name: boton }).click();
  await expect(page).toHaveURL(/:\d+\/$/);
}

async function token(page: Page): Promise<string> {
  return page.evaluate(() => localStorage.getItem("boxtwin.token") ?? "");
}

test.describe("E1 · Individual en el celular", () => {
  test.use(PIXEL7);

  test("botón Individual: subir bolsa, marcarse, Fight-Card y Mi progreso", async ({ page }) => {
    await entrarComo(page, /^Individual/);
    await expect(page.getByRole("heading", { name: "Mi progreso" })).toBeVisible();
    await expect(page.getByTestId("rotulo-demo")).toHaveText("Prueba · Individual");
    const nav = page.getByRole("navigation", { name: "Principal" }).last();
    await expect(nav.getByRole("link")).toHaveText(["Inicio", "Sesiones", "Subir", "Mi perfil"]);
    await sinViolacionesSerias(page, "Mi progreso");

    await nav.getByRole("link", { name: "Subir" }).click();
    await expect(page.getByTestId("aparece-como")).toContainText("Vas a aparecer como Boxeador de prueba");
    await page.getByRole("radio", { name: /Bolsa/ }).click();
    await page.getByTestId("input-galeria").setInputFiles(VIDEO);
    await expect(page.getByText(/sparring\.mp4 ·/).first()).toBeVisible();
    await page.getByText("Soy yo quien aparece en el video").click();
    await sinViolacionesSerias(page, "subir en Individual");
    await page.getByRole("button", { name: "Subir y procesar" }).click();

    await expect(page).toHaveURL(/\/sesion\/\?id=/, { timeout: 20_000 });
    const marcar = page.getByRole("link", { name: "Marcarme" });
    await expect(marcar).toBeVisible({ timeout: 20_000 });
    await marcar.click();
    await expect(page.getByRole("heading", { name: "¿Cuál sos vos?" })).toBeVisible();
    await sinViolacionesSerias(page, "marcarse");
    await page.getByRole("article").first().getByRole("radio", { name: "Soy yo" }).click();
    await page.getByRole("button", { name: "Completar análisis" }).click();
    await page.getByRole("button", { name: "Sí, completar" }).click();

    await expect(page).toHaveURL(/vista=fightcard/, { timeout: 20_000 });
    await expect(page.getByText("Peleador B")).toHaveCount(0);
    await sinViolacionesSerias(page, "Fight-Card en Individual");

    await page.goto("/");
    const sesiones = page.getByRole("region", { name: "Tus sesiones" });
    // Las dos de la siembra y la recien subida.
    await expect(sesiones.getByRole("link", { name: /Listo/ })).toHaveCount(3, { timeout: 10_000 });
  });
});

test.describe("E2 · Gimnasio, titular en escritorio", () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  test("botón Gimnasio · titular: Equipo, dos atletas y su consentimiento", async ({ page }) => {
    await entrarComo(page, /Gimnasio · titular/);
    await expect(page.getByRole("heading", { name: "Gimnasio de prueba" })).toBeVisible();
    const lateral = page.getByRole("complementary").getByRole("navigation", { name: "Principal" });
    await expect(lateral.getByRole("link")).toHaveText(["Panel", "Sesiones", "Subir sesión", "Plantel", "Rival Fantasma", "Equipo"]);
    await sinViolacionesSerias(page, "panel del gimnasio");

    await lateral.getByRole("link", { name: "Equipo" }).click();
    await expect(page.getByText("Entrenador de prueba")).toBeVisible();
    await expect(page.getByText("En la versión de prueba el equipo viene cargado.")).toBeVisible();
    await expect(page.locator("main")).not.toContainText("@");
    await sinViolacionesSerias(page, "equipo");

    await lateral.getByRole("link", { name: "Plantel" }).click();
    await expect(page.getByRole("heading", { name: "Plantel" })).toBeVisible();
    for (const nombre of ATLETAS) {
      await page.getByRole("button", { name: /Agregar/ }).first().click();
      await page.getByLabel("Nombre").fill(nombre);
      await page.getByRole("button", { name: "Guardar" }).click();
      // "Registrar el consentimiento ahora" viene marcado: se abre la hoja con el texto.
      await expect(page.getByText(`Consentimiento de ${nombre}`)).toBeVisible();
      await page.getByText("El atleta leyó y aceptó este texto.").click();
      await page.getByText("Tiene 18 años o más.").click();
      await page.getByRole("button", { name: "Confirmar" }).click();
      await expect(page.getByText(`Consentimiento de ${nombre}`)).toHaveCount(0);
    }
    for (const nombre of ATLETAS) {
      const tarjeta = page.getByRole("listitem").filter({ hasText: nombre });
      await expect(tarjeta.getByText("Consentimiento vigente")).toBeVisible();
    }
    await sinViolacionesSerias(page, "plantel");
  });
});

test.describe("E3 · Gimnasio, entrenador en el celular", () => {
  test.use(PIXEL7);

  test("botón Gimnasio · entrenador: sparring con los atletas de E2; Salir y Entrenador no ve nada del gimnasio", async ({ page, request }) => {
    await entrarComo(page, /Gimnasio · entrenador/);
    await expect(page.getByTestId("rotulo-demo")).toHaveText("Prueba · Gimnasio · entrenador");
    await page.goto("/sesiones/nueva/");
    await page.getByTestId("input-galeria").setInputFiles(VIDEO);
    await expect(page.getByText(/sparring\.mp4 ·/).first()).toBeVisible();
    await page.getByLabel(/Rincón rojo/).selectOption({ label: ATLETAS[0] });
    await page.getByLabel(/Rincón azul/).selectOption({ label: ATLETAS[1] });
    await page.getByText(/dieron su consentimiento/).click();
    await page.getByRole("button", { name: "Subir y procesar" }).click();

    await expect(page).toHaveURL(/\/sesion\/\?id=/, { timeout: 20_000 });
    const sesion = new URL(page.url()).searchParams.get("id")!;
    const marcar = page.getByRole("link", { name: "Marcar peleadores" });
    await expect(marcar).toBeVisible({ timeout: 20_000 });
    await marcar.click();
    const tarjetas = page.getByRole("article");
    await tarjetas.nth(0).getByRole("radio", { name: "Es A" }).click();
    await tarjetas.nth(1).getByRole("radio", { name: "Es B" }).click();
    await page.getByRole("button", { name: "Completar análisis" }).click();
    await page.getByRole("button", { name: "Sí, completar" }).click();
    await expect(page).toHaveURL(/vista=fightcard/, { timeout: 20_000 });
    await expect(page.getByText(/Subió Entrenador de prueba/)).toBeVisible();
    await sinViolacionesSerias(page, "Fight-Card en Gimnasio");

    // El perfil de un atleta que cargo el titular, con la sesion que subio este entrenador.
    await page.goto("/peleadores/");
    await page.getByRole("link", { name: new RegExp(ATLETAS[0]) }).click();
    await expect(page.getByRole("heading", { name: ATLETAS[0] })).toBeVisible();
    await expect(page.getByText(/^1 sesión/)).toBeVisible();
    await expect(page.getByText("Consentimiento vigente")).toBeVisible();
    await sinViolacionesSerias(page, "perfil del atleta");

    // Salir y entrar como Entrenador: nada del gimnasio, ni de la cuenta real.
    await page.getByRole("button", { name: "Salir" }).click();
    await expect(page.getByText("Versión de prueba")).toBeVisible();
    await page.getByRole("button", { name: /^Entrenador/ }).click();
    await expect(page.getByRole("heading", { name: "Panel" })).toBeVisible();
    await page.goto("/sesiones/");
    await expect(page.getByText("Ver Fight-Card").first()).toBeVisible();
    await expect(page.locator("main")).not.toContainText(ATLETAS[0]);
    const h = { Authorization: `Bearer ${await token(page)}` };
    expect((await request.get(`/jobs/${sesion}`, { headers: h })).status()).toBe(404);
    const reales: Record<string, string> = JSON.parse(fs.readFileSync(path.join(__dirname, ".e2e", "real.json"), "utf8"));
    for (const id of Object.values(reales)) {
      expect((await request.get(`/jobs/${id}`, { headers: h })).status()).toBe(404);
    }
  });
});
