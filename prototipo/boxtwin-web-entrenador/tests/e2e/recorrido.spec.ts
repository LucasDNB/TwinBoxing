/*
 * De punta a punta (06 §4), contra la API real y el build exportado.
 *
 * Recorridos: I1 completo; correccion; exportar CSV; perfil de un peleador con dos sesiones;
 * rival con dos perfiles. En cada pagina, axe sin violaciones de impacto serio o critico.
 */
import { expect, test, type APIRequestContext, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import path from "node:path";

const VIDEO = path.join(__dirname, ".e2e", "sparring.mp4");
const CLAVE = "sparring2026";

async function sinViolacionesSerias(page: Page, donde: string) {
  // Axe mide el contraste con la opacidad del momento: en medio de un fade de entrada da
  // falsos negativos. Se espera a que no quede ninguna animacion corriendo.
  await page.waitForFunction(() => document.getAnimations().every((a) => a.playState !== "running"), null, { timeout: 5_000 }).catch(() => {});
  await page.waitForTimeout(400);
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
  const serias = r.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(serias.map((v) => `${v.id}: ${v.help} -> ${v.nodes.map((n) => n.html.slice(0, 160)).join(" | ")}`), `axe en ${donde}`).toEqual([]);
}

async function cuenta(request: APIRequestContext, prefijo: string) {
  const email = `${prefijo}-${Date.now()}-${Math.round(Math.random() * 1e6)}@usal.edu.ar`;
  const r = await request.post("/auth/registro", { data: { email, clave: CLAVE } });
  expect(r.ok()).toBeTruthy();
  return { email, token: (await r.json()).token as string };
}

async function entrar(page: Page, email: string) {
  await page.goto("/ingresar/");
  await sinViolacionesSerias(page, "ingreso");
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Clave").fill(CLAVE);
  await page.getByRole("button", { name: "Entrar" }).click();
  await expect(page).toHaveURL(/:\d+\/$/);
  await expect(page.getByRole("heading", { name: "Panel" })).toBeVisible();
}

/* Una sesion lista por la API, sin pasar por la pantalla: para los recorridos de perfiles. */
async function sesionLista(request: APIRequestContext, token: string, a: string | null, b: string | null) {
  const h = { Authorization: `Bearer ${token}` };
  const fs = await import("node:fs");
  const r = await request.post("/videos", {
    headers: h,
    multipart: {
      archivo: { name: "sparring.mp4", mimeType: "video/mp4", buffer: fs.readFileSync(VIDEO) },
      consentimiento: "true", contexto: "sparring",
    },
  });
  expect(r.status()).toBe(202);
  const id = (await r.json()).job_id as string;
  await expect.poll(async () => (await (await request.get(`/jobs/${id}`, { headers: h })).json()).estado, { timeout: 20_000 }).toBe("espera_siembra");
  await request.post(`/jobs/${id}/siembra`, { headers: h, data: { track_a: 41, track_b: 64 } });
  await expect.poll(async () => (await (await request.get(`/jobs/${id}`, { headers: h })).json()).estado, { timeout: 20_000 }).toBe("listo");
  await request.put(`/sesiones/${id}/boxeadores`, { headers: h, data: { boxeador_a: a, boxeador_b: b } });
  return id;
}

test("I1: subir, esperar, identificar y llegar a la Fight-Card", async ({ page, request }) => {
  const { email } = await cuenta(request, "i1");
  await entrar(page, email);
  await sinViolacionesSerias(page, "panel vacío");

  await page.goto("/sesiones/nueva/");
  await expect(page.getByText("Qué se filmó")).toBeVisible();
  await sinViolacionesSerias(page, "nueva sesión");
  await page.getByTestId("input-galeria").setInputFiles(VIDEO);
  await expect(page.getByText(/sparring\.mp4 ·/).first()).toBeVisible();
  await page.getByText(/dieron su consentimiento/).click();
  await page.getByRole("button", { name: "Subir y procesar" }).click();

  await expect(page).toHaveURL(/\/sesion\/\?id=/, { timeout: 20_000 });
  const marcar = page.getByRole("link", { name: "Marcar peleadores" });
  await expect(marcar).toBeVisible({ timeout: 20_000 });
  await marcar.click();

  await expect(page.getByRole("heading", { name: "¿Quién es quién?" })).toBeVisible();
  const tarjetas = page.getByRole("article");
  await expect(tarjetas).toHaveCount(2);
  // Los recortes cargan con ticket (CA-03.3).
  const img = tarjetas.first().getByRole("img", { name: /Recorte/ });
  await expect(img).toBeVisible();
  expect(await img.evaluate((i: HTMLImageElement) => i.complete && i.naturalWidth > 0)).toBe(true);
  await sinViolacionesSerias(page, "identificación");
  await tarjetas.nth(0).getByRole("radio", { name: "Es A" }).click();
  await tarjetas.nth(1).getByRole("radio", { name: "Es B" }).click();
  await page.getByRole("button", { name: "Completar análisis" }).click();
  await page.getByRole("button", { name: "Sí, completar" }).click();

  await expect(page).toHaveURL(/vista=fightcard/, { timeout: 20_000 });
  const fila = page.locator("[data-fila]").filter({ hasText: "Golpes detectados" }).first();
  await expect(fila).toContainText("51");
  await expect(fila).toContainText("32");
  await expect(page.getByText(/conectad|ganador/i)).toHaveCount(0);
  await sinViolacionesSerias(page, "Fight-Card");
});

test("corrección, export CSV y video", async ({ page, request }) => {
  const { email, token } = await cuenta(request, "corr");
  const id = await sesionLista(request, token, null, null);
  await entrar(page, email);
  await page.goto(`/sesion/?id=${id}&vista=video`);
  const video = page.getByLabel("Video de la sesión");
  await expect(video).toBeVisible();
  expect(await video.getAttribute("src")).toMatch(/\?t=/);
  await sinViolacionesSerias(page, "video y golpes");

  // Tocar un golpe salta medio segundo antes (CA-04.1).
  await page.getByRole("button", { name: /Ir al golpe de 00:08/ }).first().click();
  await expect.poll(() => video.evaluate((v: HTMLVideoElement) => Math.round(v.currentTime * 10) / 10)).toBeCloseTo(7.7, 0);

  const consola = page.getByRole("region", { name: /Golpes de/ }).first();
  await consola.getByRole("button", { name: /Corregir el golpe/ }).first().click();
  await page.getByRole("radio", { name: "uppercut" }).click();
  await page.getByRole("button", { name: "Guardar" }).click();
  await expect(consola.getByText("corregido").first()).toBeVisible();
  await page.reload();
  await expect(page.getByRole("region", { name: /Golpes de/ }).first().getByText("corregido").first()).toBeVisible();

  await page.goto(`/sesion/?id=${id}&vista=fightcard`);
  await page.getByRole("button", { name: /Exportar/ }).click();
  const [descarga] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("menuitem", { name: "Lista de golpes (CSV)" }).click(),
  ]);
  expect(descarga.suggestedFilename()).toBe(`${id}.csv`);
});

test("perfil con dos sesiones y rival con dos perfiles", async ({ page, request }) => {
  const { email, token } = await cuenta(request, "perf");
  const h = { Authorization: `Bearer ${token}` };
  const ana = (await (await request.post("/boxeadores", { headers: h, data: { nombre: "Ana Paz", guardia: "ortodoxa", altura_m: 1.72, envergadura_m: 1.78 } })).json()).id;
  const bea = (await (await request.post("/boxeadores", { headers: h, data: { nombre: "Bea Ruiz", guardia: "zurda", altura_m: 1.68, envergadura_m: 1.7 } })).json()).id;
  await sesionLista(request, token, ana, bea);
  await sesionLista(request, token, bea, ana);

  await entrar(page, email);
  await sinViolacionesSerias(page, "panel con sesiones");
  await page.goto(`/peleador/?id=${ana}`);
  await expect(page.getByRole("heading", { name: "Ana Paz" })).toBeVisible();
  await expect(page.getByText("2 sesiones · 2 rounds")).toBeVisible();
  await sinViolacionesSerias(page, "perfil");

  await page.goto("/peleadores/");
  await expect(page.getByRole("link", { name: /Bea Ruiz/ })).toBeVisible();
  await sinViolacionesSerias(page, "peleadores");

  await page.goto(`/rival/?a=${ana}&b=${bea}`);
  await expect(page.getByRole("table", { name: "Frente a frente" })).toBeVisible();
  await expect(page.getByText(/Tiene 8 cm más de envergadura/)).toBeVisible();
  await sinViolacionesSerias(page, "rival");

  await page.goto("/sesiones/");
  await expect(page.getByText("Ver Fight-Card").first()).toBeVisible();
  await sinViolacionesSerias(page, "sesiones");
});
