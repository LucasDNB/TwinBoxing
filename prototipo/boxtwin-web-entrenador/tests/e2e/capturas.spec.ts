/*
 * Capturas de las pantallas principales, para revisar a ojo (no es una prueba de aceptacion).
 * USO: CAPTURAS=<dir> E2E_SIN_WEBKIT=1 npx playwright test capturas
 */
import { expect, test } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const DIR = process.env.CAPTURAS;
test.skip(!DIR, "solo con CAPTURAS=<dir>");

test("capturas", async ({ page, request }, info) => {
  const email = `cap-${Date.now()}@usal.edu.ar`;
  const token = (await (await request.post("/auth/registro", { data: { email, clave: "sparring2026" } })).json()).token;
  const h = { Authorization: `Bearer ${token}` };
  const ana = (await (await request.post("/boxeadores", { headers: h, data: { nombre: "Ana Paz", guardia: "ortodoxa", categoria: "Welter", altura_m: 1.72, envergadura_m: 1.78 } })).json()).id;
  const bea = (await (await request.post("/boxeadores", { headers: h, data: { nombre: "Bea Ruiz", guardia: "zurda", categoria: "Welter", altura_m: 1.68, envergadura_m: 1.7 } })).json()).id;
  const video = fs.readFileSync(path.join(__dirname, ".e2e", "sparring.mp4"));
  const r = await request.post("/videos", { headers: h, multipart: { archivo: { name: "sparring.mp4", mimeType: "video/mp4", buffer: video }, consentimiento: "true", contexto: "sparring", rounds: "3", round_s: "180", descanso_s: "60" } });
  const id = (await r.json()).job_id;
  await request.put(`/sesiones/${id}/boxeadores`, { headers: h, data: { boxeador_a: ana, boxeador_b: bea } });
  await expect.poll(async () => (await (await request.get(`/jobs/${id}`, { headers: h })).json()).estado, { timeout: 20_000 }).toBe("espera_siembra");

  const p = info.project.name;
  const foto = async (n: string) => {
    await page.waitForTimeout(700);
    await page.screenshot({ path: path.join(DIR!, `${p}-${n}.png`), fullPage: true });
  };
  await page.goto("/ingresar/");
  await foto("00-ingreso");
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Clave").fill("sparring2026");
  await page.getByRole("button", { name: "Entrar" }).click();
  await expect(page.getByRole("heading", { name: "Panel" })).toBeVisible();
  await foto("01-panel");
  await page.goto(`/sesion/identificar/?id=${id}`);
  await expect(page.getByRole("article").first()).toBeVisible();
  await foto("03-identificar");
  await page.goto("/sesiones/nueva/");
  await page.getByTestId("input-galeria").setInputFiles(path.join(__dirname, ".e2e", "sparring.mp4"));
  await foto("02-nueva");
  await request.post(`/jobs/${id}/siembra`, { headers: h, data: { track_a: 41, track_b: 64 } });
  await expect.poll(async () => (await (await request.get(`/jobs/${id}`, { headers: h })).json()).estado, { timeout: 20_000 }).toBe("listo");
  await page.goto(`/sesion/?id=${id}&vista=video`);
  await expect(page.getByLabel("Video de la sesión")).toBeVisible();
  await foto("04-video");
  await page.goto(`/sesion/?id=${id}&vista=fightcard`);
  await expect(page.getByText("Golpes detectados").first()).toBeVisible();
  await foto("05-fightcard");
  await page.goto(`/peleador/?id=${ana}`);
  await expect(page.getByRole("heading", { name: "Ana Paz" })).toBeVisible();
  await foto("06-perfil");
  await page.goto(`/rival/?a=${ana}&b=${bea}`);
  await expect(page.getByRole("table", { name: "Frente a frente" })).toBeVisible();
  await foto("07-rival");
});
