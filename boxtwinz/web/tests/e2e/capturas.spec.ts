/* CZ8: capturas a 375 px para TAREAS.md. Solo con CAPTURAS=1, en Pixel 7. */
import { test } from "@playwright/test";
import path from "node:path";

const DESTINO = path.resolve(__dirname, "../../../../docs/specs/2026-11-03-boxtwinz/capturas");
test.skip(!process.env.CAPTURAS, "solo con CAPTURAS=1");
test.use({ viewport: { width: 375, height: 812 } });

test("capturas de demo", async ({ page }, info) => {
  test.skip(info.project.name !== "celular-pixel7");
  await page.goto("/");
  await page.screenshot({ path: `${DESTINO}/01-bienvenida.png` });
  await page.getByLabel("Email").fill("demo-premium@boxtwinz.ar");
  await page.getByLabel("Contraseña").fill("clave-de-demo");
  await page.getByRole("button", { name: "Ingresar" }).click();
  await page.waitForURL(/\/inicio\/$/);
  await page.getByText("Últimas sesiones").waitFor();
  await page.screenshot({ path: `${DESTINO}/02-inicio.png`, fullPage: true });
  await page.getByRole("link", { name: /^Sparring \d/ }).first().click();
  await page.getByRole("table", { name: "Métricas de la sesión" }).waitFor();
  await page.screenshot({ path: `${DESTINO}/03-resultado-sparring.png`, fullPage: true });
  await page.goto("/rival/");
  await page.getByLabel("Rival").selectOption({ label: "El Toro Simulado (fajador)" });
  await page.getByRole("table", { name: "Lado a lado" }).waitFor();
  await page.screenshot({ path: `${DESTINO}/04-rival-fantasma.png`, fullPage: true });
  await page.goto("/ranking/");
  await page.getByRole("link", { name: /Demo/ }).first().waitFor();
  await page.screenshot({ path: `${DESTINO}/05-ranking.png` });
});
