/* Dia 3: ranking, amigos, perfil publico con denuncia y rival fantasma. */
import { expect, test } from "@playwright/test";
import { emailNuevo, registrar } from "./util";

test("ranking muestra el pie de actividad", async ({ page }) => {
  await registrar(page);
  await page.goto("/ranking/");
  await expect(page.getByText("Ranking de actividad: cuenta golpes detectados, no golpes que conectaron")).toBeVisible();
});

test("rival fantasma contra un arquetipo", async ({ page }) => {
  await registrar(page);
  await page.goto("/rival/");
  await page.getByLabel("Rival").selectOption({ label: "El Toro Simulado (fajador)" });
  await expect(page.getByRole("table", { name: "Lado a lado" })).toBeVisible();
  await expect(page.getByText("Perfil simulado").first()).toBeVisible();
  await expect(page.getByText("Los consejos salen de los golpes detectados; no son una predicción.")).toBeVisible();
});

test("amigos: solicitud, aceptar y ver el perfil; denunciar un publico", async ({ browser }) => {
  const a = await (await browser.newContext()).newPage();
  const b = await (await browser.newContext()).newPage();
  const ea = await registrar(a, emailNuevo("a"));
  const eb = await registrar(b, emailNuevo("b"));
  await a.goto("/amigos/");
  await a.getByLabel("Agregar por email").fill(eb);
  await a.getByRole("button", { name: "Enviar" }).click();
  await expect(a.getByRole("status")).toContainText("le llega la solicitud");

  await b.goto("/amigos/");
  await b.getByRole("button", { name: "Aceptar" }).click();
  await b.getByRole("link", { name: "Boxeador" }).click();
  await expect(b.getByRole("tablist", { name: "Modalidad" })).toBeVisible();
  await expect(b.getByRole("link", { name: "Usar como rival fantasma" })).toBeVisible();

  // a se vuelve publico y b lo denuncia.
  const ps = await (await a.request.get("/api/perfiles")).json();
  await a.request.patch(`/api/perfiles/${ps[0].id}`, { data: { publico: true } });
  await b.goto(`/publico/?id=${ps[0].id}`);
  await b.getByRole("button", { name: "Denunciar" }).click();
  await b.getByLabel("Aparece un menor de edad").check();
  await b.getByRole("button", { name: "Enviar denuncia" }).click();
  await expect(b.getByRole("dialog").getByRole("status")).toContainText("Recibimos la denuncia");
  expect(ea).toContain("@");
});
