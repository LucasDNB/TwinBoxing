import { expect, type Page } from "@playwright/test";
import path from "node:path";

export const VIDEO = path.join(__dirname, ".e2e", "sesion.mp4");

export function emailNuevo(prefijo = "e2e") {
  return `${prefijo}-${Date.now()}-${Math.floor(Math.random() * 1e6)}@boxtwinz.ar`;
}

export async function registrar(page: Page, email = emailNuevo(), guardia: "Ortodoxa" | "Zurda" = "Ortodoxa") {
  await page.goto("/registro/");
  await page.getByLabel("Nombre visible").fill("Boxeador");
  await page.getByLabel("Email").fill(email);
  await page.getByLabel(/Contraseña/).fill("una-clave-larga");
  await page.getByLabel(guardia).check();
  await page.getByRole("checkbox", { name: /Tengo 18 años/ }).check();
  await page.getByRole("button", { name: "Crear cuenta" }).click();
  await expect(page).toHaveURL(/\/inicio\/$/);
  return email;
}

export async function subir(page: Page, modalidad: "Sombra" | "Bolsa" | "Sparring", rounds?: string) {
  await page.goto("/subir/");
  await page.getByRole("button", { name: modalidad, exact: true }).click();
  await page.getByRole("button", { name: "Seguir" }).click();
  await page.getByRole("button", { name: "Ya lo tengo filmado" }).click();
  await page.getByLabel("Video (mp4 o mov)").setInputFiles(VIDEO);
  if (rounds) await page.getByLabel("Rounds (opcional)").fill(rounds);
  await page.getByRole("button", { name: "Subir" }).click();
  await expect(page).toHaveURL(/\/sesion\/\?id=/, { timeout: 30_000 });
  return new URL(page.url()).searchParams.get("id")!;
}

export async function valorar(page: Page, estrellas = 4) {
  await page.getByRole("button", { name: `${estrellas} estrellas` }).first().click();
  await page.getByRole("button", { name: "Enviar valoración" }).first().click();
}
