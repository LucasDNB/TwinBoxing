/*
 * De punta a punta (06 §4): el build exportado servido por la API real con SQLite y un
 * worker real que llama a un anotador falso (tests/e2e/boxtwin_falso.py). Sin GPU.
 *
 * Celular: Pixel 7 (Chromium) e iPhone 13 (WebKit). Escritorio: 1440 x 900, Chromium.
 * WebKit necesita librerias del sistema (libevent) que en la estacion piden sudo: si no
 * estan, ese proyecto se saltea con E2E_SIN_WEBKIT=1 y queda para CI y el celular real.
 */
import { defineConfig, devices } from "@playwright/test";

const PUERTO = Number(process.env.PUERTO ?? 8765);

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 60_000,
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${PUERTO}`,
    trace: "retain-on-failure",
    locale: "es-AR",
  },
  webServer: {
    command: "bash tests/e2e/levantar.sh",
    url: `http://127.0.0.1:${PUERTO}/salud`,
    reuseExistingServer: false,
    timeout: 60_000,
    env: { PUERTO: String(PUERTO) },
  },
  projects: [
    { name: "celular-pixel7", use: { ...devices["Pixel 7"] } },
    ...(process.env.E2E_SIN_WEBKIT ? [] : [{ name: "celular-iphone13", use: { ...devices["iPhone 13"] } }]),
    { name: "escritorio", use: { browserName: "chromium" as const, viewport: { width: 1440, height: 900 } } },
  ],
});
