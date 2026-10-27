/*
 * De punta a punta (06 §4): el build exportado servido por la API real con SQLite y un
 * worker real que llama a un anotador falso (tests/e2e/boxtwin_falso.py). Sin GPU.
 *
 * Celular: Pixel 7 (Chromium) e iPhone 13 (WebKit). Escritorio: 1440 x 900, Chromium.
 * WebKit necesita librerias del sistema (libevent) que en la estacion piden sudo: si no
 * estan, ese proyecto se saltea con E2E_SIN_WEBKIT=1 y queda para CI y el celular real.
 *
 * Versiones 26-10 (05 §4): E1 a E3 entran por los botones de la version de prueba, que
 * reemplaza el ingreso con email. Corren contra un segundo servidor con BOXTWIN_DEMO
 * encendida y la base sembrada por demo.py (levantar.sh con E2E_DEMO=1).
 */
import { defineConfig, devices } from "@playwright/test";

const PUERTO = Number(process.env.PUERTO ?? 8765);
const PUERTO_DEMO = PUERTO + 1;
const VERSIONES = /versiones\.spec\.ts/;

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
  webServer: [
    {
      command: "bash tests/e2e/levantar.sh",
      url: `http://127.0.0.1:${PUERTO}/salud`,
      reuseExistingServer: false,
      timeout: 60_000,
      env: { PUERTO: String(PUERTO) },
    },
    {
      command: "bash tests/e2e/levantar.sh",
      url: `http://127.0.0.1:${PUERTO_DEMO}/salud`,
      reuseExistingServer: false,
      timeout: 90_000,
      env: { PUERTO: String(PUERTO_DEMO), E2E_DEMO: "1" },
    },
  ],
  projects: [
    { name: "celular-pixel7", testIgnore: VERSIONES, use: { ...devices["Pixel 7"] } },
    ...(process.env.E2E_SIN_WEBKIT ? [] : [{ name: "celular-iphone13", testIgnore: VERSIONES, use: { ...devices["iPhone 13"] } }]),
    { name: "escritorio", testIgnore: VERSIONES, use: { browserName: "chromium" as const, viewport: { width: 1440, height: 900 } } },
    // E1 y E3 en Pixel 7, E2 en escritorio: cada prueba elige su dispositivo con test.use.
    { name: "versiones", testMatch: VERSIONES, use: { baseURL: `http://127.0.0.1:${PUERTO_DEMO}` } },
  ],
});
