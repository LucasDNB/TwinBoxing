/* Z15: lo que Chrome pide para instalar: manifest valido, iconos 192/512 y service worker activo. */
import { expect, test } from "@playwright/test";

test("manifest e iconos", async ({ page, request }) => {
  await page.goto("/");
  const href = await page.locator('link[rel="manifest"]').getAttribute("href");
  const m = await (await request.get(href!)).json();
  expect(m).toMatchObject({ display: "standalone", start_url: "/inicio/", name: "BoxTwinZ" });
  for (const t of ["192x192", "512x512"]) {
    const i = m.icons.find((x: { sizes: string }) => x.sizes === t);
    expect((await request.get(i.src)).status()).toBe(200);
  }
});

test("el service worker toma la pagina y no cachea la API", async ({ page, browserName }) => {
  test.skip(browserName === "webkit", "WebKit de Playwright no registra service workers sobre http");
  await page.goto("/");
  await page.waitForFunction(() => navigator.serviceWorker?.controller !== null || navigator.serviceWorker.ready.then(() => true), null, { timeout: 15_000 });
  await page.reload();
  expect(await page.evaluate(() => !!navigator.serviceWorker.controller)).toBe(true);
  const cacheadas = await page.evaluate(async () => {
    const ks = await caches.keys();
    const urls: string[] = [];
    for (const k of ks) for (const r of await (await caches.open(k)).keys()) urls.push(new URL(r.url).pathname);
    return urls;
  });
  expect(cacheadas.some((u) => u.startsWith("/api/"))).toBe(false);
});
