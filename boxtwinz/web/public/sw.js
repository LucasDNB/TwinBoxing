/*
 * Service worker de BoxTwinZ (spec 12): cachea el shell para que la app abra instalada, y
 * NUNCA cachea video ni la API. Estrategia: red primero para navegacion (con el shell
 * cacheado de respaldo sin conexion), cache primero para los estaticos con hash de Next.
 */
const VERSION = "boxtwinz-shell-v1";
const SHELL = ["/", "/inicio/", "/manifest.webmanifest", "/icono.svg", "/icono-192.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== self.location.origin) return;
  // Ni API ni video: siempre a la red (la API lleva cookies y el video va por rangos).
  if (url.pathname.startsWith("/api/") || e.request.headers.has("range")) return;
  if (url.pathname.startsWith("/_next/static/")) {
    e.respondWith(caches.match(e.request).then((r) => r || fetch(e.request).then((resp) => {
      const copia = resp.clone();
      caches.open(VERSION).then((c) => c.put(e.request, copia));
      return resp;
    })));
    return;
  }
  if (e.request.mode === "navigate") {
    e.respondWith(fetch(e.request).catch(() => caches.match(url.pathname).then((r) => r || caches.match("/"))));
  }
});
