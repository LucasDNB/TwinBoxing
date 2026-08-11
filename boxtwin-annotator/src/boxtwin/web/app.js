/*
 * BoxTwin - Cliente del anotador.
 *
 * POR QUE ESTA ASI
 *   Un solo sistema de coordenadas: pixeles del video ORIGINAL. La imagen que llega puede
 *   ser del proxy (960 px) o del original (1920), y los keypoints vienen siempre en
 *   resolucion original. Se dibuja la imagen estirada al tamano original y todo lo demas
 *   encima, sin conversiones intermedias. Si cada capa hiciera la suya, el esqueleto se
 *   correria del cuerpo lo justo para no notarlo y anotar mal.
 *
 *   El reloj avanza EXACTAMENTE un cuadro por tic y nunca descarta. Si la maquina o la red
 *   no llegan, la reproduccion se pone lenta, que es visible; saltear cuadros seria
 *   invisible y haria etiquetar sobre una version del video distinta de la que se exporta.
 *
 *   Los cuadros se cachean como Image ya decodificada. El servidor los manda con
 *   Cache-Control inmutable, asi que el cache del navegador evita el viaje y este Map evita
 *   volver a decodificar el JPEG.
 */

"use strict";

const SPEEDS = [0.10, 0.25, 0.50, 1.0];
const PREFETCH = 24;          // cuadros que se piden por adelantado en la direccion de marcha
const IMG_CACHE_MAX = 400;    // Images decodificadas en memoria
// El navegador permite ~6 conexiones por origen. El prefetch usa como mucho 3 y deja el
// resto libre para el cuadro que se esta mirando y para las poses. Sin este tope, la
// peticion del cuadro actual queda encolada detras de decenas de prefetch y la imagen se
// atrasa varios cuadros respecto del esqueleto.
const MAX_PREFETCH_VUELO = 3;
const POSE_WINDOW = 240;      // tiene que coincidir con MAX_POSE_RANGE del servidor
const ZOOM_MIN = 0.1, ZOOM_MAX = 12, ZOOM_STEP = 1.25;
const HIRES_ZOOM = 1.6;       // por encima de esto, y en pausa, se pide el original

const S = {
  meta: null,
  cursor: 0,
  speed: 0.25,
  direction: 1,
  playing: false,
  timer: null,
  zoom: 1, panX: 0, panY: 0, fit: true,
  imgs: new Map(),            // frame -> HTMLImageElement
  enVuelo: new Map(),         // frame -> Promise, para no volver a pedir lo que ya viaja
  cola: new Set(),            // frames encolados para prefetch
  prefetchVuelo: 0,
  poses: new Map(),           // frame -> [detecciones]
  poseReqs: new Set(),        // ventanas ya pedidas
  hiresFrame: null,           // frame del que tenemos el original cargado
  hiresImg: null,
  opts: { esqueleto: true, cajas: true, ids: true, guantes: true, hires: true,
          solo: "", umbral: 0.30 },
  events: [],
  seams: [],
  pendientes: 0,
};

const $ = (id) => document.getElementById(id);
const lienzo = $("lienzo"), ctx = lienzo.getContext("2d");
const tl = $("timeline"), tlctx = tl.getContext("2d");

/* ---------------------------------------------------------------- arranque */

async function main() {
  S.meta = await (await fetch("/api/meta")).json();
  S.seams = S.meta.seams || [];
  document.title = `${S.meta.video.name} — boxtwin`;
  $("titulo").textContent = S.meta.video.name;
  $("e-fps").textContent = `${S.meta.video.fps.toFixed(3)} fps nativo`;
  $("e-fuente").textContent = `fuente: ${S.meta.source.using_proxy ? "proxy" : "original"}`;
  S.opts.umbral = S.meta.settings.kp_score_threshold;
  $("r-umbral").value = S.opts.umbral;
  $("v-umbral").textContent = S.opts.umbral.toFixed(2);

  try {
    const a = await (await fetch("/api/annotation")).json();
    S.events = a.events || [];
  } catch (e) { S.events = []; }
  cargarIssues();

  tablaTeclas();
  conectarControles();
  window.addEventListener("resize", () => { redimensionar(); dibujar(); });
  redimensionar();
  await irA(0);
}

function redimensionar() {
  for (const [c, el] of [[lienzo, $("visor")], [tl, $("linea")]]) {
    const r = el.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(r.width * dpr);
    c.height = Math.round(r.height * dpr);
    c.style.width = r.width + "px";
    c.style.height = r.height + "px";
  }
}

/* ------------------------------------------------------------ navegacion */

async function irA(n) {
  const total = S.meta.video.total_frames;
  S.cursor = Math.max(0, Math.min(n | 0, total - 1));
  pedirPoses(S.cursor);
  prefetch();
  await asegurarCuadro(S.cursor);
  dibujar();
  actualizarEstado();
}

function paso(delta) { pausar(); irA(S.cursor + delta); }

function reproducir(dir) {
  S.direction = dir >= 0 ? 1 : -1;
  if (S.playing) return;
  S.playing = true;
  const intervalo = Math.max(1, Math.round(1000 / (S.meta.video.fps * S.speed)));
  // Un cuadro por tic, sin descartes. Si no se llega, se atrasa y se ve.
  S.timer = setInterval(() => {
    const siguiente = S.cursor + S.direction;
    if (siguiente < 0 || siguiente >= S.meta.video.total_frames) { pausar(); return; }
    irA(siguiente);
  }, intervalo);
  actualizarEstado();
}

function pausar() {
  if (!S.playing) return;
  clearInterval(S.timer); S.timer = null; S.playing = false;
  irA(S.cursor);  // al pausar puede corresponder subir a resolucion completa
  actualizarEstado();
}

function alternar(dir) { S.playing ? pausar() : reproducir(dir); }

function cambiarVelocidad(paso) {
  const i = (SPEEDS.indexOf(S.speed) + paso + SPEEDS.length) % SPEEDS.length;
  S.speed = SPEEDS[i];
  if (S.playing) { const d = S.direction; pausar(); reproducir(d); }
  actualizarEstado();
}

/* --------------------------------------------------------------- cuadros */

function urlCuadro(n) { return `/api/frame/${n}.jpg`; }

function cargarImagen(n, url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = url || urlCuadro(n);
  });
}

/*
 * Una sola peticion por cuadro, aunque se pida muchas veces.
 *
 * Sin este control, cada tic reencolaba los mismos cuadros porque todavia no habian
 * llegado a S.imgs, la cola de conexiones del navegador se llenaba de duplicados y la
 * peticion del cuadro actual quedaba al fondo: la imagen se atrasaba varios cuadros
 * respecto del esqueleto, que si estaba al dia porque las poses vienen en bloque.
 */
function pedirImagen(n, esPrefetch) {
  const listo = S.imgs.get(n);
  if (listo) return Promise.resolve(listo);
  const enCurso = S.enVuelo.get(n);
  if (enCurso) return enCurso;

  if (esPrefetch) S.prefetchVuelo++;
  const p = cargarImagen(n).then((img) => {
    S.enVuelo.delete(n);
    if (esPrefetch) S.prefetchVuelo--;
    if (img) { S.imgs.set(n, img); podarImagenes(); }
    bombear();
    return img;
  });
  S.enVuelo.set(n, p);
  return p;
}

function bombear() {
  while (S.prefetchVuelo < MAX_PREFETCH_VUELO && S.cola.size) {
    // Se atiende primero lo mas cercano al cursor en la direccion de marcha: es lo que se
    // va a necesitar antes.
    let mejor = null, mejorD = Infinity;
    for (const f of S.cola) {
      const rel = (f - S.cursor) * S.direction;
      const d = rel >= 0 ? rel : 10000 - rel;
      if (d < mejorD) { mejorD = d; mejor = f; }
    }
    S.cola.delete(mejor);
    if (!S.imgs.has(mejor) && !S.enVuelo.has(mejor)) pedirImagen(mejor, true);
  }
}

async function asegurarCuadro(n) {
  if (!S.imgs.has(n)) {
    S.pendientes++;
    $("cargando").classList.remove("oculto");
    // El cuadro actual no pasa por la cola: se pide ya y no espera al prefetch.
    await pedirImagen(n, false);
    S.pendientes--;
    if (S.pendientes === 0) $("cargando").classList.add("oculto");
  }
  // Resolucion completa solo en pausa y con zoom: decodificar el original cuesta y en
  // reproduccion no habria tiempo. Ampliar el proxy 4x deja el guante irreconocible.
  if (S.opts.hires && !S.playing && zoomEfectivo() >= HIRES_ZOOM &&
      S.meta.source.hires_available && S.hiresFrame !== n) {
    const img = await cargarImagen(n, `/api/hires/${n}.jpg`);
    if (img && S.cursor === n) { S.hiresImg = img; S.hiresFrame = n; dibujar(); }
  }
}

function podarImagenes() {
  // Desalojo por distancia al cursor y no por antiguedad: ir y venir sobre una frontera
  // desalojaria con LRU justo los cuadros que se estan mirando.
  while (S.imgs.size > IMG_CACHE_MAX) {
    let lejano = null, dist = -1;
    for (const f of S.imgs.keys()) {
      const d = Math.abs(f - S.cursor);
      if (d > dist && f !== S.cursor) { dist = d; lejano = f; }
    }
    if (lejano === null) break;
    S.imgs.delete(lejano);
  }
}

function prefetch() {
  const total = S.meta.video.total_frames;
  // Un salto o un cambio de direccion dejan la cola llena de cuadros que ya no sirven y
  // que competirian con los que si.
  for (const f of S.cola) {
    if (Math.abs(f - S.cursor) > PREFETCH * 2) S.cola.delete(f);
  }
  for (let i = 1; i <= PREFETCH; i++) {
    const f = S.cursor + i * S.direction;
    if (f < 0 || f >= total) continue;
    if (S.imgs.has(f) || S.enVuelo.has(f)) continue;
    S.cola.add(f);
  }
  bombear();
}

/* ----------------------------------------------------------------- poses */

function pedirPoses(n) {
  const inicio = Math.max(0, Math.floor(n / POSE_WINDOW) * POSE_WINDOW);
  for (const base of [inicio, inicio + POSE_WINDOW, inicio - POSE_WINDOW]) {
    if (base < 0 || base >= S.meta.video.total_frames) continue;
    if (S.poseReqs.has(base)) continue;
    S.poseReqs.add(base);
    fetch(`/api/poses?from=${base}&to=${base + POSE_WINDOW}`)
      .then((r) => r.json())
      .then((d) => {
        d.frames.forEach((dets, i) => S.poses.set(d.from + i, dets));
        if (S.cursor >= d.from && S.cursor < d.to) dibujar();
      })
      .catch(() => S.poseReqs.delete(base));
  }
}

/* ------------------------------------------------------------ coordenadas */

function zoomAjuste() {
  return Math.min(lienzo.width / S.meta.video.width, lienzo.height / S.meta.video.height);
}
function zoomEfectivo() { return S.fit ? zoomAjuste() : S.zoom; }

function offset() {
  const z = zoomEfectivo();
  return {
    x: Math.max(0, (lienzo.width - S.meta.video.width * z) / 2),
    y: Math.max(0, (lienzo.height - S.meta.video.height * z) / 2),
  };
}

function videoAPantalla(x, y) {
  const z = zoomEfectivo(), o = offset();
  return { x: (x - S.panX) * z + o.x, y: (y - S.panY) * z + o.y };
}
function pantallaAVideo(x, y) {
  const z = zoomEfectivo(), o = offset();
  return { x: (x - o.x) / z + S.panX, y: (y - o.y) / z + S.panY };
}

function limitarPan() {
  const z = zoomEfectivo();
  S.panX = Math.min(Math.max(0, S.panX), Math.max(0, S.meta.video.width - lienzo.width / z));
  S.panY = Math.min(Math.max(0, S.panY), Math.max(0, S.meta.video.height - lienzo.height / z));
}

function fijarZoom(z, anclaX, anclaY) {
  const nuevo = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, z));
  if (anclaX === undefined) { anclaX = lienzo.width / 2; anclaY = lienzo.height / 2; }
  const antes = pantallaAVideo(anclaX, anclaY);
  S.fit = false; S.zoom = nuevo;
  const despues = pantallaAVideo(anclaX, anclaY);
  S.panX += antes.x - despues.x;
  S.panY += antes.y - despues.y;
  limitarPan();
  asegurarCuadro(S.cursor);
  dibujar(); actualizarEstado();
}

function ajustar() { S.fit = true; S.panX = S.panY = 0; dibujar(); actualizarEstado(); }

/* ---------------------------------------------------------------- dibujo */

function dibujar() {
  const m = S.meta;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, lienzo.width, lienzo.height);

  const z = zoomEfectivo(), o = offset();
  ctx.setTransform(z, 0, 0, z, o.x - S.panX * z, o.y - S.panY * z);

  const usarHires = S.opts.hires && !S.playing && z >= HIRES_ZOOM && S.hiresFrame === S.cursor;
  const img = usarHires ? S.hiresImg : S.imgs.get(S.cursor);
  if (img) {
    // Siempre estirada al tamano del video ORIGINAL: por eso cambiar de fuente no mueve
    // ni un keypoint.
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(img, 0, 0, m.video.width, m.video.height);
  }
  $("e-fuente").textContent = `fuente: ${usarHires ? "original" : (m.source.using_proxy ? "proxy" : "original")}`;

  pintarPoses(z);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  pintarTimeline();
}

function colorRol(rol) {
  const c = S.meta.colors[rol || "unassigned"] || S.meta.colors.unassigned;
  return c;
}
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

function pintarPoses(z) {
  const dets = S.poses.get(S.cursor);
  // Deja constancia de que cuadro son las poses efectivamente dibujadas. Con --stamp la
  // imagen lleva el suyo escrito encima, asi que una captura basta para saber si los dos
  // coinciden en vez de discutirlo de memoria.
  S.posesDibujadas = dets ? S.cursor : null;
  if (!dets) return;
  const grosor = Math.max(0.5, 2 / z);
  const radio = Math.max(0.8, 3 / z);
  const k = S.meta.settings.glove_extrapolation_k;
  const umbral = S.opts.umbral;

  for (const d of dets) {
    if (S.opts.solo && d.role !== S.opts.solo) continue;
    const base = colorRol(d.role);
    const alfa = (d.shadowed || !d.reliable) ? 0.55 : 1.0;

    // Guantes derivados: extrapolacion sobre el antebrazo. Es funcion determinista del
    // codo y la muneca, asi que no agrega informacion a un modelo; sirve para juzgar a ojo
    // si el golpe llego, que es lo que se esta anotando.
    const xy = d.kp.slice(), ks = d.ks.slice();
    if (S.opts.guantes) {
      for (const [codo, muneca] of [[7, 9], [8, 10]]) {
        const wx = xy[muneca * 2], wy = xy[muneca * 2 + 1];
        const ex = xy[codo * 2], ey = xy[codo * 2 + 1];
        xy.push(wx + k * (wx - ex), wy + k * (wy - ey));
        ks.push(Math.min(ks[muneca], ks[codo]));
      }
    }

    if (S.opts.cajas) {
      ctx.strokeStyle = rgba(base, alfa);
      ctx.lineWidth = grosor;
      ctx.setLineDash((d.shadowed || !d.reliable) ? [grosor * 3, grosor * 2] : []);
      ctx.strokeRect(d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]);
      ctx.setLineDash([]);
    }

    if (S.opts.esqueleto) {
      const aristas = S.meta.skeleton.edges.concat(S.opts.guantes ? S.meta.skeleton.glove_edges : []);
      ctx.lineWidth = grosor;
      for (const [a, b] of aristas) {
        if (a * 2 + 1 >= xy.length || b * 2 + 1 >= xy.length) continue;
        // Los de baja confianza se atenuan, no se ocultan: una pose mala y una pose
        // incompleta son cosas distintas y se corrigen distinto.
        const fuerte = ks[a] >= umbral && ks[b] >= umbral;
        ctx.strokeStyle = rgba(base, fuerte ? alfa : alfa * 0.28);
        ctx.beginPath();
        ctx.moveTo(xy[a * 2], xy[a * 2 + 1]);
        ctx.lineTo(xy[b * 2], xy[b * 2 + 1]);
        ctx.stroke();
      }
      for (let i = 0; i < ks.length; i++) {
        ctx.fillStyle = rgba(base, ks[i] >= umbral ? alfa : alfa * 0.28);
        ctx.beginPath();
        ctx.arc(xy[i * 2], xy[i * 2 + 1], radio, 0, 6.2832);
        ctx.fill();
      }
    }

    if (S.opts.ids) {
      let txt = `${d.role || "sin asignar"} (#${d.track_id})`;
      if (d.manual) txt += " manual";
      if (d.shadowed) txt += " duplicado";
      if (!d.reliable) txt += " no confiable";
      ctx.fillStyle = rgba(base, alfa);
      ctx.font = `${Math.max(3, 13 / z)}px system-ui, sans-serif`;
      ctx.fillText(txt, d.bbox[0], d.bbox[1] - 4 / z);
    }
  }
}

/* -------------------------------------------------------------- timeline */

function pintarTimeline() {
  const total = S.meta.video.total_frames;
  const W = tl.width, H = tl.height;
  const alto = Math.floor((H - 6) / 2);
  const x = (f) => f / total * (W - 1);

  tlctx.setTransform(1, 0, 0, 1, 0, 0);
  tlctx.fillStyle = "#0e0e10";
  tlctx.fillRect(0, 0, W, H);

  const carriles = { fighter_A: 2, fighter_B: 4 + alto };
  for (const [rol, y] of Object.entries(carriles)) {
    tlctx.fillStyle = "#26262b";
    tlctx.fillRect(0, y, W, alto);
    // Franja de color a la izquierda: identifica el carril sin necesidad de leyenda.
    tlctx.fillStyle = rgba(S.meta.colors[rol], 1);
    tlctx.fillRect(0, y, 3, alto);
  }

  for (const ev of S.events) {
    const y = carriles[ev.fighter];
    if (y === undefined) continue;
    tlctx.fillStyle = rgba(S.meta.colors.punch[ev.punch_type] || [200, 200, 200], 1);
    tlctx.fillRect(x(ev.start_frame), y + 2, Math.max(2, x(ev.end_frame + 1) - x(ev.start_frame)), alto - 4);
  }

  // Costuras del preproceso: ahi el tracker se reinicio y hay corte de identidad seguro.
  tlctx.strokeStyle = rgba(S.meta.colors.unassigned, 0.9);
  tlctx.setLineDash([3, 3]);
  for (const f of S.seams) {
    tlctx.beginPath(); tlctx.moveTo(x(f), 0); tlctx.lineTo(x(f), H); tlctx.stroke();
  }
  tlctx.setLineDash([]);

  tlctx.strokeStyle = "#fff";
  tlctx.lineWidth = 1;
  tlctx.beginPath();
  tlctx.moveTo(x(S.cursor), 0); tlctx.lineTo(x(S.cursor), H); tlctx.stroke();
}

/* ---------------------------------------------------------------- estado */

function actualizarEstado() {
  const m = S.meta, t = S.cursor / m.video.fps;
  $("e-cuadro").textContent = `cuadro ${S.cursor} / ${m.video.total_frames - 1}`;
  $("e-tiempo").textContent =
    `${String(Math.floor(t / 60)).padStart(2, "0")}:${(t % 60).toFixed(3).padStart(6, "0")}`;
  $("e-play").textContent =
    `${S.playing ? "▶" : "❚❚"}${S.direction > 0 ? "→" : "←"} ${S.speed}x`;
  $("e-zoom").textContent = `zoom ${zoomEfectivo().toFixed(2)}x`;
  $("e-red").textContent = `poses ${S.posesDibujadas ?? "-"} · ${S.imgs.size} en cache · ${S.enVuelo.size} en vuelo · ${S.cola.size} en cola`;
}

async function cargarIssues() {
  try {
    const d = await (await fetch("/api/issues")).json();
    const c = $("issues");
    if (!d.issues.length) { c.textContent = "sin observaciones"; return; }
    c.innerHTML = d.issues.slice(0, 40).map((i) =>
      `<div class="issue ${i.level}"><b>${i.code}</b> ${i.ref || ""}<br>${i.message}</div>`
    ).join("");
  } catch (e) { /* la validacion es informativa: si falla no bloquea anotar */ }
}

/* ---------------------------------------------------------------- teclas */

const TECLAS = {
  " ":            () => alternar(1),
  "shift+ ":      () => alternar(-1),
  "arrowright":   () => paso(1),
  "arrowleft":    () => paso(-1),
  "shift+arrowright": () => paso(5),
  "shift+arrowleft":  () => paso(-5),
  "ctrl+arrowright":  () => paso(Math.round(S.meta.video.fps)),
  "ctrl+arrowleft":   () => paso(-Math.round(S.meta.video.fps)),
  "home":         () => { pausar(); irA(0); },
  "end":          () => { pausar(); irA(S.meta.video.total_frames - 1); },
  "g":            () => { const n = prompt("Ir al cuadro:", S.cursor); if (n !== null) { pausar(); irA(parseInt(n, 10) || 0); } },
  "+":            () => cambiarVelocidad(1),
  "-":            () => cambiarVelocidad(-1),
  "k":            () => alternarOpcion("esqueleto", "t-esqueleto"),
  "x":            () => alternarOpcion("cajas", "t-cajas"),
  "i":            () => alternarOpcion("ids", "t-ids"),
  "l":            () => alternarOpcion("guantes", "t-guantes"),
  "ctrl++":       () => fijarZoom(zoomEfectivo() * ZOOM_STEP),
  "ctrl+-":       () => fijarZoom(zoomEfectivo() / ZOOM_STEP),
  "ctrl+0":       () => ajustar(),
};

const AYUDA = [
  ["Espacio", "reproducir / pausar"],
  ["Shift+Espacio", "reproducir hacia atrás"],
  ["← →", "±1 cuadro"],
  ["Shift+← →", "±5 cuadros"],
  ["Ctrl+← →", "±1 segundo"],
  ["Inicio / Fin", "principio / final"],
  ["G", "ir al cuadro"],
  ["+ / -", "velocidad"],
  ["Ctrl + / Ctrl -", "zoom"],
  ["Ctrl+0", "ajustar a la ventana"],
  ["rueda", "zoom sobre el puntero"],
  ["arrastrar", "desplazar"],
  ["K X I L", "esqueleto, cajas, IDs, guantes"],
];

function tablaTeclas() {
  $("tabla-teclas").innerHTML = AYUDA.map(([k, d]) => `<tr><td>${k}</td><td>${d}</td></tr>`).join("");
}

function alternarOpcion(clave, id) {
  S.opts[clave] = !S.opts[clave];
  $(id).checked = S.opts[clave];
  dibujar();
}

function conectarControles() {
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    let k = e.key.toLowerCase();
    if (e.ctrlKey) k = "ctrl+" + k;
    else if (e.shiftKey && k.length > 1) k = "shift+" + k;
    else if (e.shiftKey && k === " ") k = "shift+ ";
    const fn = TECLAS[k];
    if (fn) { e.preventDefault(); fn(); }
  });

  for (const [clave, id] of Object.entries({
    esqueleto: "t-esqueleto", cajas: "t-cajas", ids: "t-ids",
    guantes: "t-guantes", hires: "t-hires",
  })) {
    $(id).addEventListener("change", (e) => {
      S.opts[clave] = e.target.checked;
      if (clave === "hires") { S.hiresFrame = null; asegurarCuadro(S.cursor); }
      dibujar();
    });
  }
  $("s-solo").addEventListener("change", (e) => { S.opts.solo = e.target.value; dibujar(); });
  $("r-umbral").addEventListener("input", (e) => {
    S.opts.umbral = parseFloat(e.target.value);
    $("v-umbral").textContent = S.opts.umbral.toFixed(2);
    dibujar();
  });

  lienzo.addEventListener("wheel", (e) => {
    e.preventDefault();
    const dpr = window.devicePixelRatio || 1;
    const r = lienzo.getBoundingClientRect();
    fijarZoom(zoomEfectivo() * Math.pow(ZOOM_STEP, -Math.sign(e.deltaY)),
              (e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr);
  }, { passive: false });

  let arrastrando = false, ultimo = null;
  lienzo.addEventListener("mousedown", (e) => { arrastrando = true; ultimo = [e.clientX, e.clientY]; });
  window.addEventListener("mouseup", () => { arrastrando = false; });
  window.addEventListener("mousemove", (e) => {
    if (!arrastrando) return;
    const z = zoomEfectivo(), dpr = window.devicePixelRatio || 1;
    S.panX -= (e.clientX - ultimo[0]) * dpr / z;
    S.panY -= (e.clientY - ultimo[1]) * dpr / z;
    ultimo = [e.clientX, e.clientY];
    S.fit = false; limitarPan(); dibujar(); actualizarEstado();
  });
  lienzo.addEventListener("dblclick", () => ajustar());

  const saltar = (e) => {
    const r = tl.getBoundingClientRect();
    pausar();
    irA(Math.round((e.clientX - r.left) / r.width * S.meta.video.total_frames));
  };
  tl.addEventListener("mousedown", saltar);
  tl.addEventListener("mousemove", (e) => { if (e.buttons & 1) saltar(e); });
}

main();
