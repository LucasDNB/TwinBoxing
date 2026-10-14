/*
 * La API simulada con MSW (06 §1): respuestas copiadas de una sesion real, en un "mundo" que
 * cada prueba puede hacer avanzar de estado. No depende de la GPU ni del worker.
 */
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import fc01 from "../fixtures/sesion-real/fightcard-0.1.json";
import fc02 from "../fixtures/sesion-real/fightcard-0.2.json";
import fc03 from "../fixtures/sesion-real/fightcard-0.3.json";
import jobEnCola from "../fixtures/sesion-real/job-en_cola.json";
import jobProcesando from "../fixtures/sesion-real/job-procesando.json";
import jobEspera from "../fixtures/sesion-real/job-espera_siembra.json";
import jobCompletando from "../fixtures/sesion-real/job-completando.json";
import jobListo from "../fixtures/sesion-real/job-listo.json";
import jobFallo from "../fixtures/sesion-real/job-fallo.json";

export const ID = jobListo.id;
export const JOBS = {
  en_cola: jobEnCola, procesando: jobProcesando, espera_siembra: jobEspera,
  completando: jobCompletando, listo: jobListo, fallo: jobFallo,
} as const;
export const FIGHTCARDS = { "0.1": fc01, "0.2": fc02, "0.3": fc03 };

export function productoBase(banderas: Partial<Record<string, boolean>> = {}) {
  return {
    registro: "invitacion",
    max_bytes: 6 * 1024 ** 3,
    max_minutos: 30,
    formatos: [".mp4", ".mov", ".m4v"],
    factor_procesamiento: { min: 1.134, mediana: 1.447, max: 2.624, medido_sobre: "campaña de costo del Cap 5" },
    criterios: { C2: { estado: "pendiente", fecha: null }, C3: { estado: "pendiente", fecha: null } },
    banderas: { comparacion_rounds: false, guardia: false, sesiones_individuales: false, postura_2d: false, ...banderas },
    dominio: { combinacion: { patron: "cualquiera", separacion: "inicio_a_inicio", max_s: 0.5, min_golpes: 2 }, perfil_preliminar_sesiones: 5 },
  };
}

export interface Mundo {
  producto: ReturnType<typeof productoBase>;
  job: Record<string, unknown>;
  fightcard: Record<string, unknown> | null;
  boxeadores: Record<string, unknown>[];
  pedidos: { metodo: string; ruta: string; cuerpo?: unknown; auth?: string | null }[];
  corregir: "ok" | "422";
  tokenValido: string;
  tramos: Record<string, unknown> | null;
  perfiles: Record<string, Record<string, unknown>>;
  /* GET /rival (B18): lo que devuelve la API, armado a mano en cada prueba. */
  rival: Record<string, unknown> | null;
}

/* Como el servidor con {"objetivo"} (B11): pisa la altura y recuenta cabeza y cuerpo. */
function corregirAltura(golpe: string, objetivo: "cabeza" | "cuerpo") {
  type G = { id: string; t_inicio: number; objetivo?: string | null; objetivo_corregido?: unknown; objetivo_confianza?: number | null };
  type C = { total: number; cabeza?: number; cuerpo?: number; objetivo_sin_dato?: number; inicio_s?: number; fin_s?: number };
  const fc = mundo.fightcard as { peleadores: Record<string, { golpes: G[]; total: C; por_round: C[] }> } | null;
  let ev: G | undefined;
  const tres = (gs: G[]) => {
    const cabeza = gs.filter((g) => g.objetivo === "cabeza").length;
    const cuerpo = gs.filter((g) => g.objetivo === "cuerpo").length;
    return { cabeza, cuerpo, objetivo_sin_dato: gs.length - cabeza - cuerpo };
  };
  for (const p of Object.values(fc?.peleadores ?? {})) {
    const g = p.golpes.find((x) => x.id === golpe);
    if (!g) continue;
    g.objetivo_corregido = { objetivo, objetivo_original: g.objetivo ?? null };
    g.objetivo = objetivo;
    g.objetivo_confianza = null;
    ev = g;
    Object.assign(p.total, tres(p.golpes));
    for (const r of p.por_round) Object.assign(r, tres(p.golpes.filter((x) => r.inicio_s! <= x.t_inicio && x.t_inicio < r.fin_s!)));
  }
  return HttpResponse.json(ev ?? { id: golpe, objetivo });
}

export function mundoNuevo(): Mundo {
  return {
    producto: productoBase(),
    job: structuredClone(jobListo),
    fightcard: structuredClone(fc03),
    boxeadores: [],
    pedidos: [],
    corregir: "ok",
    tokenValido: "token-bueno",
    perfiles: {},
    rival: null,
    tramos: {
      cobertura: { A: 0.7948, B: 0.9339, sin_asignar: 0.8554 },
      separacion_color: 1.375,
      tramos: [
        { track: 228, t_desde: 89.5, t_hasta: 104.0, motivo: "color_insuficiente", recorte_url: `/jobs/${jobListo.id}/tramos/228/recorte`, asignado: null },
        { track: 430, t_desde: 139.2, t_hasta: 142.0, motivo: "sin_coexistencia", recorte_url: null, asignado: null },
      ],
    },
  };
}

export let mundo = mundoNuevo();
export function reiniciarMundo() {
  mundo = mundoNuevo();
  return mundo;
}

function autorizado(req: Request) {
  return req.headers.get("authorization") === `Bearer ${mundo.tokenValido}`;
}

function registrar(req: Request, cuerpo?: unknown) {
  const u = new URL(req.url);
  mundo.pedidos.push({ metodo: req.method, ruta: u.pathname + u.search, cuerpo, auth: req.headers.get("authorization") });
}

const no401 = () => HttpResponse.json({ detail: "hace falta iniciar sesion" }, { status: 401 });

export const servidor = setupServer(
  http.get("*/producto", ({ request }) => {
    registrar(request);
    return HttpResponse.json(mundo.producto);
  }),
  http.post("*/auth/login", async ({ request }) => {
    const c = (await request.json()) as { email: string; clave: string };
    registrar(request, c);
    if (c.clave !== "sparring2026") return HttpResponse.json({ detail: "email o clave incorrectos" }, { status: 401 });
    return HttpResponse.json({ token: mundo.tokenValido, tipo: "bearer", usuario: c.email });
  }),
  http.post("*/auth/registro", async ({ request }) => {
    const c = await request.json();
    registrar(request, c);
    return HttpResponse.json({ detail: "el codigo de invitacion no es correcto" }, { status: 403 });
  }),
  http.get("*/jobs", ({ request }) => {
    registrar(request);
    if (!autorizado(request)) return no401();
    return HttpResponse.json([mundo.job]);
  }),
  http.get("*/jobs/:id", ({ request }) => {
    registrar(request);
    if (!autorizado(request)) return no401();
    return HttpResponse.json(mundo.job);
  }),
  http.post("*/jobs/:id/siembra", async ({ request }) => {
    const c = await request.json();
    registrar(request, c);
    if (!autorizado(request)) return no401();
    mundo.job = { ...mundo.job, estado: "en_cola", siembra: c };
    return HttpResponse.json({ job_id: ID, estado: "en_cola" }, { status: 202 });
  }),
  http.get("*/fightcards/:id", ({ request }) => {
    registrar(request);
    if (!autorizado(request)) return no401();
    if (!mundo.fightcard) return HttpResponse.json({ detail: "la Fight-Card todavia no esta lista" }, { status: 409 });
    return HttpResponse.json(mundo.fightcard);
  }),
  http.patch("*/fightcards/:id/golpes/:golpe", async ({ request, params }) => {
    const c = (await request.json()) as { tipo?: string; objetivo?: "cabeza" | "cuerpo" };
    registrar(request, c);
    if (!autorizado(request)) return no401();
    if (mundo.corregir === "422") return HttpResponse.json({ detail: "tipo desconocido" }, { status: 422 });
    if (c.objetivo) return corregirAltura(String(params.golpe), c.objetivo);
    // Como el servidor: la correccion queda en el documento (sobrevive a recargar).
    type G = { id: string; tipo: string | null; corregido: unknown; descartado?: boolean; confianza_tipo: number | null };
    const fc = mundo.fightcard as { peleadores: Record<string, { golpes: G[]; descartados?: number }> } | null;
    let ev: G | undefined;
    for (const p of Object.values(fc?.peleadores ?? {})) {
      for (const g of p.golpes) {
        if (g.id !== params.golpe) continue;
        g.corregido = { tipo: c.tipo, tipo_original: g.tipo };
        g.confianza_tipo = null;
        g.descartado = c.tipo === "no_golpe";
        if (c.tipo && c.tipo !== "no_golpe") g.tipo = c.tipo;
        ev = g;
      }
      p.descartados = p.golpes.filter((g) => g.descartado).length;
    }
    return HttpResponse.json(ev ?? { id: params.golpe, tipo: c.tipo });
  }),
  http.get("*/videos/:id/ticket", ({ request }) => {
    registrar(request);
    if (!autorizado(request)) return no401();
    const n = mundo.pedidos.filter((p) => p.ruta.includes("/ticket")).length;
    return HttpResponse.json({ ticket: `ticket-${n}`, minutos: 30 });
  }),
  http.head("*/videos/:id/procesado", ({ request }) => {
    registrar(request);
    return new HttpResponse(null, { status: 404 });
  }),
  http.get("*/boxeadores", ({ request }) => {
    registrar(request);
    if (!autorizado(request)) return no401();
    return HttpResponse.json(mundo.boxeadores);
  }),
  http.get("*/boxeadores/:id/perfil", ({ request, params }) => {
    registrar(request);
    if (!autorizado(request)) return no401();
    const p = mundo.perfiles[String(params.id)];
    if (!p) return HttpResponse.json({ detail: "no existe ese boxeador" }, { status: 404 });
    return HttpResponse.json(p);
  }),
  http.get("*/rival", ({ request }) => {
    registrar(request);
    if (!autorizado(request)) return no401();
    const sesgo = { recall_cabeza: 0.512, recall_cuerpo: 0.348, corregido: true };
    return HttpResponse.json(mundo.rival ?? {
      cruces: [], lecturas: [], faltan: null, sesgo_detector: sesgo,
      base: { A: { ataque: 0, exposicion: 0 }, B: { ataque: 0, exposicion: 0 } },
    });
  }),
  http.post("*/boxeadores", async ({ request }) => {
    const c = (await request.json()) as Record<string, unknown>;
    registrar(request, c);
    if (!autorizado(request)) return no401();
    const b = { id: `b${mundo.boxeadores.length + 1}`, notas: null, creado: "2026-10-13T10:00:00", sesiones: 0, ultima_sesion: null, categoria: null, altura_m: null, envergadura_m: null, guardia: null, ...c };
    mundo.boxeadores.push(b);
    return HttpResponse.json(b, { status: 201 });
  }),
  http.patch("*/boxeadores/:id", async ({ request, params }) => {
    const c = (await request.json()) as Record<string, unknown>;
    registrar(request, c);
    const i = mundo.boxeadores.findIndex((b) => b.id === params.id);
    mundo.boxeadores[i] = { ...mundo.boxeadores[i], ...c };
    return HttpResponse.json(mundo.boxeadores[i]);
  }),
  http.get("*/panel", ({ request }) => {
    registrar(request);
    if (!autorizado(request)) return no401();
    return HttpResponse.json({
      peleadores_activos: 0, peleadores_preliminares: 0,
      sesiones_mes: { total: 1, listo: mundo.job.estado === "listo" ? 1 : 0, en_curso: ["en_cola", "procesando", "completando"].includes(String(mundo.job.estado)) ? 1 : 0, espera_siembra: mundo.job.estado === "espera_siembra" ? 1 : 0, fallo: mundo.job.estado === "fallo" ? 1 : 0 },
      rounds_mes: { total: 1, sparring: 1 },
      golpes_detectados_mes: 83,
      recall_detector: 0.484,
      pendientes: mundo.job.estado === "espera_siembra" ? [{ sesion_id: ID, estado: "espera_siembra", titulo: "Sparring del martes" }] : [],
    });
  }),
  http.get("*/jobs/:id/tramos", ({ request }) => {
    registrar(request);
    if (!autorizado(request)) return no401();
    if (!mundo.tramos) return HttpResponse.json({ detail: "esta sesion se proceso antes de la revision de tramos; reprocesala para revisarlos" }, { status: 409 });
    return HttpResponse.json(mundo.tramos);
  }),
  http.put("*/jobs/:id/tramos", async ({ request }) => {
    const c = await request.json();
    registrar(request, c);
    if (!autorizado(request)) return no401();
    mundo.job = { ...mundo.job, estado: "en_cola" };
    return HttpResponse.json({ job_id: ID, estado: "en_cola" }, { status: 202 });
  }),
  http.put("*/sesiones/:id/boxeadores", async ({ request }) => {
    const c = await request.json();
    registrar(request, c);
    return HttpResponse.json({ sesion: ID });
  }),
);
