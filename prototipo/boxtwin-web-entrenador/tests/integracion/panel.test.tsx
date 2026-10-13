/*
 * W4: CA-01.1 y CA-01.3 (04 §01). CA-01.2 lo verifica scripts/sin-demo.mjs sobre out/.
 */
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, screen, waitFor, within } from "@testing-library/react";
import { conSesion, enRuta, montar } from "./entorno";
import { JOBS, mundo, productoBase, servidor } from "./api-simulada";
import { http, HttpResponse } from "msw";
import { PantallaPanel } from "@/components/pantallas/panel/PantallaPanel";
import { PantallaSesiones } from "@/components/pantallas/sesiones/PantallaSesiones";

beforeEach(() => {
  conSesion();
  enRuta("/");
});

afterEach(() => vi.useRealTimers());

const boxeador = {
  id: "b1", nombre: "Ana Paz", guardia: "ortodoxa", notas: null, creado: "2026-09-01T00:00:00",
  categoria: "Welter", altura_m: null, envergadura_m: null, sesiones: 3, ultima_sesion: "2026-09-21T10:00:00",
};

function perfilCon(caida: number | null) {
  return {
    boxeador: { id: "b1", nombre: "Ana Paz", guardia: "ortodoxa", notas: null },
    sesiones: [], totales: { sesiones: 3, golpes: 0, mezcla: {}, brazos: {}, sin_clasificar: 0, golpes_por_minuto_medio: null },
    evolucion: [], avisos: [],
    por_contexto: [{ contexto: "sparring", golpes_por_minuto: 12, jab_fraccion: null, guardia_opuesta_arriba: null, caida_ultimo_round: caida }],
  };
}

describe("01 · Panel", () => {
  it("CA-01.1: una sesion que pasa a espera_siembra genera el aviso sin recargar, en menos de 15 s", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    mundo.job = structuredClone(JOBS.procesando);
    montar(<PantallaPanel />);
    expect((await screen.findAllByText(/Procesando · pose · 64%/)).length).toBeGreaterThan(0);
    expect(screen.queryByText("falta un paso")).toBeNull();

    mundo.job = structuredClone(JOBS.espera_siembra);
    await act(async () => { await vi.advanceTimersByTimeAsync(10_500); });
    await waitFor(() => expect(screen.getByText("falta un paso")).toBeInTheDocument());
    const aviso = screen.getByText("falta un paso").closest("a")!;
    expect(aviso.getAttribute("href")).toBe(`/sesion/identificar/?id=${JOBS.espera_siembra.id}`);
  });

  it("los indicadores salen de /panel, con el margen del recall", async () => {
    montar(<PantallaPanel />);
    expect(await screen.findByText("Golpes detectados")).toBeInTheDocument();
    expect(screen.getByText("83")).toBeInTheDocument();
    expect(screen.getByText("El sistema encuentra cerca de 5 de cada 10 golpes: el total real es mayor.")).toBeInTheDocument();
  });

  it("un usuario sin sesiones ve 'Subí tu primer sparring' en vez de indicadores", async () => {
    servidor.use(http.get("*/jobs", () => HttpResponse.json([])));
    montar(<PantallaPanel />);
    expect(await screen.findByText("Subí tu primer sparring")).toBeInTheDocument();
    expect(screen.queryByText("Golpes detectados")).toBeNull();
    expect(screen.getByText("Cargá a tus peleadores para armar sus perfiles")).toBeInTheDocument();
  });

  it("CA-01.3: con C2 apagado no aparece ninguna frase de tendencia entre rounds", async () => {
    mundo.boxeadores = [boxeador];
    servidor.use(http.get("*/boxeadores/:id/perfil", () => HttpResponse.json(perfilCon(-0.3))));
    montar(<PantallaPanel />);
    const lista = await screen.findByRole("heading", { name: "Mis peleadores" });
    const seccion = lista.closest("section")!;
    expect(await within(seccion).findByText("última sesión 21/09")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/el volumen (cae|sube)/);
    expect(mundo.pedidos.some((p) => p.ruta.includes("/perfil"))).toBe(false);
  });

  it("con C2 aprobado la tendencia sale del perfil (B7)", async () => {
    mundo.producto = productoBase({ comparacion_rounds: true });
    mundo.boxeadores = [boxeador];
    servidor.use(http.get("*/boxeadores/:id/perfil", () => HttpResponse.json(perfilCon(-0.3))));
    montar(<PantallaPanel />);
    expect(await screen.findByText("el volumen cae 30% en el último round")).toBeInTheDocument();
  });

  it("una sesion en fallo muestra el error y la accion", async () => {
    mundo.job = structuredClone(JOBS.fallo);
    montar(<PantallaPanel />);
    const alerta = await screen.findByText("falló");
    expect(alerta.closest("[role=alert]")).toHaveTextContent("la etapa de pose se quedo sin memoria de GPU");
    expect(screen.getByRole("link", { name: "Subir de nuevo" })).toBeInTheDocument();
  });
});

describe("Lista de sesiones", () => {
  it("filtra por estado desde la query y busca por nombre", async () => {
    enRuta("/sesiones/?estado=fallo");
    montar(<PantallaSesiones />);
    expect(await screen.findByText("Ninguna sesión coincide con el filtro")).toBeInTheDocument();
  });

  it("sin filtro muestra la sesion con su accion", async () => {
    enRuta("/sesiones/");
    montar(<PantallaSesiones />);
    expect(await screen.findByText("Ver Fight-Card")).toBeInTheDocument();
  });
});
