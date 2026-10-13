/*
 * I1 (06 §1): subir -> espera -> identificar -> Fight-Card, con la API simulada avanzando de
 * estado. Cruza tres pantallas con un ruteador minimo que mira usePathname, como el App
 * Router en la exportacion estatica.
 */
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { conSesion, enRuta, montar } from "./entorno";
// Despues de ./entorno: next/navigation tiene que llegar ya simulado.
import { usePathname } from "next/navigation";
import { FIGHTCARDS, ID, JOBS, mundo } from "./api-simulada";
import { videoSimulado } from "./video";

const subidas: { avanzar: (e: number, t: number) => void; resolver: (r: unknown) => void }[] = [];
vi.mock("@/lib/api/cliente", async (orig) => {
  const real = await orig<typeof import("@/lib/api/cliente")>();
  return {
    ...real,
    subirConProgreso: (_d: FormData, avanzar: (e: number, t: number) => void) => {
      let resolver!: (r: unknown) => void;
      const promesa = new Promise((a) => { resolver = a; });
      subidas.push({ avanzar, resolver });
      return { promesa, cancelar: () => {} };
    },
  };
});
vi.mock("@/lib/subida", async (orig) => {
  const real = await orig<typeof import("@/lib/subida")>();
  return { ...real, leerMetadatos: async () => ({ duracion_s: 174.9, ancho: 1920, alto: 1080 }) };
});

import { PantallaNueva } from "@/components/pantallas/nueva/PantallaNueva";
import { PantallaSesion } from "@/components/pantallas/sesion/PantallaSesion";
import { PantallaIdentificar } from "@/components/pantallas/identificar/PantallaIdentificar";

function App() {
  const ruta = usePathname();
  if (ruta.startsWith("/sesiones/nueva")) return <PantallaNueva />;
  if (ruta.startsWith("/sesion/identificar")) return <PantallaIdentificar />;
  if (ruta.startsWith("/sesion")) return <PantallaSesion />;
  return null;
}

beforeEach(() => {
  subidas.length = 0;
  conSesion();
  videoSimulado();
});
afterEach(() => vi.useRealTimers());

async function avanzar(ms: number) {
  await act(async () => { await vi.advanceTimersByTimeAsync(ms); });
}

describe("I1 · de la subida a la Fight-Card", () => {
  it("sube con progreso, sondea, identifica y termina en la Fight-Card con los totales del documento", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    enRuta("/sesiones/nueva/");
    montar(<App />);

    // 1. Subir
    await screen.findByText("Qué se filmó");
    const f = new File(["x"], "sparring.mp4", { type: "video/mp4" });
    fireEvent.change(screen.getByTestId("input-galeria"), { target: { files: [f] } });
    await screen.findByText(/sparring\.mp4 ·/);
    await userEvent.click(screen.getAllByLabelText(/dieron su consentimiento/)[0]);
    await userEvent.click(screen.getByRole("button", { name: "Subir y procesar" }));
    act(() => subidas[0].avanzar(64, 100));
    expect(await screen.findByText("Subiendo · 64%")).toBeInTheDocument();
    mundo.job = structuredClone(JOBS.procesando);
    await act(async () => subidas[0].resolver({ job_id: ID, estado: "en_cola", bytes: 100 }));

    // 2. Espera: la sesion se sondea sola
    expect(await screen.findByText("Buscando a las personas y su pose.")).toBeInTheDocument();
    mundo.job = structuredClone(JOBS.espera_siembra);
    await avanzar(3_500);
    const ir = await screen.findByRole("link", { name: "Marcar peleadores" });
    await userEvent.click(ir);

    // 3. Identificar
    const tarjetas = await screen.findAllByRole("article");
    await userEvent.click(within(tarjetas[0]).getByRole("radio", { name: "Es A" }));
    await userEvent.click(within(tarjetas[1]).getByRole("radio", { name: "Es B" }));
    await userEvent.click(screen.getByRole("button", { name: "Completar análisis" }));
    await userEvent.click(await screen.findByRole("button", { name: "Sí, completar" }));
    await waitFor(() => expect(mundo.pedidos.find((p) => p.ruta.endsWith("/siembra"))?.cuerpo).toEqual({ track_a: 41, track_b: 64 }));
    expect(await screen.findByText("Esperando su turno en la estación.")).toBeInTheDocument();

    // 4. El worker termina sin dejar nada sin asignar: la pantalla va a la Fight-Card
    const fc = structuredClone(FIGHTCARDS["0.2"]) as any;
    fc.identidad.sin_asignar = 0;
    mundo.fightcard = fc;
    mundo.job = structuredClone(JOBS.listo);
    await avanzar(3_500);
    await waitFor(() => expect(window.location.pathname).toBe("/sesion/"));
    expect(new URLSearchParams(window.location.search).get("vista")).toBe("fightcard");
    const detectados = (await screen.findByText("Golpes detectados", { selector: "span" })).closest("[data-fila]") as HTMLElement;
    expect(within(detectados).getByText(String(fc.peleadores.A.total.total))).toBeInTheDocument();
    expect(within(detectados).getByText(String(fc.peleadores.B.total.total))).toBeInTheDocument();
  });
});
