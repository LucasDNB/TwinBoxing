/*
 * W6: CA-04.1, CA-04.3, CA-04.4 e I4 (salto al instante y ticket).
 */
import { describe, expect, it, beforeEach, vi, afterEach } from "vitest";
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { conSesion, enRuta, montar } from "./entorno";
import { FIGHTCARDS, ID, JOBS, mundo, productoBase } from "./api-simulada";
import { videoSimulado, reproduccion } from "./video";
import { PantallaSesion } from "@/components/pantallas/sesion/PantallaSesion";

function fcCon347() {
  const fc = structuredClone(FIGHTCARDS["0.2"]) as typeof FIGHTCARDS["0.2"];
  fc.video.duracion_s = 400;
  fc.video.rounds = [{ round: 1, inicio_s: 0, fin_s: 400 }];
  const g = fc.peleadores.A.golpes[3];
  g.t_inicio = 347.0;
  g.t_fin = 347.3;
  return fc;
}

beforeEach(() => {
  conSesion();
  videoSimulado();
  reproduccion.reproducciones = 0;
  mundo.job = structuredClone(JOBS.listo);
  enRuta(`/sesion/?id=${ID}&vista=video`);
});
afterEach(() => vi.useRealTimers());

describe("04 · Video y golpes", () => {
  it("I4 / CA-04.1: tocar un golpe a los 05:47 deja el video en 05:46,5 y reproduciendo", async () => {
    mundo.fightcard = fcCon347();
    montar(<PantallaSesion />);
    const tick = await screen.findByRole("button", { name: /Golpe detectado de Peleador A, .* 5 minutos 47/ });
    const video = (await screen.findByLabelText("Video de la sesión")) as HTMLVideoElement;
    await userEvent.click(tick);
    expect(video.currentTime).toBeCloseTo(346.5, 1);
    expect(video.paused).toBe(false);
    expect(reproduccion.reproducciones).toBe(1);
  });

  it("I4: la URL del video lleva ?t=<ticket> y nunca el token de sesion", async () => {
    montar(<PantallaSesion />);
    const video = (await screen.findByLabelText("Video de la sesión")) as HTMLVideoElement;
    const src = video.getAttribute("src")!;
    expect(src).toMatch(new RegExp(`^/videos/${ID}/stream\\?t=ticket-\\d+$`));
    expect(src).not.toContain("token-bueno");
    // Sin procesado (HEAD 404) se usa el original y se ofrece armarlo.
    expect(await screen.findByRole("button", { name: "Armar video con esqueleto" })).toBeInTheDocument();
  });

  it("I4: el ticket se renueva a los 25 minutos sin perder la posicion", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    montar(<PantallaSesion />);
    const video = (await screen.findByLabelText("Video de la sesión")) as HTMLVideoElement;
    const antes = video.getAttribute("src");
    video.currentTime = 100;
    await act(async () => { await vi.advanceTimersByTimeAsync(25 * 60 * 1000 + 1000); });
    await waitFor(() => expect(video.getAttribute("src")).not.toBe(antes));
    fireEvent.loadedMetadata(video);
    expect(video.currentTime).toBe(100);
    expect(mundo.pedidos.filter((p) => p.ruta.endsWith("/ticket")).length).toBeGreaterThanOrEqual(2);
  });

  it("CA-04.4: la linea se recorre con flechas y cada tick dice que es", async () => {
    montar(<PantallaSesion />);
    const grupo = await screen.findByRole("group", { name: /Línea de tiempo de golpes detectados/ });
    const enOrden = within(grupo).getAllByRole("button").filter((b) => b.tabIndex === 0);
    expect(enOrden).toHaveLength(1);
    enOrden[0].focus();
    await userEvent.keyboard("{ArrowRight}");
    await waitFor(() => expect(document.activeElement?.getAttribute("aria-label")).toMatch(/^Golpe detectado de Peleador A, .*8 segundos$/));
    await userEvent.keyboard("{ArrowDown}");
    await waitFor(() => expect(document.activeElement?.getAttribute("aria-label")).toMatch(/^Golpe detectado de Peleador B/));
    await userEvent.keyboard("{ArrowLeft}");
    expect(document.activeElement?.getAttribute("aria-label")).toMatch(/^Golpe detectado de Peleador B/);
  });

  it("CA-04.3: con C3 apagado no aparece guardia en ningun golpe; con C3 si", async () => {
    const { unmount } = montar(<PantallaSesion />);
    await screen.findAllByRole("button", { name: /Ir al golpe/ });
    expect(document.body.textContent).not.toMatch(/guardia (arriba|caída)/);
    unmount();
    mundo.producto = productoBase({ guardia: true });
    montar(<PantallaSesion />);
    await screen.findAllByRole("button", { name: /Ir al golpe/ });
    await waitFor(() => expect(document.body.textContent).toMatch(/guardia caída/));
  });

  it("el tipo estimado se ve como estimacion y la nota dice el checkpoint", async () => {
    montar(<PantallaSesion />);
    await screen.findAllByRole("button", { name: /Ir al golpe/ });
    expect(screen.getByText(/estimación del modelo prueba-fixture-no-es-un-modelo/)).toBeInTheDocument();
    expect(screen.getAllByText("corregido").length).toBeGreaterThan(0);
  });

  it("si la sesion no esta lista muestra el estado en vez del contenido", async () => {
    mundo.job = structuredClone(JOBS.procesando);
    montar(<PantallaSesion />);
    expect(await screen.findByText("Buscando a las personas y su pose.")).toBeInTheDocument();
    expect(screen.getByText("pose · 64%")).toBeInTheDocument();
    expect(screen.queryByLabelText("Video de la sesión")).toBeNull();
  });

  it("en espera de siembra ofrece ir a marcar", async () => {
    mundo.job = structuredClone(JOBS.espera_siembra);
    montar(<PantallaSesion />);
    expect(await screen.findByRole("link", { name: "Marcar peleadores" })).toHaveAttribute("href", `/sesion/identificar/?id=${ID}`);
  });
});
