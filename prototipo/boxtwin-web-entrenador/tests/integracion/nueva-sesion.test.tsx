/*
 * W3: CA-02.1 a CA-02.3 (04 §02).
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { conSesion, enRuta, montar } from "./entorno";
import { mundo } from "./api-simulada";

const subidas: { datos: FormData; avanzar: (e: number, t: number) => void; resolver: (r: unknown) => void; rechazar: (e: unknown) => void; cancelada: boolean }[] = [];

vi.mock("@/lib/api/cliente", async (orig) => {
  const real = await orig<typeof import("@/lib/api/cliente")>();
  return {
    ...real,
    subirConProgreso: (datos: FormData, avanzar: (e: number, t: number) => void) => {
      let resolver!: (r: unknown) => void;
      let rechazar!: (e: unknown) => void;
      const promesa = new Promise((a, b) => { resolver = a; rechazar = b; });
      const s = { datos, avanzar, resolver, rechazar, cancelada: false };
      subidas.push(s);
      return {
        promesa,
        cancelar: () => { s.cancelada = true; rechazar(new real.ErrorApi("subida cancelada", -1)); },
      };
    },
  };
});

vi.mock("@/lib/subida", async (orig) => {
  const real = await orig<typeof import("@/lib/subida")>();
  return { ...real, leerMetadatos: async () => ({ duracion_s: 600, ancho: 1920, alto: 1080 }) };
});

import { PantallaNueva } from "@/components/pantallas/nueva/PantallaNueva";
import { act } from "react";

function video(nombre = "sparring.mp4", bytes = 1000) {
  const f = new File(["x"], nombre, { type: "video/mp4" });
  Object.defineProperty(f, "size", { value: bytes });
  return f;
}

async function elegir(f: File) {
  const input = screen.getByTestId("input-galeria") as HTMLInputElement;
  fireEvent.change(input, { target: { files: [f] } });
}

beforeEach(() => {
  subidas.length = 0;
  conSesion();
  enRuta("/sesiones/nueva/");
});

describe("02 / 08 · Nueva sesión", () => {
  it("CA-02.1: un .avi se rechaza antes de subir, con el motivo", async () => {
    montar(<PantallaNueva />);
    await screen.findByText("Qué se filmó");
    await elegir(video("sparring.avi"));
    expect(await screen.findByRole("alert")).toHaveTextContent("El formato .avi no se acepta");
    expect(subidas).toHaveLength(0);
  });

  it("CA-02.1: un archivo mayor que max_bytes se rechaza antes de subir", async () => {
    montar(<PantallaNueva />);
    await screen.findByText("Qué se filmó");
    await elegir(video("sparring.mp4", 7 * 1024 ** 3));
    expect(await screen.findByRole("alert")).toHaveTextContent("el máximo es 6,00 GB");
  });

  it("CA-02.2: sin consentimiento no se puede enviar y dice por que", async () => {
    montar(<PantallaNueva />);
    await screen.findByText("Qué se filmó");
    await elegir(video());
    await screen.findByText(/sparring\.mp4 ·/);
    const boton = screen.getByRole("button", { name: "Subir y procesar" });
    expect(boton).toBeDisabled();
    expect(screen.getByText("Falta confirmar el consentimiento de las personas filmadas.")).toBeInTheDocument();
    await userEvent.click(screen.getByLabelText(/dieron su consentimiento/));
    await waitFor(() => expect(screen.getByRole("button", { name: "Subir y procesar" })).toBeEnabled());
  });

  it("la estimacion sale del factor medido de /producto (D2)", async () => {
    montar(<PantallaNueva />);
    await screen.findByText("Qué se filmó");
    await elegir(video());
    expect((await screen.findAllByText("Entre 11 y 26 min")).length).toBeGreaterThan(0);
  });

  it("bolsa y sombra quedan en 'Próximamente' sin la bandera de B8", async () => {
    montar(<PantallaNueva />);
    expect((await screen.findAllByText("Próximamente")).length).toBe(2);
  });

  it("envia contexto, rounds y consentimiento, y al terminar va a la sesion", async () => {
    montar(<PantallaNueva />);
    await screen.findByText("Qué se filmó");
    await elegir(video());
    await screen.findByText(/sparring\.mp4 ·/);
    await userEvent.click(screen.getByLabelText(/dieron su consentimiento/));
    await userEvent.click(screen.getByRole("button", { name: "Subir y procesar" }));
    expect(subidas).toHaveLength(1);
    const d = subidas[0].datos;
    expect(d.get("contexto")).toBe("sparring");
    expect(d.get("consentimiento")).toBe("true");
    expect(d.get("rounds")).toBe("3");
    expect(d.get("round_s")).toBe("180");
    expect(d.get("descanso_s")).toBe("60");
    await act(async () => subidas[0].resolver({ job_id: "nueva", estado: "en_cola", bytes: 1000 }));
    await waitFor(() => expect(window.location.search).toBe("?id=nueva"));
    expect(window.location.pathname).toBe("/sesion/");
  });

  it("CA-02.3: cancelar al 40% aborta, no navega y no deja sesion", async () => {
    montar(<PantallaNueva />);
    await screen.findByText("Qué se filmó");
    await elegir(video("sparring.mp4", 1000));
    await screen.findByText(/sparring\.mp4 ·/);
    await userEvent.click(screen.getByLabelText(/dieron su consentimiento/));
    await userEvent.click(screen.getByRole("button", { name: "Subir y procesar" }));
    act(() => subidas[0].avanzar(400, 1000));
    expect(await screen.findByText("Subiendo · 40%")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Cancelar" }));
    await userEvent.click(await screen.findByRole("button", { name: "Cancelar subida" }));
    await waitFor(() => expect(subidas[0].cancelada).toBe(true));
    expect(await screen.findByRole("button", { name: "Subir y procesar" })).toBeInTheDocument();
    expect(window.location.pathname).toBe("/sesiones/nueva/");
    expect(mundo.pedidos.some((p) => p.ruta.includes("/sesiones/") && p.metodo === "PUT")).toBe(false);
  });
});
