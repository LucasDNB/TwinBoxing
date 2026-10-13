/*
 * I2 (06 §1) y CA-04.2: correccion de tipo y "no fue un golpe".
 */
import { describe, expect, it, beforeEach } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { conSesion, enRuta, montar } from "./entorno";
import { ID, JOBS, mundo } from "./api-simulada";
import { videoSimulado } from "./video";
import { PantallaSesion } from "@/components/pantallas/sesion/PantallaSesion";

beforeEach(() => {
  conSesion();
  videoSimulado();
  mundo.job = structuredClone(JOBS.listo);
  enRuta(`/sesion/?id=${ID}&vista=video`);
});

async function primeraFila() {
  const consola = await screen.findByRole("region", { name: "Golpes de Peleador A" });
  const filas = within(consola).getAllByRole("listitem");
  return { consola, fila: filas[0] };
}

async function corregirA(fila: HTMLElement, opcion: string) {
  await userEvent.click(within(fila).getByRole("button", { name: /Corregir el golpe/ }));
  await userEvent.click(await screen.findByRole("radio", { name: opcion }));
  await userEvent.click(screen.getByRole("button", { name: "Guardar" }));
}

describe("I2 · Corrección", () => {
  it("la lista cambia en el acto y la correccion sobrevive a recargar (CA-04.2)", async () => {
    const { unmount } = montar(<PantallaSesion />);
    const { fila } = await primeraFila();
    expect(within(fila).getByText("jab")).toBeInTheDocument();
    await corregirA(fila, "hook");
    await waitFor(() => expect(within(fila).getByText("corregido")).toBeInTheDocument());
    expect(within(fila).getByText("hook")).toBeInTheDocument();
    expect(within(fila).getByText("jab").tagName).toBe("S");
    const p = mundo.pedidos.find((x) => x.metodo === "PATCH");
    expect(p?.cuerpo).toEqual({ tipo: "hook" });

    unmount();
    montar(<PantallaSesion />);
    const otra = await primeraFila();
    expect(within(otra.fila).getByText("hook")).toBeInTheDocument();
    expect(within(otra.fila).getByText("corregido")).toBeInTheDocument();
  });

  it("si la API responde 422, vuelve atras con aviso", async () => {
    mundo.corregir = "422";
    montar(<PantallaSesion />);
    const { fila } = await primeraFila();
    await corregirA(fila, "uppercut");
    expect(await screen.findByText(/No se pudo guardar la corrección: tipo desconocido/)).toBeInTheDocument();
    await waitFor(() => expect(within(fila).queryByText("corregido")).toBeNull());
    expect(within(fila).getByText("jab")).toBeInTheDocument();
  });

  it("no fue un golpe no cambia detectados y suma descartados", async () => {
    const { unmount } = montar(<PantallaSesion />);
    const { fila } = await primeraFila();
    await corregirA(fila, "No fue un golpe");
    await waitFor(() => expect(within(fila).getByText("no fue un golpe")).toBeInTheDocument());
    unmount();

    enRuta(`/sesion/?id=${ID}&vista=fightcard`);
    montar(<PantallaSesion />);
    // 51 detectados de A no cambia; descartados pasa de 1 (fixture) a 2.
    expect(await screen.findByText("Descartados por el entrenador")).toBeInTheDocument();
    const fila2 = screen.getByText("Descartados por el entrenador").closest("[data-fila]")!;
    expect(within(fila2 as HTMLElement).getByText("2")).toBeInTheDocument();
    const detectados = screen.getByText("Golpes detectados").closest("[data-fila]")!;
    expect(within(detectados as HTMLElement).getAllByText("51").length).toBeGreaterThan(0);
  });
});
