/*
 * B8 del lado de la web: sesiones de un solo peleador, con la bandera encendida.
 */
import { describe, expect, it, beforeEach } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { conSesion, enRuta, montar } from "./entorno";
import { FIGHTCARDS, ID, JOBS, mundo, productoBase } from "./api-simulada";
import { videoSimulado } from "./video";
import { PantallaNueva } from "@/components/pantallas/nueva/PantallaNueva";
import { PantallaIdentificar } from "@/components/pantallas/identificar/PantallaIdentificar";
import { PantallaSesion } from "@/components/pantallas/sesion/PantallaSesion";

beforeEach(() => {
  conSesion();
  videoSimulado();
  mundo.producto = productoBase({ sesiones_individuales: true });
});

describe("B8 · un solo peleador", () => {
  it("con la bandera, bolsa se puede elegir y pide un solo rincon", async () => {
    enRuta("/sesiones/nueva/");
    montar(<PantallaNueva />);
    const bolsa = await screen.findByRole("radio", { name: /Bolsa/ });
    await waitFor(() => expect(bolsa).toBeEnabled());
    await userEvent.click(bolsa);
    expect(screen.queryByLabelText(/Rincón azul/)).toBeNull();
    expect(screen.getByLabelText(/Rincón rojo/)).toBeInTheDocument();
  });

  it("la identificacion manda una sola semilla", async () => {
    mundo.job = { ...structuredClone(JOBS.espera_siembra), contexto: "bolsa" };
    enRuta(`/sesion/identificar/?id=${ID}`);
    montar(<PantallaIdentificar />);
    const t = await screen.findAllByRole("article");
    expect(within(t[0]).getByRole("radio", { name: "Es B" })).toBeDisabled();
    await userEvent.click(within(t[0]).getByRole("radio", { name: "Es A" }));
    await userEvent.click(screen.getByRole("button", { name: "Completar análisis" }));
    await userEvent.click(await screen.findByRole("button", { name: "Sí, completar" }));
    await waitFor(() => expect(mundo.pedidos.find((p) => p.ruta.endsWith("/siembra"))?.cuerpo).toEqual({ track_a: 41, track_b: null }));
  });

  it("una Fight-Card solo con A se dibuja sin columna B", async () => {
    const fc = structuredClone(FIGHTCARDS["0.2"]) as any;
    delete fc.peleadores.B;
    mundo.fightcard = fc;
    mundo.job = { ...structuredClone(JOBS.listo), contexto: "bolsa" };
    enRuta(`/sesion/?id=${ID}&vista=fightcard`);
    montar(<PantallaSesion />);
    const fila = (await screen.findByText("Golpes detectados", { selector: "span" })).closest("[data-fila]") as HTMLElement;
    expect(within(fila).getByText("51")).toBeInTheDocument();
    expect(screen.queryByText("Peleador B")).toBeNull();
  });
});
