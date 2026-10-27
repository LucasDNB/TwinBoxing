/*
 * W5: identificacion fase 1. CA-03.1 (la siembra sale con los tracks elegidos) y CA-03.3
 * (los recortes cargan con ticket, nunca con el token).
 */
import { describe, expect, it, beforeEach } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { conSesion, enRuta, montar } from "./entorno";
import { ID, JOBS, mundo } from "./api-simulada";
import { PantallaIdentificar, marcar } from "@/components/pantallas/identificar/PantallaIdentificar";

beforeEach(() => {
  conSesion();
  mundo.job = structuredClone(JOBS.espera_siembra);
  enRuta(`/sesion/identificar/?id=${ID}`);
});

describe("03 · Identificación, fase 1", () => {
  it("muestra las personas con la pareja sugerida primero y rotulada", async () => {
    montar(<PantallaIdentificar />);
    const tarjetas = await screen.findAllByRole("article");
    expect(tarjetas).toHaveLength(6);
    expect(within(tarjetas[0]).getByText("sugerido")).toBeInTheDocument();
    expect(within(tarjetas[1]).getByText("sugerido")).toBeInTheDocument();
    expect(within(tarjetas[2]).queryByText("sugerido")).toBeNull();
  });

  it("CA-03.3: los recortes llevan el ticket en la URL y nunca el token", async () => {
    montar(<PantallaIdentificar />);
    await screen.findAllByRole("article");
    const imgs = await screen.findAllByRole("img", { name: /Recorte de la persona/ });
    for (const img of imgs) {
      const src = img.getAttribute("src")!;
      expect(src).toMatch(/\/jobs\/.+\/candidatos\/\d+\?t=ticket-\d+$/);
      expect(src).not.toContain("token-bueno");
    }
  });

  it("un solo A: marcar un segundo A le saca la marca al anterior", () => {
    const { nuevo, quitado } = marcar({ 41: "A", 64: "B" }, 228, "A");
    expect(nuevo).toEqual({ 64: "B", 228: "A" });
    expect(quitado).toBe(41);
  });

  it("CA-03.1: con A y B marcados y confirmados, manda la siembra con esos tracks", async () => {
    montar(<PantallaIdentificar />);
    const tarjetas = await screen.findAllByRole("article");
    const completar = screen.getByRole("button", { name: "Completar análisis" });
    expect(completar).toBeDisabled();
    await userEvent.click(within(tarjetas[0]).getByRole("radio", { name: "Es A" }));
    await userEvent.click(within(tarjetas[1]).getByRole("radio", { name: "Es B" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Completar análisis" })).toBeEnabled());
    await userEvent.click(screen.getByRole("button", { name: "Completar análisis" }));
    expect(await screen.findByText(/¿Persona 1 es A y persona 2 es B\?/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Sí, completar" }));
    await waitFor(() => {
      const p = mundo.pedidos.find((x) => x.metodo === "POST" && x.ruta.endsWith("/siembra"));
      expect(p?.cuerpo).toEqual({ track_a: 41, track_b: 64 });
    });
    // La sesion pasa a en_cola y la pantalla muestra la espera, sin otra intervencion.
    expect(await screen.findByText("Esperando su turno en la estación.")).toBeInTheDocument();
  });

  it("sin pareja sugerida avisa que no se vieron los dos juntos", async () => {
    mundo.job = { ...structuredClone(JOBS.espera_siembra), pareja_sugerida: null };
    montar(<PantallaIdentificar />);
    expect(await screen.findByText(/No encontramos un momento con los dos en cuadro a la vez/)).toBeInTheDocument();
  });

  it("con B2, muestra color de guante, tramo y altura", async () => {
    const j = structuredClone(JOBS.espera_siembra);
    j.candidatos[0] = { ...j.candidatos[0], color_guante: { hex: "#C8412B", nombre: "rojos" }, t_desde: 4, t_hasta: 190, alto_relativo: 0.97 } as never;
    mundo.job = j;
    montar(<PantallaIdentificar />);
    const t = (await screen.findAllByRole("article"))[0];
    expect(within(t).getByText("guantes rojos")).toBeInTheDocument();
    expect(within(t).getByText("00:04 a 03:10")).toBeInTheDocument();
    expect(within(t).getByText("altura 0,97 de la mediana")).toBeInTheDocument();
  });
});
