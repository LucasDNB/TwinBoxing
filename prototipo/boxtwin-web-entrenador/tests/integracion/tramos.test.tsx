/*
 * W9: identificacion fase 2 (04 §03). CA-03.2.
 */
import { describe, expect, it, beforeEach, vi, afterEach } from "vitest";
import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { conSesion, enRuta, montar } from "./entorno";
import { FIGHTCARDS, ID, JOBS, mundo } from "./api-simulada";
import { PantallaIdentificar } from "@/components/pantallas/identificar/PantallaIdentificar";

beforeEach(() => {
  conSesion();
  mundo.job = structuredClone(JOBS.listo);
  enRuta(`/sesion/identificar/?id=${ID}`);
});
afterEach(() => vi.useRealTimers());

describe("03 · Identificación, fase 2", () => {
  it("dice el resultado de la propagacion como es (D6)", async () => {
    montar(<PantallaIdentificar />);
    expect(await screen.findByText("13 de 20 tramos asignados solos")).toBeInTheDocument();
    expect(screen.getByText("separación de color 1,38 (mínimo 0,55)")).toBeInTheDocument();
    expect(screen.getByText(/El 86% del video quedó sin saber quién es quién/)).toBeInTheDocument();
  });

  it("lista los tramos con motivo y los manda con Aplicar", async () => {
    montar(<PantallaIdentificar />);
    const lista = await screen.findByRole("heading", { name: "Revisar tramos dudosos" });
    const items = within(lista.closest("section")!).getAllByRole("listitem");
    expect(within(items[0]).getByText("guante tapado o poca luz")).toBeInTheDocument();
    expect(within(items[1]).getByText("cruce o cuerpo a cuerpo")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Aplicar" })).toBeDisabled();
    await userEvent.click(within(items[0]).getByRole("radio", { name: "A" }));
    await userEvent.click(within(items[1]).getByRole("radio", { name: "Ignorar" }));
    await userEvent.click(screen.getByRole("button", { name: "Aplicar (2)" }));
    await waitFor(() => {
      const p = mundo.pedidos.find((x) => x.metodo === "PUT" && x.ruta.endsWith("/tramos"));
      expect(p?.cuerpo).toEqual({ asignaciones: [{ track: 228, rol: "A" }, { track: 430, rol: "ninguno" }] });
    });
  });

  it("CA-03.2: despues de aplicar vuelve a completar y la Fight-Card siguiente trae lo asignado", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    montar(<PantallaIdentificar />);
    const lista = await screen.findByRole("heading", { name: "Revisar tramos dudosos" });
    const items = within(lista.closest("section")!).getAllByRole("listitem");
    await userEvent.click(within(items[0]).getByRole("radio", { name: "A" }));
    await userEvent.click(screen.getByRole("button", { name: "Aplicar (1)" }));
    expect(await screen.findByText("Esperando su turno en la estación.")).toBeInTheDocument();

    // El worker termina: mas cobertura de A, un tramo asignado a mano, sus golpes sumados.
    const fc = structuredClone(FIGHTCARDS["0.2"]) as any;
    fc.identidad.cobertura_A = 0.93;
    fc.identidad.tramos_asignados_manual = 1;
    fc.peleadores.A.golpes.push({ ...fc.peleadores.A.golpes[0], id: "A-izq-2700", t_inicio: 90, t_fin: 90.3 });
    fc.peleadores.A.total.total += 1;
    fc.peleadores.A.por_round[0].total += 1;
    mundo.fightcard = fc;
    mundo.tramos = { ...(mundo.tramos as object), tramos: [{ ...(mundo.tramos as any).tramos[0], asignado: "A" }] };
    mundo.job = structuredClone(JOBS.listo);
    await act(async () => { await vi.advanceTimersByTimeAsync(3_500); });
    expect(await screen.findByText("13 de 20 tramos asignados solos, 1 a mano")).toBeInTheDocument();
    expect(screen.getByText("asignado a mano: A")).toBeInTheDocument();
  });

  it("una sesion 0.1 dice que se proceso antes de la revision de tramos", async () => {
    mundo.fightcard = structuredClone(FIGHTCARDS["0.1"]);
    mundo.tramos = null;
    montar(<PantallaIdentificar />);
    expect(await screen.findByText(/se procesó antes de la revisión de tramos/)).toBeInTheDocument();
  });

  it("sin nada sin asignar, va directo a la Fight-Card", async () => {
    const fc = structuredClone(FIGHTCARDS["0.2"]) as any;
    fc.identidad.sin_asignar = 0;
    mundo.fightcard = fc;
    montar(<PantallaIdentificar />);
    await waitFor(() => expect(window.location.search).toBe(`?id=${ID}&vista=fightcard`));
    expect(window.location.pathname).toBe("/sesion/");
  });
});
