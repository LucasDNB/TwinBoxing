/*
 * W8 (05 y 09): CA-05.1 a CA-05.4, mas I3 (banderas) e I5 (0.1 y 0.2) de 06 §1.
 */
import { describe, expect, it, beforeEach } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { conSesion, enRuta, montar } from "./entorno";
import { FIGHTCARDS, ID, JOBS, mundo, productoBase } from "./api-simulada";
import { PantallaSesion } from "@/components/pantallas/sesion/PantallaSesion";

function fcTresRounds() {
  // La sesion real dividida en tres rounds declarados, con los totales que suman.
  const fc = structuredClone(FIGHTCARDS["0.2"]) as Record<string, any>;
  const lim = [[0, 60], [60, 120], [120, 174.933]];
  fc.video.rounds = lim.map(([a, b], i) => ({ round: i + 1, inicio_s: a, fin_s: b }));
  for (const p of Object.values(fc.peleadores) as any[]) {
    p.por_round = lim.map(([a, b], i) => {
      const g = p.golpes.filter((x: any) => a <= x.t_inicio && x.t_inicio < b);
      const izq = g.filter((x: any) => x.brazo === "izq").length;
      return { round: i + 1, inicio_s: a, fin_s: b, total: g.length, izq, der: g.length - izq, por_minuto: Math.round(g.length * 60 / (b - a) * 100) / 100 };
    });
  }
  return fc;
}

beforeEach(() => {
  conSesion();
  mundo.job = structuredClone(JOBS.listo);
  enRuta(`/sesion/?id=${ID}&vista=fightcard`);
});

const fila = (titulo: string) => screen.getByText(titulo, { selector: "span" }).closest("[data-fila]") as HTMLElement;

describe("05 / 09 · Fight-Card", () => {
  it("CA-05.1: los detectados coinciden con peleadores.X.total del documento", async () => {
    montar(<PantallaSesion />);
    await screen.findByText("Golpes detectados");
    const f = fila("Golpes detectados");
    expect(within(f).getByText("51")).toBeInTheDocument();
    expect(within(f).getByText("32")).toBeInTheDocument();
  });

  it("CA-05.1 y CA-05.2: con C2, la suma de los rounds da el total y cambiar de round no pide nada", async () => {
    mundo.producto = productoBase({ comparacion_rounds: true });
    mundo.fightcard = fcTresRounds();
    montar(<PantallaSesion />);
    await screen.findByText("Golpes detectados");
    const fc = mundo.fightcard as any;
    let suma = 0;
    await screen.findByRole("tab", { name: "R1" });
    const antes = mundo.pedidos.length;
    for (const r of [1, 2, 3]) {
      await userEvent.click(screen.getByRole("tab", { name: `R${r}` }));
      await waitFor(() => expect(screen.getByText(`Volumen del round ${r}`)).toBeInTheDocument());
      const total = fc.peleadores.A.por_round[r - 1].total;
      expect(within(fila("Golpes detectados")).getByText(String(total))).toBeInTheDocument();
      suma += total;
    }
    expect(suma).toBe(fc.peleadores.A.total.total);
    expect(mundo.pedidos.slice(antes).filter((p) => p.ruta.startsWith("/fightcards"))).toHaveLength(0);
  });

  it("CA-05.3: el margen sale del recall del documento", async () => {
    const fc = structuredClone(FIGHTCARDS["0.2"]) as any;
    fc.detector.recall_medido = 0.7;
    mundo.fightcard = fc;
    montar(<PantallaSesion />);
    expect(await screen.findByText(/se le escapa 3 de cada 10/)).toBeInTheDocument();
  });

  it("I3 / CA-05.4: con C2 y C3 apagados no hay guardia ni rounds; conectados, puntos y ganador nunca", async () => {
    mundo.fightcard = fcTresRounds();
    montar(<PantallaSesion />);
    await screen.findByText("Golpes detectados");
    await waitFor(() => expect(mundo.pedidos.some((p) => p.ruta === "/producto")).toBe(true));
    const texto = document.body.textContent!;
    expect(screen.queryByRole("tab", { name: "R1" })).toBeNull();
    expect(screen.getByText("comparación entre rounds en validación")).toBeInTheDocument();
    expect(texto).not.toMatch(/Golpes con la otra mano arriba/);
    expect(texto).not.toMatch(/De R\d a R\d/);
    expect(screen.getAllByText("en validación").length).toBe(2);
    expect(texto).not.toMatch(/conectad|puntos|ganador/i);
  });

  it("I3: con las dos banderas encendidas aparecen guardia y la lectura por round", async () => {
    mundo.producto = productoBase({ comparacion_rounds: true, guardia: true });
    mundo.fightcard = fcTresRounds();
    montar(<PantallaSesion />);
    await screen.findByText("Golpes detectados");
    expect(await screen.findByText("Golpes con la otra mano arriba")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /De R2 a R3/ })).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/conectad|ganador/i);
  });

  it("I5: una 0.1 se dibuja y dice lo de los tramos", async () => {
    mundo.fightcard = structuredClone(FIGHTCARDS["0.1"]);
    montar(<PantallaSesion />);
    expect(await screen.findByText("Golpes detectados")).toBeInTheDocument();
    expect(screen.getByText(/se procesó antes de la revisión de tramos/)).toBeInTheDocument();
  });

  it("I5: una version 0.4 muestra el error de version y no una tarjeta a medias", async () => {
    mundo.fightcard = { ...structuredClone(FIGHTCARDS["0.3"]), version: "0.4" };
    montar(<PantallaSesion />);
    expect(await screen.findByText("No se pudo leer la Fight-Card")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(/versión 0.4/);
    expect(screen.queryByText("Golpes detectados")).toBeNull();
  });

  it("I5: un campo obligatorio faltante tambien", async () => {
    const fc = structuredClone(FIGHTCARDS["0.2"]) as any;
    delete fc.detector;
    mundo.fightcard = fc;
    montar(<PantallaSesion />);
    expect(await screen.findByText("No se pudo leer la Fight-Card")).toBeInTheDocument();
    expect(screen.queryByText("Golpes detectados")).toBeNull();
  });

  it("la distribucion por tipo va como estimacion con el checkpoint", async () => {
    montar(<PantallaSesion />);
    await screen.findByText("Tipo de golpe");
    const bloque = screen.getByRole("region", { name: "Tipo de golpe" });
    expect(bloque).toHaveAttribute("data-estado", "estimacion");
    expect(within(bloque).getByText("gancho")).toBeInTheDocument();
    expect(within(bloque).getByText("tipo menos confiable")).toBeInTheDocument();
    expect(within(bloque).getByText(/prueba-fixture-no-es-un-modelo/)).toBeInTheDocument();
  });

  it("exportar ofrece imprimir, CSV y JSON", async () => {
    montar(<PantallaSesion />);
    await userEvent.click(await screen.findByRole("button", { name: /Exportar/ }));
    expect(await screen.findByRole("menuitem", { name: "Para imprimir (PDF)" })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: "Lista de golpes (CSV)" })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: "Datos completos (JSON)" })).toBeInTheDocument();
  });
});
