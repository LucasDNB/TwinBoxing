/*
 * W13 (ampliacion 01 §4) con la regla del 14-10: cualquier secuencia de 2+ golpes con menos de
 * 0,5 s entre cada uno, en el analisis, la Fight-Card y el perfil.
 * CA-13.1 a CA-13.3. La regla no se prueba aca: vive en el anotador (test_mvp_combinaciones.py)
 * y la web solo dibuja lo que trae el documento.
 */
import { describe, expect, it, beforeEach } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { conSesion, enRuta, montar } from "./entorno";
import { FIGHTCARDS, ID, JOBS, mundo, productoBase } from "./api-simulada";
import { perfilDePrueba } from "./perfil-de-prueba";
import { videoSimulado } from "./video";
import { PantallaSesion } from "@/components/pantallas/sesion/PantallaSesion";
import { PantallaPeleador } from "@/components/pantallas/peleador/PantallaPeleador";

const MARGEN = /Para contar una combinación el sistema tiene que encontrar al menos dos golpes seguidos. Encuentra cerca de 5 de cada 10/;
const SECUENCIA = /^(izq|der)(-(izq|der))+$/;

function fcTresRounds() {
  // La 0.3 real (A: 3 combinaciones en R1, 1 en R2, 3 en R3; B: 2, 3 y 0) con tres rounds declarados.
  const fc = structuredClone(FIGHTCARDS["0.3"]) as Record<string, any>;
  const lim = [[0, 60], [60, 120], [120, 174.933]];
  fc.video.rounds = lim.map(([a, b], i) => ({ round: i + 1, inicio_s: a, fin_s: b }));
  for (const p of Object.values(fc.peleadores) as any[]) {
    p.por_round = lim.map(([a, b], i) => {
      const g = p.golpes.filter((x: any) => a <= x.t_inicio && x.t_inicio < b);
      const izq = g.filter((x: any) => x.brazo === "izq").length;
      return { round: i + 1, inicio_s: a, fin_s: b, total: g.length, izq, der: g.length - izq, por_minuto: 1 };
    });
    p.combinaciones.por_round = lim.map(([a, b], i) => ({
      round: i + 1, total: p.combinaciones.lista.filter((c: any) => a <= c.t_inicio && c.t_inicio < b).length,
    }));
  }
  return fc;
}

beforeEach(() => {
  conSesion();
  videoSimulado();
  mundo.job = structuredClone(JOBS.listo);
});

describe("W13 · combinaciones (regla del 14-10)", () => {
  it("CA-13.1: la linea marca cada combinacion y el corchete lleva al primer golpe menos medio segundo", async () => {
    enRuta(`/sesion/?id=${ID}&vista=video`);
    montar(<PantallaSesion />);
    const corchetes = await screen.findAllByRole("button", { name: /^Combinación / });
    expect(corchetes).toHaveLength(12);
    expect(corchetes[0]).toHaveAccessibleName(/^Combinación izquierda, derecha de/);
    const video = (await screen.findByLabelText("Video de la sesión")) as HTMLVideoElement;
    await userEvent.click(corchetes[0]);
    expect(video.currentTime).toBeCloseTo(15.2, 1);
  });

  it("CA-13.2: solo los golpes de una combinacion llevan la insignia, con su secuencia de manos", async () => {
    enRuta(`/sesion/?id=${ID}&vista=video`);
    montar(<PantallaSesion />);
    const lista = await screen.findByRole("region", { name: /Golpes de Peleador A/ });
    const enCombinacion = (FIGHTCARDS["0.3"] as any).peleadores.A.combinaciones.lista.flatMap((c: any) => c.golpes);
    expect(within(lista).getAllByText(SECUENCIA)).toHaveLength(enCombinacion.length);
    expect(within(lista).getAllByText("izq-izq-der-izq-der-izq")).toHaveLength(6);
    const solo = within(lista).getByRole("button", { name: "Ir al golpe de 00:02" }).closest("li")!;
    expect(within(solo).queryByText(SECUENCIA)).toBeNull();
  });

  it("CA-13.1 y CA-13.3: la Fight-Card suma igual que la linea, con su margen", async () => {
    enRuta(`/sesion/?id=${ID}&vista=fightcard`);
    montar(<PantallaSesion />);
    const bloque = await screen.findByRole("region", { name: "Combinaciones" });
    expect(bloque).toHaveAttribute("data-estado", "medido");
    expect(within(bloque).getByText("7")).toBeInTheDocument();
    expect(within(bloque).getByText("5")).toBeInTheDocument();
    expect(within(bloque).getByText(MARGEN)).toBeInTheDocument();
  });

  it("CA-13.1: por round solo con C2, y los rounds suman la sesion", async () => {
    mundo.producto = productoBase({ comparacion_rounds: true });
    mundo.fightcard = fcTresRounds();
    enRuta(`/sesion/?id=${ID}&vista=fightcard&round=3`);
    montar(<PantallaSesion />);
    await waitFor(() => expect(screen.getByRole("tab", { name: "R3" })).toHaveAttribute("aria-selected", "true"));
    const bloque = screen.getByRole("region", { name: "Combinaciones" });
    expect(within(bloque).getByText("3")).toBeInTheDocument();
    expect(within(bloque).getByText("0")).toBeInTheDocument();
    expect(within(bloque).getByText(MARGEN)).toBeInTheDocument();
  });

  it("una 0.2 dice que se proceso antes y no muestra ni numero ni margen", async () => {
    mundo.fightcard = structuredClone(FIGHTCARDS["0.2"]);
    enRuta(`/sesion/?id=${ID}&vista=fightcard`);
    montar(<PantallaSesion />);
    const bloque = await screen.findByRole("region", { name: "Combinaciones" });
    expect(within(bloque).getByText(/se procesó antes de contar combinaciones/)).toBeInTheDocument();
    expect(within(bloque).queryByText(MARGEN)).toBeNull();
  });

  it("sin ninguna combinacion dice que el sistema no las encontro, no que no combina", async () => {
    const fc = structuredClone(FIGHTCARDS["0.3"]) as any;
    for (const p of Object.values(fc.peleadores) as any[]) {
      p.combinaciones = { ...p.combinaciones, lista: [], total: 0, por_round: [{ round: 1, total: 0 }] };
    }
    mundo.fightcard = fc;
    enRuta(`/sesion/?id=${ID}&vista=fightcard`);
    montar(<PantallaSesion />);
    const bloque = await screen.findByRole("region", { name: "Combinaciones" });
    expect(within(bloque).getByText("El sistema no encontró combinaciones completas en esta sesión.")).toBeInTheDocument();
  });

  it("CA-13.1 y CA-13.3: el perfil muestra total, por minuto y por contexto, con el margen y los tipos aparte", async () => {
    mundo.perfiles = { b1: perfilDePrueba("b1", "Ana Paz") };
    enRuta("/peleador/?id=b1");
    montar(<PantallaPeleador />);
    const bloque = await screen.findByRole("region", { name: "Combinaciones" });
    expect(bloque).toHaveAttribute("data-estado", "medido");
    expect(within(bloque).getByText("23")).toBeInTheDocument();
    expect(within(bloque).getByText("0,41")).toBeInTheDocument();
    expect(within(bloque).getByText("Sparring")).toBeInTheDocument();
    expect(within(bloque).getByText(MARGEN)).toBeInTheDocument();
    expect(within(bloque).getByText("estimación")).toBeInTheDocument();
    expect(within(bloque).getByText(/jab · gancho · hook/)).toBeInTheDocument();
    expect(within(bloque).getByText("izq-der")).toBeInTheDocument();
    expect(within(bloque).getByText("14 de 2 golpes · 7 de 3 golpes · 2 de 4+ golpes")).toBeInTheDocument();
    expect(bloque.textContent).not.toMatch(/1-2-3/);
  });
});
