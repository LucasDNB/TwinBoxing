/*
 * W12: Rival Fantasma (04 §07). CA-07.1 y CA-07.2. Con los cambios del 14-10
 * (specs/cambios-14-10/02): avatares 3D con brazos que se estiran (W25, CA-25.1) y el bloque
 * unificado "Ataque contra exposicion" (W26, CA-26.1 a CA-26.4).
 */
import { describe, expect, it, beforeEach } from "vitest";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { conSesion, enRuta, montar } from "./entorno";
import { mundo, productoBase } from "./api-simulada";
import { perfilDePrueba, rivalDePrueba } from "./perfil-de-prueba";
import { PantallaRival } from "@/components/pantallas/rival/PantallaRival";

const box = (id: string, nombre: string) => ({ id, nombre, guardia: null, notas: null, creado: "2026-10-01", categoria: null, altura_m: null, envergadura_m: null, sesiones: 3, ultima_sesion: null });

beforeEach(() => {
  conSesion();
  mundo.boxeadores = [box("b1", "Ana Paz"), box("b2", "Bea Ruiz")];
  mundo.perfiles = {
    b1: perfilDePrueba("b1", "Ana Paz"),
    b2: perfilDePrueba("b2", "Bea Ruiz", {
      boxeador: { id: "b2", nombre: "Bea Ruiz", guardia: "zurda", notas: null, categoria: null, altura_m: 1.7, envergadura_m: null },
      por_contexto: [{ contexto: "sparring", sesiones: 3, golpes_por_minuto: 10, jab_fraccion: 0.3, guardia_opuesta_arriba: 0.4, caida_ultimo_round: -0.3 }],
    }),
  };
});

describe("07 · Rival Fantasma", () => {
  it("CA-07.2: no se puede elegir el mismo peleador en A y B", async () => {
    enRuta("/rival/?a=b1");
    montar(<PantallaRival />);
    const b = await screen.findByLabelText(/Perfil B/);
    expect(within(b).getByRole("option", { name: /Ana Paz/ })).toBeDisabled();
    expect(within(b).getByRole("option", { name: /Bea Ruiz/ })).toBeEnabled();
  });

  it("dice que no es un pronostico, dibuja los dos avatares y dice que medida falta", async () => {
    enRuta("/rival/?a=b1&b=b2");
    montar(<PantallaRival />);
    expect(await screen.findByText(/No es un pronóstico ni una probabilidad de victoria/)).toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /Ana Paz: 1,75 m de altura y 1,80 m de envergadura/ })).toBeInTheDocument();
    expect(screen.getAllByText(/Bea Ruiz: 1,70 m de altura; falta la envergadura/).length).toBeGreaterThan(0);
    expect(screen.getByText("ilustrativo, no reconstruido del video")).toBeInTheDocument();
    expect(screen.getByText("distancia de presentación, no medida")).toBeInTheDocument();
  });

  it("CA-25.1: mover un brazo cambia su alcance y no toca al otro avatar", async () => {
    enRuta("/rival/?a=b1&b=b2");
    montar(<PantallaRival />);
    const control = await screen.findByLabelText(/Brazo adelantado/, { selector: "#brazo-A-adelantada" });
    const alcanceA = () => document.querySelector('[data-alcance="A-adelantada"]')!.textContent;
    const alcanceB = () => document.querySelector('[data-alcance="B-adelantada"]')!.textContent;
    const antesA = alcanceA();
    const antesB = alcanceB();
    fireEvent.change(control, { target: { value: "100" } });
    // Extendido, el alcance es el largo entero del brazo: (1,80 - 0,23 x 1,75) / 2 = 0,70 m.
    expect(alcanceA()).toBe("alcance 0,70 m");
    expect(alcanceA()).not.toBe(antesA);
    expect(alcanceB()).toBe(antesB);
  });

  it("CA-26: un solo bloque con las lecturas de zona a cada lado, oportunidad y amenaza", async () => {
    mundo.producto = productoBase({ objetivo: true, exposicion: true });
    mundo.rival = rivalDePrueba();
    enRuta("/rival/?a=b1&b=b2");
    montar(<PantallaRival />);
    const bloque = await screen.findByRole("region", { name: "Ataque contra exposición" });
    expect(bloque).toHaveAttribute("data-estado", "estimacion");
    expect(screen.queryByRole("region", { name: "Oportunidades y amenazas" })).toBeNull();
    expect(screen.queryByRole("region", { name: /Dónde ataca uno/ })).toBeNull();
    await within(bloque).findByText("Hacia dónde tira Ana Paz");
    expect(within(bloque).getAllByText("oportunidad")).toHaveLength(2);
    expect(within(bloque).getAllByText("amenaza")).toHaveLength(2);
    expect(bloque.textContent).not.toMatch(/conect|vulnerable|probabilidad de (que|conectar)/i);
    expect(within(bloque).getByRole("table", { name: /Ataque contra exposición/ }).querySelectorAll("tbody tr")).toHaveLength(8);
    expect(within(bloque).getByText(/Corregido por el detector: encuentra 3 de cada 10 golpes al cuerpo/)).toBeInTheDocument();
    expect(mundo.pedidos.some((p) => p.ruta.startsWith("/rival"))).toBe(true);
  });

  it("CA-26.2: con datos de menos no hay lecturas y dice cuantos faltan", async () => {
    mundo.producto = productoBase({ objetivo: true, exposicion: true });
    mundo.rival = { ...rivalDePrueba(), lecturas: [], faltan: { minimo: 50, base: {} }, base: { A: { ataque: 49, exposicion: 90 }, B: { ataque: 80, exposicion: 70 } } };
    enRuta("/rival/?a=b1&b=b2");
    montar(<PantallaRival />);
    const bloque = await screen.findByRole("region", { name: "Ataque contra exposición" });
    expect(await within(bloque).findByText(/Faltan sesiones para cruzar zonas: 49 y 90 golpes/)).toBeInTheDocument();
    expect(within(bloque).queryByText("oportunidad")).toBeNull();
  });

  it("CA-26.1: con la altura apagada el bloque no esta disponible y no aparece ninguna zona", async () => {
    mundo.producto = productoBase({ objetivo: false, exposicion: false });
    enRuta("/rival/?a=b1&b=b2");
    montar(<PantallaRival />);
    const titulo = await screen.findByText("Ataque contra exposición");
    expect(within(titulo.closest("button")!).getByText("no disponible en esta versión")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/Hacia dónde tira/);
    expect(mundo.pedidos.some((p) => p.ruta.startsWith("/rival"))).toBe(false);
  });

  it("el frente a frente sigue mostrando lo medido, con lo condicionado en validacion", async () => {
    enRuta("/rival/?a=b1&b=b2");
    montar(<PantallaRival />);
    const tabla = await screen.findByRole("table", { name: "Frente a frente" });
    await waitFor(() => expect(mundo.pedidos.some((p) => p.ruta === "/producto")).toBe(true));
    expect(within(tabla).getAllByText("en validación")).toHaveLength(2);
    expect(within(tabla).getByText("1,80 m")).toBeInTheDocument();
  });
});
