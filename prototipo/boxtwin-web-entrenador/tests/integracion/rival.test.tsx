/*
 * W12: Rival Fantasma (04 §07). CA-07.1 y CA-07.2.
 */
import { describe, expect, it, beforeEach } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import { conSesion, enRuta, montar } from "./entorno";
import { mundo, productoBase } from "./api-simulada";
import { perfilDePrueba } from "./perfil-de-prueba";
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

  it("dice que no es un pronostico, dibuja la silueta que puede y dice que falta", async () => {
    enRuta("/rival/?a=b1&b=b2");
    montar(<PantallaRival />);
    expect(await screen.findByText(/No es un pronóstico ni una probabilidad de victoria/)).toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /Ana Paz: 1,75 m de altura y 1,80 m de envergadura/ })).toBeInTheDocument();
    expect(screen.getByText(/De Bea Ruiz falta la envergadura/)).toBeInTheDocument();
  });

  it("oportunidades y amenazas: En construcción, sin ninguna lectura (CA-07.1 se cumple siempre)", async () => {
    mundo.producto = productoBase({ comparacion_rounds: true, guardia: true });
    enRuta("/rival/?a=b1&b=b2");
    montar(<PantallaRival />);
    const bloque = await screen.findByRole("region", { name: "Oportunidades y amenazas" });
    expect(bloque).toHaveAttribute("data-estado", "en_construccion");
    expect(bloque).toHaveTextContent("en construcción");
    expect(bloque.textContent).not.toMatch(/oportunidad ·|amenaza ·|último round|otra mano/);
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
