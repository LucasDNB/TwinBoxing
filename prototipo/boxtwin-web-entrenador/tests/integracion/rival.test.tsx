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

  it("CA-07.1: con C2 y C3 apagados ninguna lectura cita caida ni guardia opuesta", async () => {
    enRuta("/rival/?a=b1&b=b2");
    montar(<PantallaRival />);
    await screen.findByText("Oportunidades y amenazas");
    await waitFor(() => expect(mundo.pedidos.some((p) => p.ruta === "/producto")).toBe(true));
    const bloque = screen.getByRole("region", { name: "Oportunidades y amenazas" });
    expect(bloque.textContent).not.toMatch(/último round|otra mano/);
    // Guardias opuestas y ritmo si aplican (12,8 / 15,1 promedio contra 10).
    expect(within(bloque).getAllByText(/amenaza · guardia$/)).toHaveLength(2);
    expect(within(bloque).getByText(/oportunidad · golpes por minuto/)).toBeInTheDocument();
    const tabla = screen.getByRole("table", { name: "Frente a frente" });
    expect(within(tabla).getAllByText("en validación")).toHaveLength(2);
  });

  it("con C2 y C3 aprobados entran caida y guardia opuesta", async () => {
    mundo.producto = productoBase({ comparacion_rounds: true, guardia: true });
    (mundo.perfiles.b1 as any).por_contexto[0].guardia_opuesta_arriba = 0.7;
    (mundo.perfiles.b1 as any).por_contexto[1].guardia_opuesta_arriba = 0.7;
    enRuta("/rival/?a=b1&b=b2");
    montar(<PantallaRival />);
    expect(await screen.findByText("Su volumen cae 30% en el último round.")).toBeInTheDocument();
    expect(screen.getByText(/Deja la otra mano abajo más seguido/)).toBeInTheDocument();
  });
});
