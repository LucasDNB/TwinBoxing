/*
 * W11: perfil tactico (04 §06). CA-06.2 y los estados de cada bloque.
 */
import { describe, expect, it, beforeEach } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { conSesion, enRuta, montar } from "./entorno";
import { mundo, productoBase } from "./api-simulada";
import { perfilDePrueba } from "./perfil-de-prueba";
import { PantallaPeleador } from "@/components/pantallas/peleador/PantallaPeleador";

beforeEach(() => {
  conSesion();
  mundo.perfiles = { b1: perfilDePrueba("b1", "Ana Paz") };
  enRuta("/peleador/?id=b1");
});

describe("06 · Perfil táctico", () => {
  it("encabezado, filtro con cantidades y aviso de perfil preliminar", async () => {
    montar(<PantallaPeleador />);
    expect(await screen.findByRole("heading", { name: "Ana Paz" })).toBeInTheDocument();
    expect(screen.getByText("Welter · ortodoxa · 1,75 m · envergadura 1,80 m")).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Sparring · 2" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Bolsa · 0" })).toBeDisabled();
    expect(screen.getByText(/Perfil preliminar: con 5 sesiones o más/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Comparar en Rival Fantasma" })).toHaveAttribute("href", "/rival/?a=b1");
  });

  it("cada bloque con su estado: estimacion, medido, en validacion y no disponible", async () => {
    montar(<PantallaPeleador />);
    await screen.findByRole("heading", { name: "Ana Paz" });
    expect(screen.getByRole("region", { name: "Distribución de golpes" })).toHaveAttribute("data-estado", "estimacion");
    expect(screen.getByRole("region", { name: "Combinaciones frecuentes" })).toHaveAttribute("data-estado", "estimacion");
    expect(screen.getByRole("region", { name: "Evolución entre sesiones" })).toHaveAttribute("data-estado", "medido");
    await waitFor(() => expect(mundo.pedidos.some((p) => p.ruta === "/producto")).toBe(true));
    expect(screen.getByRole("region", { name: "Volumen por round promedio" })).toHaveAttribute("data-estado", "apagado");
    const tabla = screen.getByRole("table", { name: "Métricas por contexto" });
    expect(within(tabla).getAllByText("en validación")).toHaveLength(4);
    expect(screen.getByText("Mapa de exposición")).toBeInTheDocument();
    expect(screen.getByText("Avatar 3D")).toBeInTheDocument();
    expect(screen.getByText("1-2")).toBeInTheDocument();
    expect(screen.getByText("gancho")).toBeInTheDocument();
  });

  it("la evolucion se corta donde cambio el detector", async () => {
    montar(<PantallaPeleador />);
    expect(await screen.findByText(/La línea se corta donde cambió el detector/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /05\/10 · 15,1\/min/ })).toHaveAttribute("href", "/sesion/?id=s3&vista=fightcard");
  });

  it("con C2 y C3 aprobados aparecen los numeros", async () => {
    mundo.producto = productoBase({ comparacion_rounds: true, guardia: true });
    mundo.perfiles.b1 = perfilDePrueba("b1", "Ana Paz", {
      por_contexto: [{ contexto: "sparring", sesiones: 2, golpes_por_minuto: 12.8, jab_fraccion: 0.37, guardia_opuesta_arriba: 0.64, caida_ultimo_round: -0.18 }],
      rounds_promedio: [{ round: 1, golpes_por_minuto: 14 }, { round: 2, golpes_por_minuto: 12 }, { round: 3, golpes_por_minuto: 10 }],
    });
    montar(<PantallaPeleador />);
    expect(await screen.findByText("64%")).toBeInTheDocument();
    expect(screen.getByText("-18%")).toBeInTheDocument();
    expect(screen.getByText("10,0/min")).toBeInTheDocument();
  });

  it("filtrar por contexto pide el perfil de ese contexto", async () => {
    montar(<PantallaPeleador />);
    await userEvent.click(await screen.findByRole("radio", { name: "Combate · 1" }));
    await waitFor(() => expect(mundo.pedidos.some((p) => p.ruta === "/boxeadores/b1/perfil?contexto=combate")).toBe(true));
    expect(window.location.search).toBe("?id=b1&contexto=combate");
  });

  it("CA-06.2: al volver, el perfil se pide de nuevo y muestra la correccion", async () => {
    const { unmount } = montar(<PantallaPeleador />);
    expect(await screen.findByText("37% · 37")).toBeInTheDocument();
    unmount();
    // Una correccion en una sesion cambia la mezcla del perfil.
    (mundo.perfiles.b1 as any).totales.mezcla = { jab: 38, cross: 30, hook: 19, uppercut: 13 };
    montar(<PantallaPeleador />);
    expect(await screen.findByText("38% · 38")).toBeInTheDocument();
  });
});
