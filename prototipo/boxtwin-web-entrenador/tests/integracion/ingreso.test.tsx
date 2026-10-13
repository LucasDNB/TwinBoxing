/*
 * CA-00.1 y CA-00.2 (04 §00).
 */
import { describe, expect, it } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { conSesion, enRuta, montar } from "./entorno";
import { mundo } from "./api-simulada";
import Ingresar from "@/app/ingresar/page";
import { pedir } from "@/lib/api/cliente";
import { destinoSeguro } from "@/lib/navegacion";

describe("00 · Ingreso", () => {
  it("CA-00.1: con credenciales validas entra y va al panel", async () => {
    enRuta("/ingresar/");
    montar(<Ingresar />);
    await userEvent.type(screen.getByLabelText("Email"), "lucas@usal.edu.ar");
    await userEvent.type(screen.getByLabelText("Clave"), "sparring2026");
    await userEvent.click(screen.getByRole("button", { name: "Entrar" }));
    await waitFor(() => expect(window.location.pathname).toBe("/"));
    expect(localStorage.getItem("boxtwin.token")).toBe("token-bueno");
  });

  it("un 401 en el ingreso dice 'Email o clave incorrectos' sin decir cual campo", async () => {
    enRuta("/ingresar/");
    montar(<Ingresar />);
    await userEvent.type(screen.getByLabelText("Email"), "lucas@usal.edu.ar");
    await userEvent.type(screen.getByLabelText("Clave"), "otra-clave");
    await userEvent.click(screen.getByRole("button", { name: "Entrar" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Email o clave incorrectos.");
    expect(window.location.pathname).toBe("/ingresar/");
  });

  it("con registro por invitacion, 'Tengo una invitacion' pide el codigo y cambia el boton", async () => {
    enRuta("/ingresar/");
    montar(<Ingresar />);
    await userEvent.click(await screen.findByRole("button", { name: "Tengo una invitación" }));
    expect(screen.getByLabelText("Código de invitación")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Crear cuenta" })).toBeInTheDocument();
  });

  it("CA-00.2: un 401 en cualquier pantalla lleva a ingreso y, al entrar, vuelve ahi", async () => {
    conSesion("token-vencido");
    enRuta("/sesion/?id=abc&vista=fightcard");
    await expect(pedir("/jobs")).rejects.toThrow("se venció la sesión");
    expect(localStorage.getItem("boxtwin.token")).toBeNull();
    expect(window.location.pathname).toBe("/ingresar/");
    const siguiente = new URLSearchParams(window.location.search).get("siguiente");
    expect(siguiente).toBe("/sesion/?id=abc&vista=fightcard");

    montar(<Ingresar />);
    await userEvent.type(screen.getByLabelText("Email"), "lucas@usal.edu.ar");
    await userEvent.type(screen.getByLabelText("Clave"), "sparring2026");
    await userEvent.click(screen.getByRole("button", { name: "Entrar" }));
    await waitFor(() => expect(window.location.pathname).toBe("/sesion/"));
    expect(window.location.search).toBe("?id=abc&vista=fightcard");
    expect(mundo.pedidos.some((p) => p.ruta === "/auth/login")).toBe(true);
  });

  it("?siguiente= no sirve de redireccion abierta", () => {
    expect(destinoSeguro("//malo.com")).toBe("/");
    expect(destinoSeguro("https://malo.com")).toBe("/");
    expect(destinoSeguro("/peleador/?id=1")).toBe("/peleador/?id=1");
  });
});
