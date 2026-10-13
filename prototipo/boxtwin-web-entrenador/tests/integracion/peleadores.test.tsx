/*
 * W10: lista, alta y edicion de peleadores desde el celular.
 */
import { describe, expect, it, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { conSesion, enRuta, montar } from "./entorno";
import { mundo } from "./api-simulada";
import { PantallaPeleadores } from "@/components/pantallas/peleadores/PantallaPeleadores";

beforeEach(() => {
  conSesion();
  enRuta("/peleadores/");
});

describe("Lista de peleadores", () => {
  it("vacia, invita a cargar", async () => {
    montar(<PantallaPeleadores />);
    expect(await screen.findByText("Cargá a tus peleadores para armar sus perfiles")).toBeInTheDocument();
  });

  it("alta con altura y envergadura en metros con coma", async () => {
    montar(<PantallaPeleadores />);
    await userEvent.click((await screen.findAllByRole("button", { name: /Agregar/ }))[0]);
    await userEvent.type(await screen.findByLabelText("Nombre"), "Ana Paz");
    await userEvent.type(screen.getByLabelText("Categoría"), "Welter");
    await userEvent.selectOptions(screen.getByLabelText("Guardia"), "zurda");
    await userEvent.type(screen.getByLabelText("Altura (m)"), "1,72");
    await userEvent.type(screen.getByLabelText("Envergadura (m)"), "1,78");
    await userEvent.click(screen.getByRole("button", { name: "Guardar" }));
    await waitFor(() => {
      const p = mundo.pedidos.find((x) => x.metodo === "POST" && x.ruta === "/boxeadores");
      expect(p?.cuerpo).toMatchObject({ nombre: "Ana Paz", categoria: "Welter", guardia: "zurda", altura_m: 1.72, envergadura_m: 1.78 });
    });
    expect(await screen.findByRole("link", { name: /Ana Paz/ })).toHaveAttribute("href", "/peleador/?id=b1");
  });

  it("una altura en centimetros se rechaza en el cliente", async () => {
    montar(<PantallaPeleadores />);
    await userEvent.click((await screen.findAllByRole("button", { name: /Agregar/ }))[0]);
    await userEvent.type(await screen.findByLabelText("Nombre"), "Ana");
    await userEvent.type(screen.getByLabelText("Altura (m)"), "172");
    await userEvent.click(screen.getByRole("button", { name: "Guardar" }));
    expect(await screen.findByText("La altura va en metros, entre 1,20 y 2,20.")).toBeInTheDocument();
    expect(mundo.pedidos.some((x) => x.metodo === "POST")).toBe(false);
  });

  it("editar manda PATCH con los datos", async () => {
    mundo.boxeadores = [{ id: "b1", nombre: "Ana", guardia: null, notas: null, creado: "2026-10-01", categoria: null, altura_m: null, envergadura_m: null, sesiones: 2, ultima_sesion: "2026-10-10T10:00:00" }];
    montar(<PantallaPeleadores />);
    expect(await screen.findByText("2 sesiones · última 10/10")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Editar a Ana" }));
    await userEvent.type(await screen.findByLabelText("Categoría"), "Liviano");
    await userEvent.click(screen.getByRole("button", { name: "Guardar" }));
    await waitFor(() => expect(mundo.pedidos.find((x) => x.metodo === "PATCH")?.cuerpo).toMatchObject({ categoria: "Liviano" }));
  });
});
