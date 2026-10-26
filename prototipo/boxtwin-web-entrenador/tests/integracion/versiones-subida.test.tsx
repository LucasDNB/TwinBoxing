/*
 * W30 y W33: subir en Individual (03 §I-02) y en Gimnasio (03 §G-02). CA-I02.1 a CA-I02.3 y
 * CA-G02.1. Las horas salen de cuenta.horas y la duracion del archivo (B20).
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { conSesion, enRuta, montar } from "./entorno";
import { mundo, productoBase } from "./api-simulada";
import { cuentaGimnasio, cuentaIndividual } from "../fixtures/cuentas";

const subidas: { datos: FormData; resolver: (r: unknown) => void }[] = [];

vi.mock("@/lib/api/cliente", async (orig) => {
  const real = await orig<typeof import("@/lib/api/cliente")>();
  return {
    ...real,
    subirConProgreso: (datos: FormData) => {
      let resolver!: (r: unknown) => void;
      const promesa = new Promise((a) => { resolver = a; });
      subidas.push({ datos, resolver });
      return { promesa, cancelar: () => {} };
    },
  };
});

vi.mock("@/lib/subida", async (orig) => {
  const real = await orig<typeof import("@/lib/subida")>();
  return { ...real, leerMetadatos: async () => ({ duracion_s: 180, ancho: 1920, alto: 1080 }) };
});

import { PantallaNueva } from "@/components/pantallas/nueva/PantallaNueva";

function video() {
  const f = new File(["x"], "bolsa.mp4", { type: "video/mp4" });
  Object.defineProperty(f, "size", { value: 1000 });
  return f;
}

async function elegirVideo() {
  fireEvent.change(screen.getByTestId("input-galeria"), { target: { files: [video()] } });
  await screen.findByText(/bolsa\.mp4 ·/);
}

beforeEach(() => {
  subidas.length = 0;
  conSesion();
  enRuta("/sesiones/nueva/");
  mundo.producto = productoBase({ sesiones_individuales: true });
});

describe("W30 · subir en Individual", () => {
  beforeEach(() => { mundo.cuenta = cuentaIndividual(); });

  it("CA-I02.1: no hay selector de peleadores y no se asigna nada despues de subir", async () => {
    montar(<PantallaNueva />);
    expect(await screen.findByTestId("aparece-como")).toHaveTextContent("Vas a aparecer como Lucas");
    expect(screen.queryByLabelText(/Rincón/)).toBeNull();
    await userEvent.click(screen.getByRole("radio", { name: /Bolsa/ }));
    expect(screen.getByText(/todavía no se midió en bolsa y sombra/)).toBeInTheDocument();
    await elegirVideo();
    await userEvent.click(screen.getByLabelText(/Soy yo quien aparece en el video/));
    await userEvent.click(screen.getByRole("button", { name: "Subir y procesar" }));
    expect(subidas).toHaveLength(1);
    expect(subidas[0].datos.get("contexto")).toBe("bolsa");
    expect(subidas[0].datos.get("duracion_declarada_s")).toBe("180");
    await act(async () => subidas[0].resolver({ job_id: "nueva", estado: "en_cola", bytes: 1000 }));
    await waitFor(() => expect(window.location.pathname).toBe("/sesion/"));
    // La API pone al propio como A (B22): la web no manda ninguna asignacion.
    expect(mundo.pedidos.some((p) => p.metodo === "PUT" && p.ruta.includes("/boxeadores"))).toBe(false);
  });

  it("CA-I02.2: en sparring, sin la segunda casilla no se puede enviar", async () => {
    montar(<PantallaNueva />);
    await screen.findByTestId("aparece-como");
    await elegirVideo();
    await userEvent.click(screen.getByLabelText(/Soy yo quien aparece en el video/));
    const subir = screen.getByRole("button", { name: "Subir y procesar" });
    expect(subir).toBeDisabled();
    await userEvent.click(screen.getByLabelText(/La otra persona que aparece aceptó/));
    expect(subir).toBeEnabled();
  });

  it("las horas dicen cuanto usa la sesion de lo que queda", async () => {
    montar(<PantallaNueva />);
    await screen.findByTestId("aparece-como");
    await elegirVideo();
    expect(screen.getByTestId("horas-sesion")).toHaveTextContent("Esta sesión usa unos 3 min de las 2,8 h que te quedan.");
  });

  it("CA-I02.3: con bloquear y sin horas el boton queda deshabilitado con el motivo", async () => {
    const c = cuentaIndividual();
    c.horas = { ...c.horas, usadas: 4, restantes: 0, modo: "bloquear" };
    mundo.cuenta = c;
    montar(<PantallaNueva />);
    await screen.findByTestId("aparece-como");
    await elegirVideo();
    await userEvent.click(screen.getByLabelText(/Soy yo quien aparece en el video/));
    await userEvent.click(screen.getByLabelText(/La otra persona que aparece aceptó/));
    expect(screen.getByRole("button", { name: "Subir y procesar" })).toBeDisabled();
    expect(screen.getAllByText("Esta sesión usaría 0,1 h y quedan 0,0 h este mes.").length).toBeGreaterThan(0);
  });

  it("con la subida apagada en la version de prueba, lo dice y no deja subir", async () => {
    mundo.producto = { ...productoBase({ sesiones_individuales: true }), demo: { activo: true, subir: false, perfiles: [] } } as never;
    montar(<PantallaNueva />);
    expect(await screen.findByRole("status")).toHaveTextContent("En la versión de prueba la subida está apagada en este momento.");
  });
});

describe("W33 · subir en Gimnasio", () => {
  beforeEach(() => {
    mundo.cuenta = cuentaGimnasio("entrenador");
    mundo.boxeadores = [
      { id: "b1", nombre: "Tomás Ríos", guardia: null, notas: null, creado: "2026-10-01T10:00:00", consentimiento: { estado: "vigente", desde: "2026-10-26T10:00:00" } },
      { id: "b2", nombre: "Nico Ferreyra", guardia: null, notas: null, creado: "2026-10-01T10:00:00", consentimiento: { estado: "falta", desde: null } },
    ];
  });

  it("CA-G02.1: un atleta sin consentimiento vigente no se puede elegir y se ofrece registrarlo", async () => {
    montar(<PantallaNueva />);
    const selectA = await screen.findByLabelText(/Rincón rojo/);
    // Una opcion por selector (A y B): las dos deshabilitadas.
    await waitFor(() => expect(screen.getAllByRole("option", { name: /Nico Ferreyra · Falta el consentimiento/ })).toHaveLength(2));
    for (const o of screen.getAllByRole("option", { name: /Nico Ferreyra · Falta el consentimiento/ })) expect(o).toBeDisabled();
    expect(screen.getAllByRole("option", { name: "Tomás Ríos" })[0]).toBeEnabled();
    expect(screen.getByText("Falta el consentimiento de Nico Ferreyra.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Registrarlo" }));
    expect(await screen.findByText("Consentimiento de Nico Ferreyra")).toBeInTheDocument();
    expect(selectA).toBeInTheDocument();
  });
});
