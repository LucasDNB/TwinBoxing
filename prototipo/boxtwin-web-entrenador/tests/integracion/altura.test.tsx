/*
 * W14 (ampliacion 03 §6 y §7). C8 no paso el 13-10 (0,726); el 14-10 se decidio mostrar la
 * altura igual, como estimacion y con lo medido en la nota, encendiendo la bandera `objetivo`
 * desde la API. Se prueban los dos lados: apagada no llega a ninguna pantalla (CA-14.2) y
 * encendida suma igual que los detectados (CA-14.1) y se corrige sin reprocesar (CA-14.3).
 */
import { describe, expect, it, beforeEach } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import { mmss } from "@/lib/formato";
import userEvent from "@testing-library/user-event";
import { conSesion, enRuta, montar } from "./entorno";
import { FIGHTCARDS, ID, JOBS, mundo, productoBase } from "./api-simulada";
import { perfilDePrueba } from "./perfil-de-prueba";
import { videoSimulado } from "./video";
import { PantallaSesion } from "@/components/pantallas/sesion/PantallaSesion";
import { PantallaPeleador } from "@/components/pantallas/peleador/PantallaPeleador";

const PALABRAS = /\b(cabeza|cuerpo)\b/i;

/* El texto de la pantalla sin el bloque no disponible, que se titula "A la cabeza y al cuerpo". */
function textoFueraDelBloque(): string {
  const copia = document.body.cloneNode(true) as HTMLElement;
  copia.querySelectorAll("[data-orientation]").forEach((n) => {
    if (/A la cabeza y al cuerpo/.test(n.textContent ?? "")) n.remove();
  });
  return copia.textContent ?? "";
}

beforeEach(() => {
  conSesion();
  videoSimulado();
  mundo.job = structuredClone(JOBS.listo);
  mundo.producto = productoBase({ objetivo: false });
});

describe("W14 · altura con la bandera apagada", () => {
  it("el fixture 0.3 trae la altura calculada sobre la sesion real", () => {
    const a = (FIGHTCARDS["0.3"] as any).peleadores.A;
    expect(a.golpes.some((g: any) => g.objetivo === "cuerpo")).toBe(true);
    expect(a.total.cabeza + a.total.cuerpo + a.total.objetivo_sin_dato).toBe(a.total.total);
  });

  it("CA-14.2: la Fight-Card no dice cabeza ni cuerpo fuera del bloque no disponible", async () => {
    enRuta(`/sesion/?id=${ID}&vista=fightcard`);
    montar(<PantallaSesion />);
    await screen.findByText("Golpes detectados", { selector: "span" });
    const titulo = screen.getByText("A la cabeza y al cuerpo");
    expect(within(titulo.closest("button")!).getByText("no disponible en esta versión")).toBeInTheDocument();
    expect(textoFueraDelBloque()).not.toMatch(PALABRAS);
  });

  it("CA-14.2: el análisis de sesión no rotula golpes con la altura", async () => {
    enRuta(`/sesion/?id=${ID}&vista=video`);
    montar(<PantallaSesion />);
    const lista = await screen.findByRole("region", { name: /Golpes de Peleador A/ });
    expect(within(lista).queryByText(PALABRAS)).toBeNull();
    expect(document.body.textContent).not.toMatch(PALABRAS);
  });

  it("CA-14.2: el perfil tampoco", async () => {
    mundo.perfiles = { b1: perfilDePrueba("b1", "Ana Paz") };
    enRuta("/peleador/?id=b1");
    montar(<PantallaPeleador />);
    await screen.findByRole("heading", { name: "Ana Paz" });
    expect(document.body.textContent).not.toMatch(PALABRAS);
  });
});

describe("W14 · altura con la bandera encendida (decision de producto del 14-10)", () => {
  beforeEach(() => {
    mundo.producto = productoBase({ objetivo: true });
  });

  it("CA-14.1: la Fight-Card muestra cabeza y cuerpo como estimacion, con lo medido y el sin dato", async () => {
    enRuta(`/sesion/?id=${ID}&vista=fightcard`);
    montar(<PantallaSesion />);
    const bloque = await screen.findByRole("region", { name: "A la cabeza y al cuerpo" });
    expect(bloque).toHaveAttribute("data-estado", "estimacion");
    const a = (FIGHTCARDS["0.3"] as any).peleadores.A.total;
    expect(within(bloque).getByLabelText(`Peleador A: ${a.cabeza} a la cabeza, ${a.cuerpo} al cuerpo`)).toBeInTheDocument();
    expect(within(bloque).getByText(/Exactitud medida: 73% sobre 806 golpes/)).toBeInTheDocument();
    expect(within(bloque).getByText(/Todavía no alcanza el 75%/)).toBeInTheDocument();
    expect(within(bloque).getByText(/^Sin dato: Peleador A 5 · Peleador B 11\./)).toBeInTheDocument();
    expect(screen.queryByText("no disponible en esta versión", { exact: false })).not.toBeNull(); // contacto sigue no disponible
  });

  it("el analisis rotula cada golpe con su altura estimada", async () => {
    enRuta(`/sesion/?id=${ID}&vista=video`);
    montar(<PantallaSesion />);
    const lista = await screen.findByRole("region", { name: /Golpes de Peleador A/ });
    expect((await within(lista).findAllByText(/≈ (cabeza|cuerpo)/)).length).toBeGreaterThan(0);
  });

  it("CA-14.3: corregir la altura de un golpe cambia la fila al volver, sin reprocesar", async () => {
    const fc = mundo.fightcard as any;
    const g = fc.peleadores.A.golpes.find((x: any) => x.objetivo === "cabeza" && !x.descartado);
    const antes = { ...fc.peleadores.A.total };
    enRuta(`/sesion/?id=${ID}&vista=video`);
    const { unmount } = montar(<PantallaSesion />);
    const lista = await screen.findByRole("region", { name: /Golpes de Peleador A/ });
    const fila = within(lista).getAllByRole("listitem").find((li) => li.querySelector(`[aria-label="Ir al golpe de ${mmss(g.t_inicio)}"]`))!;
    await userEvent.click(within(fila).getByRole("button", { name: /Corregir el golpe/ }));
    await userEvent.click(await screen.findByRole("radio", { name: "Al cuerpo" }));
    await userEvent.click(screen.getByRole("button", { name: "Guardar" }));
    await waitFor(() => expect(mundo.pedidos.some((x) => x.metodo === "PATCH" && (x.cuerpo as any)?.objetivo === "cuerpo")).toBe(true));
    expect(mundo.pedidos.find((x) => x.metodo === "PATCH")!.cuerpo).toEqual({ objetivo: "cuerpo" });
    unmount();

    enRuta(`/sesion/?id=${ID}&vista=fightcard`);
    montar(<PantallaSesion />);
    const bloque = await screen.findByRole("region", { name: "A la cabeza y al cuerpo" });
    expect(within(bloque).getByLabelText(`Peleador A: ${antes.cabeza - 1} a la cabeza, ${antes.cuerpo + 1} al cuerpo`)).toBeInTheDocument();
  });
});
