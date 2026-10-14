/*
 * W14 con la bandera apagada (ampliacion 03 §6 y §7). C8 no paso el 13-10, asi que W14 no se
 * construye: lo que se prueba es que la altura que trae la Fight-Card 0.3 no llegue a ninguna
 * pantalla (CA-14.2) y que el bloque siga "No disponible" con el texto nuevo.
 */
import { describe, expect, it, beforeEach } from "vitest";
import { screen, within } from "@testing-library/react";
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
