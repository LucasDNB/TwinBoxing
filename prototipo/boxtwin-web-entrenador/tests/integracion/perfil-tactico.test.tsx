/*
 * Cambios del 14-10 al perfil tactico (specs/cambios-14-10/01 §4): W20 a W24.
 */
import { describe, expect, it, beforeEach } from "vitest";
import { screen, within } from "@testing-library/react";
import { conSesion, enRuta, montar } from "./entorno";
import { mundo, productoBase } from "./api-simulada";
import { perfilDePrueba, zonasDePrueba } from "./perfil-de-prueba";
import { PantallaPeleador } from "@/components/pantallas/peleador/PantallaPeleador";

const PROHIBIDAS = /recibió|conectó|le pegaron|alcanzad/i;

function perfil(extra: Record<string, unknown> = {}) {
  return perfilDePrueba("b1", "Ana Paz", {
    ataque: zonasDePrueba([40, 20, 10, 10]),
    exposicion: {
      ...zonasDePrueba([30, 30, 10, 20]),
      tipo: "dirigidos", sesiones: 3,
      por_rival: [{ boxeador_id: "b2", nombre: "Bea Ruiz", total: 60 }, { boxeador_id: null, nombre: "sin perfil", total: 35 }],
    },
    ...extra,
  });
}

beforeEach(() => {
  conSesion();
  mundo.producto = productoBase({ objetivo: true, exposicion: true, comparacion_rounds: true });
  mundo.perfiles = { b1: perfil() };
  enRuta("/peleador/?id=b1");
});

describe("W20 · distribucion por mano y altura", () => {
  it("CA-20.1: mano medida, celdas estimadas que suman los golpes con altura", async () => {
    montar(<PantallaPeleador />);
    const bloque = await screen.findByRole("region", { name: "Distribución de golpes" });
    const tabla = within(bloque).getByRole("table", { name: "Golpes por mano y altura" });
    const cabeza = within(tabla).getByRole("row", { name: /Cabeza/ });
    expect(within(cabeza).getByText("40 · 50%")).toBeInTheDocument();
    expect(within(cabeza).getByText("60 · 75%")).toBeInTheDocument();
    const cuerpo = within(tabla).getByRole("row", { name: /Cuerpo/ });
    expect(within(cuerpo).getByText("20 · 25%")).toBeInTheDocument();
    expect(within(bloque).getByText("53 · 62%")).toBeInTheDocument(); // izquierda: 40 + 10 + 3 de 85
    expect(within(bloque).getByText(/encuentra cerca de 3 de cada 10 golpes al cuerpo/)).toBeInTheDocument();
  });

  it("con la altura apagada solo queda la fila por mano", async () => {
    mundo.producto = productoBase({ objetivo: false });
    montar(<PantallaPeleador />);
    const bloque = await screen.findByRole("region", { name: "Distribución de golpes" });
    expect(within(bloque).queryByRole("table")).toBeNull();
    expect(within(bloque).getByText("Por mano")).toBeInTheDocument();
  });
});

describe("W21 · volumen por round", () => {
  it("segmentos de 3 minutos con S y el rotulo de C2 sin validar", async () => {
    mundo.perfiles = { b1: perfil({ rounds_promedio: [
      { round: 1, golpes_por_minuto: 14.2, origen: "segmentos_180s" },
      { round: 2, golpes_por_minuto: 11.0, origen: "segmentos_180s" },
    ] }) };
    montar(<PantallaPeleador />);
    const bloque = await screen.findByRole("region", { name: "Volumen por round promedio" });
    expect(bloque).toHaveAttribute("data-estado", "medido");
    expect(within(bloque).getByText("S1")).toBeInTheDocument();
    expect(within(bloque).getByText("comparación entre rounds sin validar")).toBeInTheDocument();
    expect(within(bloque).getByText(/se dividió en segmentos de 3 minutos/)).toBeInTheDocument();
  });
});

describe("W22 · mapa de exposicion", () => {
  it("dirigidos, corregido, con tabla accesible y sin palabras de contacto", async () => {
    montar(<PantallaPeleador />);
    const bloque = await screen.findByRole("region", { name: "Hacia dónde le tiran" });
    expect(bloque).toHaveAttribute("data-estado", "estimacion");
    expect(within(bloque).getByText(/No dice si tocaron/)).toBeInTheDocument();
    expect(within(bloque).getByText(/Corregido por el detector/)).toBeInTheDocument();
    const filas = within(bloque).getByRole("table").querySelectorAll("tbody tr");
    expect(filas).toHaveLength(4);
    expect(within(filas[0] as HTMLElement).getByText("30")).toBeInTheDocument();
    expect(within(bloque).getByRole("img", { name: /Silueta con las cuatro zonas/ })).toBeInTheDocument();
    expect(bloque.textContent).not.toMatch(PROHIBIDAS);
    expect(within(bloque).queryByText(/El mapa describe sobre todo/)).toBeNull(); // 60 de 95 no pasa del 70%
  });

  it("con menos de 50 golpes es preliminar y no colorea", async () => {
    mundo.perfiles = { b1: perfil({ exposicion: { ...zonasDePrueba([10, 10, 5, 5]), tipo: "dirigidos", sesiones: 1, por_rival: [] } }) };
    montar(<PantallaPeleador />);
    const bloque = await screen.findByRole("region", { name: "Hacia dónde le tiran" });
    expect(within(bloque).getByText(/Mapa preliminar: 30 golpes dirigidos/)).toBeInTheDocument();
    for (const z of bloque.querySelectorAll("[data-zona] path")) expect(z.getAttribute("fill")).toBe("transparent");
  });

  it("un rival que aporta mas del 70% se nombra", async () => {
    mundo.perfiles = { b1: perfil({ exposicion: { ...zonasDePrueba([30, 30, 10, 20]), tipo: "dirigidos", sesiones: 2, por_rival: [{ boxeador_id: "b2", nombre: "Bea Ruiz", total: 90 }] } }) };
    montar(<PantallaPeleador />);
    expect(await screen.findByText(/El mapa describe sobre todo a Bea Ruiz/)).toBeInTheDocument();
  });

  it("en bolsa no hay rival", async () => {
    mundo.perfiles = { b1: perfil({ exposicion: null }) };
    enRuta("/peleador/?id=b1&contexto=bolsa");
    montar(<PantallaPeleador />);
    const bloque = await screen.findByRole("region", { name: "Hacia dónde le tiran" });
    expect(within(bloque).getByText("En estos contextos no hay rival.")).toBeInTheDocument();
  });

  it("CA-15.3: con la bandera apagada sigue no disponible", async () => {
    mundo.producto = productoBase({ objetivo: false, exposicion: false });
    montar(<PantallaPeleador />);
    const titulo = await screen.findByText("Mapa de exposición");
    expect(within(titulo.closest("button")!).getByText("no disponible en esta versión")).toBeInTheDocument();
  });
});

describe("W23 y W24 · avatar y distancia", () => {
  it("el avatar es ilustrativo y describe sus medidas", async () => {
    montar(<PantallaPeleador />);
    const bloque = await screen.findByRole("region", { name: "Avatar 3D" });
    expect(within(bloque).getByText("ilustrativo, no reconstruido del video")).toBeInTheDocument();
    expect(within(bloque).getByRole("img", { name: /Maniquí en guardia ortodoxa, 1,75 m de altura y 1,80 m de envergadura/ })).toBeInTheDocument();
  });

  it("CA-24.1: distancia y estilo cargados a mano se ven en el bloque y en el encabezado", async () => {
    mundo.perfiles = { b1: perfil({ boxeador: { id: "b1", nombre: "Ana Paz", guardia: "ortodoxa", notas: null, categoria: "Welter", altura_m: 1.75, envergadura_m: 1.8, distancia: "corta", estilo: "fajador" } }) };
    montar(<PantallaPeleador />);
    const bloque = await screen.findByRole("region", { name: "Distancia de trabajo" });
    expect(within(bloque).getByText("Corta")).toBeInTheDocument();
    expect(within(bloque).getByText("Fajador")).toBeInTheDocument();
    expect(within(bloque).getByText("cargado a mano")).toBeInTheDocument();
    expect(screen.getByText(/fajador · distancia corta/)).toBeInTheDocument();
  });

  it("sin cargar, invita a cargarlos", async () => {
    montar(<PantallaPeleador />);
    const bloque = await screen.findByRole("region", { name: "Distancia de trabajo" });
    expect(within(bloque).getByRole("link", { name: "Cargarlos en Peleadores" })).toHaveAttribute("href", "/peleadores/");
  });
});
