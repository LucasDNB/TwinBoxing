/*
 * B8 y W19 (ampliacion 02 §3) del lado de la web: sesiones de un solo peleador, con la
 * bandera encendida. CA-19.1 a CA-19.4.
 */
import { describe, expect, it, beforeEach, vi, afterEach } from "vitest";
import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { conSesion, enRuta, montar } from "./entorno";
import { usePathname } from "next/navigation";
import { FIGHTCARDS, ID, JOBS, mundo, productoBase } from "./api-simulada";
import { videoSimulado } from "./video";
import { PantallaNueva } from "@/components/pantallas/nueva/PantallaNueva";
import { PantallaIdentificar } from "@/components/pantallas/identificar/PantallaIdentificar";
import { PantallaSesion } from "@/components/pantallas/sesion/PantallaSesion";

/* Candidatos con B2: el 64 esta mas tiempo en cuadro; el 41 y el 228 coexisten con los mismos guantes. */
function esperaBolsa(contexto = "bolsa") {
  const job = { ...structuredClone(JOBS.espera_siembra), contexto } as any;
  const datos: Record<number, Record<string, unknown>> = {
    41: { t_desde: 10, t_hasta: 60, color_guante: { hex: "#C02020", nombre: "rojos" }, mismo_color_a_la_vez: [228] },
    64: { t_desde: 0, t_hasta: 170, color_guante: { hex: "#C02020", nombre: "rojos" }, mismo_color_a_la_vez: [] },
    228: { t_desde: 10, t_hasta: 58, color_guante: { hex: "#C02020", nombre: "rojos" }, mismo_color_a_la_vez: [41] },
  };
  job.candidatos = job.candidatos.map((c: any) => ({ ...c, ...(datos[c.track] ?? { t_desde: 100, t_hasta: 101 }) }));
  return job;
}

function fcIndividual(sinAsignar = 0.1) {
  const fc = structuredClone(FIGHTCARDS["0.3"]) as any;
  delete fc.peleadores.B;
  fc.detector.medido_en = "sparring";
  fc.identidad.sin_asignar = sinAsignar;
  // GET /fightcards/{id} suma el bloque sesion desde la base (05 §4).
  fc.sesion = { contexto: "bolsa", practica: null, boxeadores: { A: null, B: null } };
  return fc;
}

beforeEach(() => {
  conSesion();
  videoSimulado();
  mundo.producto = productoBase({ sesiones_individuales: true });
});
afterEach(() => vi.useRealTimers());

describe("B8 · un solo peleador", () => {
  it("con la bandera, bolsa se puede elegir y pide un solo rincon", async () => {
    enRuta("/sesiones/nueva/");
    montar(<PantallaNueva />);
    const bolsa = await screen.findByRole("radio", { name: /Bolsa/ });
    await waitFor(() => expect(bolsa).toBeEnabled());
    await userEvent.click(bolsa);
    expect(screen.queryByLabelText(/Rincón azul/)).toBeNull();
    expect(screen.getByLabelText(/Rincón rojo/)).toBeInTheDocument();
  });

  it("una Fight-Card solo con A se dibuja sin columna B", async () => {
    mundo.fightcard = fcIndividual();
    mundo.job = { ...structuredClone(JOBS.listo), contexto: "bolsa" };
    enRuta(`/sesion/?id=${ID}&vista=fightcard`);
    montar(<PantallaSesion />);
    const fila = (await screen.findByText("Golpes detectados", { selector: "span" })).closest("[data-fila]") as HTMLElement;
    expect(within(fila).getByText("51")).toBeInTheDocument();
    expect(screen.queryByText("Peleador B")).toBeNull();
  });
});

describe("W19 · identificación para una persona", () => {
  it("CA-19.1: sin A, sin B y sin rincones; el de más tiempo en cuadro primero y sugerido", async () => {
    mundo.job = esperaBolsa();
    enRuta(`/sesion/identificar/?id=${ID}`);
    montar(<PantallaIdentificar />);
    expect(await screen.findByRole("heading", { name: "¿Cuál es el boxeador?" })).toBeInTheDocument();
    expect(screen.getByText(/Tocá una vez al que está trabajando/)).toBeInTheDocument();
    const tarjetas = screen.getAllByRole("article");
    expect(within(tarjetas[0]).getByText("00:00 a 02:50")).toBeInTheDocument();
    expect(within(tarjetas[0]).getByText("sugerido")).toBeInTheDocument();
    expect(screen.getAllByText("sugerido")).toHaveLength(1);
    expect(screen.queryByRole("radio", { name: /Es A|Es B/ })).toBeNull();
    expect(document.body.textContent).not.toMatch(/rinc[oó]n/i);
  });

  it("una sola elección en toda la pantalla", async () => {
    mundo.job = esperaBolsa();
    enRuta(`/sesion/identificar/?id=${ID}`);
    montar(<PantallaIdentificar />);
    const tarjetas = await screen.findAllByRole("article");
    await userEvent.click(within(tarjetas[0]).getByRole("radio", { name: "Es él" }));
    await userEvent.click(within(tarjetas[1]).getByRole("radio", { name: "Es él" }));
    expect(within(tarjetas[0]).getByRole("radio", { name: "Es él" })).toHaveAttribute("aria-checked", "false");
    expect(within(tarjetas[1]).getByRole("radio", { name: "Es él" })).toHaveAttribute("aria-checked", "true");
  });

  it("confirma con el nombre del peleador y manda una sola semilla", async () => {
    mundo.job = { ...esperaBolsa(), boxeadores: { A: { id: "b1", nombre: "Tomás Ríos" }, B: null } };
    enRuta(`/sesion/identificar/?id=${ID}`);
    montar(<PantallaIdentificar />);
    const tarjetas = await screen.findAllByRole("article");
    await userEvent.click(within(tarjetas[0]).getByRole("radio", { name: "Es él" }));
    await userEvent.click(screen.getByRole("button", { name: "Completar análisis" }));
    const dialogo = await screen.findByRole("alertdialog");
    expect(within(dialogo).getByText("¿Este es Tomás Ríos?")).toBeInTheDocument();
    expect(within(dialogo).queryByText(/espejo/)).toBeNull();
    await userEvent.click(within(dialogo).getByRole("button", { name: "Sí, completar" }));
    await waitFor(() => expect(mundo.pedidos.find((p) => p.ruta.endsWith("/siembra"))?.cuerpo).toEqual({ track_a: 64, track_b: null }));
  });

  it("aviso de espejo: el elegido coexiste con guantes parecidos y se ven los dos recortes", async () => {
    mundo.job = esperaBolsa("sombra");
    enRuta(`/sesion/identificar/?id=${ID}`);
    montar(<PantallaIdentificar />);
    const tarjetas = await screen.findAllByRole("article");
    const del41 = tarjetas.find((t) => within(t).queryByText("00:10 a 01:00"))!;
    await userEvent.click(within(del41).getByRole("radio", { name: "Es él" }));
    await userEvent.click(screen.getByRole("button", { name: "Completar análisis" }));
    const dialogo = await screen.findByRole("alertdialog");
    expect(within(dialogo).getByText("¿Este es el boxeador?")).toBeInTheDocument();
    expect(within(dialogo).getByText(/Si estás frente a un espejo, elegí al de verdad, no al reflejo/)).toBeInTheDocument();
    expect(within(dialogo).getByText("a la vez, mismos guantes")).toBeInTheDocument();
    expect(within(dialogo).getAllByRole("img")).toHaveLength(2);
  });

  it("CA-19.2: marcada y confirmada, la sesión llega a lista y va a la Fight-Card sin otro paso", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    mundo.job = esperaBolsa();
    enRuta(`/sesion/identificar/?id=${ID}`);
    function App() {
      const ruta = usePathname();
      return ruta.startsWith("/sesion/identificar") ? <PantallaIdentificar /> : <PantallaSesion />;
    }
    montar(<App />);
    const tarjetas = await screen.findAllByRole("article");
    await userEvent.click(within(tarjetas[0]).getByRole("radio", { name: "Es él" }));
    await userEvent.click(screen.getByRole("button", { name: "Completar análisis" }));
    await userEvent.click(await screen.findByRole("button", { name: "Sí, completar" }));
    mundo.fightcard = fcIndividual(0);
    mundo.job = { ...structuredClone(JOBS.listo), contexto: "bolsa" };
    await act(async () => { await vi.advanceTimersByTimeAsync(3_500); });
    await waitFor(() => expect(window.location.pathname).toBe("/sesion/"));
    expect(new URLSearchParams(window.location.search).get("vista")).toBe("fightcard");
  });

  it("fase 2: un tramo es él o se ignora", async () => {
    mundo.job = { ...structuredClone(JOBS.listo), contexto: "bolsa" };
    mundo.fightcard = fcIndividual(0.2);
    mundo.tramos = { ...(mundo.tramos as any), cobertura: { A: 0.8, sin_asignar: 0.2 } };
    enRuta(`/sesion/identificar/?id=${ID}`);
    montar(<PantallaIdentificar />);
    const lista = await screen.findByRole("heading", { name: "Revisar tramos dudosos" });
    const items = within(lista.closest("section")!).getAllByRole("listitem");
    expect(within(items[0]).getAllByRole("radio").map((r) => r.textContent)).toEqual(["Es él", "Ignorar"]);
    await userEvent.click(within(items[0]).getByRole("radio", { name: "Es él" }));
    await userEvent.click(screen.getByRole("button", { name: /Aplicar/ }));
    await waitFor(() => expect(mundo.pedidos.find((p) => p.metodo === "PUT" && p.ruta.endsWith("/tramos"))?.cuerpo)
      .toEqual({ asignaciones: [{ track: 228, rol: "A" }] }));
  });

  it("CA-19.3: en bolsa la Fight-Card no muestra el margen de sparring como si fuera de bolsa", async () => {
    mundo.fightcard = fcIndividual();
    mundo.job = { ...structuredClone(JOBS.listo), contexto: "bolsa" };
    enRuta(`/sesion/?id=${ID}&vista=fightcard`);
    montar(<PantallaSesion />);
    expect(await screen.findByText(/se midió en sparring; en bolsa todavía no se midió/)).toBeInTheDocument();
    expect(screen.queryByText(/Acierta \d+ de cada 10/)).toBeNull();
  });

  it("en sparring el margen sigue como siempre", async () => {
    const fc = structuredClone(FIGHTCARDS["0.3"]) as any;
    fc.detector.medido_en = "sparring";
    mundo.fightcard = fc;
    mundo.job = structuredClone(JOBS.listo);
    enRuta(`/sesion/?id=${ID}&vista=fightcard`);
    montar(<PantallaSesion />);
    expect(await screen.findByText(/Acierta 9 de cada 10/)).toBeInTheDocument();
  });

  it("CA-19.4: con la bandera apagada bolsa sigue en construcción", async () => {
    mundo.producto = productoBase();
    enRuta("/sesiones/nueva/");
    montar(<PantallaNueva />);
    await userEvent.click(await screen.findByRole("radio", { name: /Bolsa/ }));
    expect(await screen.findByText(/todavía se están validando/)).toBeInTheDocument();
  });
});
