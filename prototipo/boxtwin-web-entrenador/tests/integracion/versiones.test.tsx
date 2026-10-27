/*
 * Versiones Individual y Gimnasio (specs/versiones-26-10, 03 y 05 §4): navegacion por version
 * y rol, ingreso de prueba, Mi progreso, marca y Fight-Card en segunda persona, panel,
 * plantel, consentimiento y equipo del gimnasio.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { conSesion, enRuta, montar } from "./entorno";
import { FIGHTCARDS, ID, JOBS, mundo, productoBase } from "./api-simulada";
import { videoSimulado } from "./video";
import { cuentaEntrenador, cuentaGimnasio, cuentaIndividual } from "../fixtures/cuentas";
import { BarraInferior, BarraLateral, FranjaDemo, destinosDe } from "@/components/boxtwin/Navegacion";
import { estadoDe } from "@/lib/banderas";
import Ingresar from "@/app/ingresar/page";
import PaginaRival from "@/app/(app)/rival/page";
import PaginaInicio from "@/app/(app)/page";
import { PantallaIdentificar } from "@/components/pantallas/identificar/PantallaIdentificar";
import { PantallaSesion } from "@/components/pantallas/sesion/PantallaSesion";
import { PantallaPeleador } from "@/components/pantallas/peleador/PantallaPeleador";
import { PantallaPeleadores } from "@/components/pantallas/peleadores/PantallaPeleadores";
import { PantallaEquipo } from "@/components/pantallas/equipo/PantallaEquipo";
import { EsperaSesion, textoDeCola } from "@/components/pantallas/sesion/EsperaSesion";
import { lineasPorContexto } from "@/components/pantallas/progreso/PantallaProgreso";
import { perfilDePrueba } from "./perfil-de-prueba";
import type { Cuenta, Perfil, Sesion } from "@/lib/api/esquemas";

beforeEach(() => {
  conSesion();
  videoSimulado();
});

const comoCuenta = (c: unknown) => c as Cuenta;

function productoDemo(activo = true) {
  return {
    ...productoBase({ sesiones_individuales: true }),
    cupo: "informar",
    demo: activo
      ? {
          activo: true, subir: true,
          perfiles: [
            { perfil: "individual", nombre: "Individual", version: "individual", rol: "titular", descripcion: "Un boxeador con su bolsa, su sombra y sus sparrings.", disponible: true },
            { perfil: "entrenador", nombre: "Entrenador", version: "entrenador", rol: "titular", descripcion: "Un entrenador con sus peleadores.", disponible: true },
            { perfil: "gimnasio_titular", nombre: "Gimnasio · titular", version: "gimnasio", rol: "titular", descripcion: "Administra el gimnasio: plantel, equipo y horas.", disponible: true },
            { perfil: "gimnasio_entrenador", nombre: "Gimnasio · entrenador", version: "gimnasio", rol: "entrenador", descripcion: "Un entrenador del mismo gimnasio.", disponible: false },
          ],
        }
      : { activo: false },
    versiones: { individual: { horas_mes: 4 }, entrenador: { horas_mes: 15 }, gimnasio: { horas_mes: 50 } },
  };
}

// ---------------------------------------------------------------------------
// W27 · navegacion
// ---------------------------------------------------------------------------

describe("W27 · la navegacion sale de la cuenta", () => {
  const nombres = (c: unknown, barra: "inferior" | "lateral") => destinosDe(comoCuenta(c))[barra].map((d) => d.nombre);

  it("CA-27.1: cada version y rol tiene exactamente sus destinos", () => {
    expect(nombres(cuentaIndividual(), "inferior")).toEqual(["Inicio", "Sesiones", "Subir", "Mi perfil"]);
    expect(nombres(cuentaIndividual(), "lateral")).toEqual(["Inicio", "Sesiones", "Subir", "Mi perfil"]);
    expect(nombres(cuentaEntrenador(), "inferior")).toEqual(["Panel", "Sesiones", "Subir", "Peleadores", "Rival"]);
    expect(nombres(cuentaGimnasio("entrenador"), "inferior")).toEqual(["Panel", "Sesiones", "Subir", "Plantel", "Rival"]);
    expect(nombres(cuentaGimnasio("titular"), "inferior")).toEqual(["Panel", "Sesiones", "Subir", "Plantel", "Equipo"]);
    expect(nombres(cuentaGimnasio("titular"), "lateral")).toEqual(["Panel", "Sesiones", "Subir", "Plantel", "Rival", "Equipo"]);
  });

  it("CA-27.1 en pantalla: la barra inferior de Individual tiene cuatro destinos y Mi perfil va al propio", () => {
    enRuta("/");
    montar(<BarraInferior cuenta={comoCuenta(cuentaIndividual())} />);
    const nav = screen.getByRole("navigation", { name: "Principal" });
    const enlaces = within(nav).getAllByRole("link");
    expect(enlaces.map((a) => a.textContent)).toEqual(["Inicio", "Sesiones", "Subir", "Mi perfil"]);
    expect(enlaces[3]).toHaveAttribute("href", "/peleador/?id=propio");
  });

  it("la barra lateral dice la version y, en Gimnasio, el nombre", () => {
    enRuta("/");
    montar(<BarraLateral usuario="" cuenta={comoCuenta(cuentaGimnasio("titular"))} />);
    expect(screen.getByText("versión gimnasio")).toBeInTheDocument();
    expect(screen.getByText("Gimnasio de prueba")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Rival Fantasma/ })).toBeInTheDocument();
  });

  it("CA-27.2: en Individual el Rival Fantasma no existe, ni siquiera rotulado", () => {
    const p = productoBase({ exposicion: true });
    for (const b of ["07.seleccion", "07.frente", "07.cruces"] as const) {
      expect(estadoDe(b, p as never, comoCuenta(cuentaIndividual()))).toBe("fuera");
      expect(estadoDe(b, p as never, comoCuenta(cuentaEntrenador()))).not.toBe("fuera");
    }
    expect(estadoDe("01.plan", p as never, comoCuenta(cuentaIndividual()))).toBe("medido");
    expect(estadoDe("01.plan", p as never)).toBe("fuera");
  });

  it("CA-I06.1: /rival en Individual muestra el aviso y no la pantalla", async () => {
    mundo.cuenta = cuentaIndividual();
    enRuta("/rival/");
    montar(<PaginaRival />);
    expect(await screen.findByText("Esta función no está en la versión Individual.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Ir al inicio" })).toBeInTheDocument();
    expect(mundo.pedidos.some((x) => x.ruta.startsWith("/rival"))).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// W38 · ingreso de prueba
// ---------------------------------------------------------------------------

describe("W38 · ingreso con un boton por version", () => {
  it("CA-00.7: con la version de prueba no hay ningun campo de texto", async () => {
    localStorage.clear();
    mundo.producto = productoDemo() as never;
    enRuta("/ingresar/");
    const { container } = montar(<Ingresar />);
    expect(await screen.findByText("Versión de prueba")).toBeInTheDocument();
    expect(container.querySelectorAll("input")).toHaveLength(0);
    expect(screen.queryByText("Crear una cuenta")).toBeNull();
    const botones = screen.getAllByRole("button");
    expect(botones.map((b) => b.querySelector("span")?.textContent)).toEqual([
      "Individual", "Entrenador", "Gimnasio · titular", "Gimnasio · entrenador",
    ]);
    expect(screen.getByRole("button", { name: /Individual/ })).toHaveTextContent("4 h de video por mes");
    // Un perfil no disponible se ve deshabilitado con "En construcción".
    expect(screen.getByRole("button", { name: /Gimnasio · entrenador/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Gimnasio · entrenador/ })).toHaveTextContent("En construcción");
  });

  it("CA-00.8: un boton entra con su perfil y va al inicio", async () => {
    localStorage.clear();
    mundo.producto = productoDemo() as never;
    enRuta("/ingresar/");
    montar(<Ingresar />);
    await userEvent.click(await screen.findByRole("button", { name: /Gimnasio · titular/ }));
    await waitFor(() => expect(window.location.pathname).toBe("/"));
    expect(localStorage.getItem("boxtwin.token")).toBe("token-bueno");
    expect(mundo.pedidos.find((x) => x.ruta === "/auth/demo")?.cuerpo).toEqual({ perfil: "gimnasio_titular" });
  });

  it("CA-00.9: apagada, la pagina es la de siempre y los botones no estan", async () => {
    localStorage.clear();
    mundo.producto = productoDemo(false) as never;
    enRuta("/ingresar/");
    montar(<Ingresar />);
    expect(await screen.findByLabelText("Email")).toBeInTheDocument();
    expect(screen.queryByText("Versión de prueba")).toBeNull();
    expect(screen.queryByRole("button", { name: /Gimnasio · titular/ })).toBeNull();
  });

  it("CA-00.10: el rotulo dice la cuenta y Salir borra la sesion y vuelve al ingreso con recarga", async () => {
    const assign = vi.fn();
    const original = window.location;
    Object.defineProperty(window, "location", { configurable: true, value: { ...original, assign } });
    try {
      montar(<FranjaDemo cuenta={comoCuenta(cuentaGimnasio("entrenador"))} />);
      expect(screen.getByTestId("rotulo-demo")).toHaveTextContent("Prueba · Gimnasio · entrenador");
      await userEvent.click(screen.getByRole("button", { name: "Salir" }));
      expect(localStorage.getItem("boxtwin.token")).toBeNull();
      // Recarga completa: no queda nada de la cuenta anterior en la cache de consultas.
      expect(assign).toHaveBeenCalledWith("/ingresar/");
    } finally {
      Object.defineProperty(window, "location", { configurable: true, value: original });
    }
  });

  it("una cuenta real no tiene franja de prueba", () => {
    montar(<FranjaDemo cuenta={comoCuenta(cuentaEntrenador())} />);
    expect(screen.queryByTestId("rotulo-demo")).toBeNull();
    expect(screen.queryByRole("button", { name: "Salir" })).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// W29 · Mi progreso
// ---------------------------------------------------------------------------

function sesion(id: string, contexto: string, creada: string, estado = "listo"): Sesion {
  return { ...structuredClone(JOBS.listo), id, contexto, creada, estado, boxeadores: { A: { id: "propio", nombre: "Lucas" }, B: null } } as unknown as Sesion;
}

function perfilCon(evolucion: Perfil["evolucion"]): Perfil {
  return perfilDePrueba("propio", "Lucas", { evolucion }) as unknown as Perfil;
}

describe("W29 · Mi progreso", () => {
  const hoy = new Date();
  const dias = (n: number) => new Date(hoy.getTime() - n * 86400000).toISOString();

  it("CA-I01.2: la tendencia nunca mezcla contextos", () => {
    const ss = [sesion("b1", "bolsa", dias(1)), sesion("s1", "sombra", dias(2)), sesion("b2", "bolsa", dias(3))];
    const p = perfilCon([
      { fecha: dias(3), sesion_id: "b2", golpes: 100, golpes_por_minuto: 30 },
      { fecha: dias(2), sesion_id: "s1", golpes: 50, golpes_por_minuto: 20 },
      { fecha: dias(1), sesion_id: "b1", golpes: 110, golpes_por_minuto: 33 },
    ]);
    const lineas = lineasPorContexto(p, ss, hoy);
    expect(lineas.map((l) => [l.contexto, l.puntos.map((x) => x.sesion_id)])).toEqual([
      ["bolsa", ["b2", "b1"]], ["sombra", ["s1"]],
    ]);
  });

  it("CA-I01.3: con una sola sesion de un contexto no hay linea y aparece el texto", async () => {
    mundo.cuenta = cuentaIndividual();
    const s = sesion(ID, "bolsa", dias(1));
    mundo.job = s as never;
    mundo.perfiles.propio = perfilDePrueba("propio", "Lucas", { evolucion: [{ fecha: dias(1), sesion_id: ID, golpes: 87, golpes_por_minuto: 25.4 }] });
    enRuta("/");
    montar(<PaginaInicio />);
    expect(await screen.findByRole("heading", { name: "Mi progreso" })).toBeInTheDocument();
    expect(await screen.findByText("Con dos sesiones de bolsa ya se puede comparar.")).toBeInTheDocument();
    // Ultima sesion: los numeros salen de la evolucion del perfil (CA-I01.1).
    const ultima = screen.getByRole("region", { name: "Última sesión" });
    expect(within(ultima).getByText("87")).toBeInTheDocument();
    expect(within(ultima).getByText(/se midió en sparring; en bolsa todavía no/)).toBeInTheDocument();
    expect(screen.getByTestId("horas-uso")).toHaveTextContent("1,2 de 4 h usadas en octubre.");
  });

  it("un pendiente le habla al boxeador", async () => {
    mundo.cuenta = cuentaIndividual();
    mundo.job = sesion(ID, "bolsa", "2026-10-26T10:00:00", "espera_siembra") as never;
    mundo.perfiles.propio = perfilDePrueba("propio", "Lucas", { evolucion: [] });
    enRuta("/");
    montar(<PaginaInicio />);
    expect(await screen.findByText("Falta un paso: marcate en la sesión de bolsa del 26/10.")).toBeInTheDocument();
  });

  it("en Entrenador el inicio sigue siendo el panel", async () => {
    enRuta("/");
    montar(<PaginaInicio />);
    expect(await screen.findByRole("heading", { name: "Panel" })).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// W30 y W31 · marcarse, sesion y perfil en segunda persona
// ---------------------------------------------------------------------------

function esperaSparring() {
  const job = { ...structuredClone(JOBS.espera_siembra), contexto: "sparring" } as Record<string, unknown> & { candidatos: Record<string, unknown>[] };
  const colores: Record<number, { hex: string; nombre: string }> = { 0: { hex: "#C02020", nombre: "rojos" }, 1: { hex: "#2040C0", nombre: "azules" } };
  job.candidatos = job.candidatos.slice(0, 3).map((c, i) => ({ ...c, t_desde: 0, t_hasta: 100 - i, color_guante: colores[i] ?? null }));
  return job;
}

describe("W30 · marcarse en sparring", () => {
  it("CA-I03.1 y CA-I03.2: '¿Cuál sos vos?', después el compañero, sin A, B ni rincón", async () => {
    mundo.cuenta = cuentaIndividual();
    const job = esperaSparring();
    mundo.job = job as never;
    enRuta(`/sesion/identificar/?id=${ID}`);
    const { container } = montar(<PantallaIdentificar />);
    expect(await screen.findByRole("heading", { name: "¿Cuál sos vos?", level: 1 })).toBeInTheDocument();
    const soyYo = screen.getAllByRole("radio", { name: "Soy yo" });
    await userEvent.click(soyYo[0]);
    expect(await screen.findByRole("heading", { name: "¿Y tu compañero?" })).toBeInTheDocument();
    await userEvent.click(screen.getAllByRole("radio", { name: "Es mi compañero" })[0]);
    for (const prohibido of ["Es A", "Es B", "Peleador B", "rincón", "Rincón"]) {
      expect(container.textContent).not.toContain(prohibido);
    }
    await userEvent.click(screen.getByRole("button", { name: "Completar análisis" }));
    expect(await screen.findByText("¿Vos sos el de guantes rojos y tu compañero el de azules?")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Sí, completar" }));
    const candidatos = job.candidatos as { track: number }[];
    await waitFor(() => expect(mundo.pedidos.find((x) => x.ruta.endsWith("/siembra"))?.cuerpo)
      .toEqual({ track_a: candidatos[0].track, track_b: candidatos[1].track }));
  });

  it("en bolsa le habla al boxeador: 'Soy yo'", async () => {
    mundo.cuenta = cuentaIndividual();
    mundo.producto = productoBase({ sesiones_individuales: true });
    const job = { ...structuredClone(JOBS.espera_siembra), contexto: "bolsa" } as never;
    mundo.job = job;
    enRuta(`/sesion/identificar/?id=${ID}`);
    montar(<PantallaIdentificar />);
    expect(await screen.findByText(textoIndicacion)).toBeInTheDocument();
    expect(screen.getAllByRole("radio", { name: "Soy yo" }).length).toBeGreaterThan(0);
    expect(screen.queryByRole("radio", { name: "Es él" })).toBeNull();
  });
});

const textoIndicacion = "Tocá una vez tu figura. El resto del video se sigue por el color de tus guantes.";

describe("W31 · sesion, Fight-Card y perfil", () => {
  it("CA-I05.1: en Individual 'Peleador B' no aparece en ninguna vista de la sesion", async () => {
    mundo.cuenta = cuentaIndividual();
    mundo.job = { ...structuredClone(JOBS.listo), boxeadores: { A: { id: "propio", nombre: "Lucas" }, B: null } } as never;
    mundo.boxeadores = [{ id: "propio", nombre: "Lucas", guardia: null, notas: null, creado: "2026-10-01T10:00:00" }];
    for (const vista of ["fightcard", "video", "perfiles"]) {
      enRuta(`/sesion/?id=${ID}&vista=${vista}`);
      const { container, unmount } = montar(<PantallaSesion />);
      await screen.findByRole("tab", { name: "Fight-Card" });
      await waitFor(() => expect(container.textContent).toContain("Compañero"));
      expect(container.textContent).not.toContain("Peleador B");
      expect(container.textContent).not.toContain("Peleador A");
      unmount();
    }
  });

  it("CA-I06.1: Mi perfil no tiene enlace al Rival Fantasma y se titula para el boxeador", async () => {
    mundo.cuenta = cuentaIndividual();
    mundo.perfiles.propio = perfilDePrueba("propio", "Lucas");
    mundo.boxeadores = [{ id: "propio", nombre: "Lucas", guardia: null, notas: null, creado: "2026-10-01T10:00:00" }];
    enRuta("/peleador/?id=propio");
    montar(<PantallaPeleador />);
    expect(await screen.findByRole("button", { name: "Editar mis datos" })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /Rival Fantasma/ })).toBeNull();
  });

  it("en Gimnasio la cabecera dice quien subio y un golpe corregido dice por quien", async () => {
    mundo.cuenta = cuentaGimnasio("entrenador");
    const fc = structuredClone(FIGHTCARDS["0.3"]) as Record<string, any>;
    const golpe = fc.peleadores.A.golpes[0];
    golpe.corregido = { tipo: "hook", tipo_original: golpe.tipo };
    fc.sesion = { contexto: "sparring", practica: null, boxeadores: { A: null, B: null },
      autoria: { subida_por: "Ana", correcciones: { [golpe.id]: "Juan" } } };
    mundo.fightcard = fc;
    enRuta(`/sesion/?id=${ID}&vista=video`);
    montar(<PantallaSesion />);
    expect(await screen.findByText(/Subió Ana/)).toBeInTheDocument();
    expect((await screen.findAllByText("corregido por Juan")).length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// W34, W33, W32 · Gimnasio
// ---------------------------------------------------------------------------

describe("W34 · panel del gimnasio", () => {
  function panelGimnasio(porEntrenador: unknown) {
    mundo.job = { ...structuredClone(JOBS.espera_siembra), subida_por: { id: "u-otro", nombre: "Juan" } } as never;
    const original = mundo.job;
    return { original, porEntrenador };
  }

  it("CA-G01.2: un pendiente de otro entrenador dice quien lo subio y se puede resolver", async () => {
    mundo.cuenta = cuentaGimnasio("entrenador");
    panelGimnasio(null);
    enRuta("/");
    montar(<PaginaInicio />);
    expect(await screen.findByRole("heading", { name: "Gimnasio de prueba" })).toBeInTheDocument();
    const pendientes = await screen.findByRole("region", { name: "Pendientes" });
    expect(within(pendientes).getByText("Subió Juan")).toBeInTheDocument();
    expect(within(pendientes).getByRole("link")).toHaveAttribute("href", `/sesion/identificar/?id=${ID}`);
    // "Mías" esconde lo que subio otro.
    await userEvent.click(screen.getByRole("radio", { name: "Mías" }));
    await waitFor(() => expect(screen.queryByRole("region", { name: "Pendientes" })).toBeNull());
  });

  it("CA-G01.1: el entrenador no ve el bloque por entrenador; el titular si", async () => {
    mundo.cuenta = cuentaGimnasio("entrenador");
    enRuta("/");
    const { unmount } = montar(<PaginaInicio />);
    await screen.findByRole("heading", { name: "Gimnasio de prueba" });
    expect(screen.queryByText("El equipo este mes")).toBeNull();
    expect(screen.getByRole("region", { name: "Horas del mes" })).toBeInTheDocument();
    unmount();
  });
});

describe("W33 · plantel y consentimiento", () => {
  beforeEach(() => {
    mundo.cuenta = cuentaGimnasio("titular");
    mundo.boxeadores = [
      { id: "b1", nombre: "Tomás Ríos", guardia: null, notas: null, creado: "2026-10-01T10:00:00", entrenador: { id: "u-tit", nombre: "Tita" }, consentimiento: { estado: "vigente", desde: "2026-10-26T10:00:00" } },
      { id: "b2", nombre: "Nico Ferreyra", guardia: null, notas: null, creado: "2026-10-01T10:00:00", entrenador: null, consentimiento: { estado: "falta", desde: null } },
    ];
  });

  it("CA-G07.1: el plantel muestra el consentimiento y filtra por 'Falta consentimiento'", async () => {
    enRuta("/peleadores/");
    montar(<PantallaPeleadores />);
    expect(await screen.findByRole("heading", { name: "Plantel" })).toBeInTheDocument();
    expect(await screen.findByText("Falta consentimiento", { selector: "span" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("radio", { name: "Falta consentimiento" }));
    await waitFor(() => expect(screen.queryByText("Tomás Ríos")).toBeNull());
    expect(screen.getByText("Nico Ferreyra")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("radio", { name: "A mi cargo" }));
    await waitFor(() => expect(screen.getByText("Tomás Ríos")).toBeInTheDocument());
    expect(screen.queryByText("Nico Ferreyra")).toBeNull();
  });

  it("CA-G08.1: registrar y revocar dejan la historia", async () => {
    mundo.perfiles.b2 = perfilDePrueba("b2", "Nico Ferreyra");
    enRuta("/peleador/?id=b2");
    montar(<PantallaPeleador />);
    await userEvent.click(await screen.findByRole("button", { name: "Registrar" }));
    expect(await screen.findByText("Texto de prueba del consentimiento.")).toBeInTheDocument();
    const confirmar = screen.getByRole("button", { name: "Confirmar" });
    expect(confirmar).toBeDisabled();
    await userEvent.click(screen.getByLabelText("El atleta leyó y aceptó este texto."));
    await userEvent.click(screen.getByLabelText("Tiene 18 años o más."));
    await userEvent.click(confirmar);
    expect(await screen.findByText(/Registrado por Tita el 26\/10/)).toBeInTheDocument();
    // La hoja se cierra con animacion: hasta entonces el resto de la pagina esta inerte.
    await userEvent.click(await screen.findByRole("button", { name: "Revocar" }, { timeout: 3000 }));
    await userEvent.click(await screen.findByRole("button", { name: "Revocar" }));
    expect(await screen.findByText(/Revocado por Tita el 30\/10/)).toBeInTheDocument();
    expect(screen.getByText(/Registrado por Tita el 26\/10/)).toBeInTheDocument();
  });

  it("sin texto cargado la hoja lo dice y no muestra uno inventado", async () => {
    mundo.textoConsentimiento = null;
    mundo.perfiles.b2 = perfilDePrueba("b2", "Nico Ferreyra");
    enRuta("/peleador/?id=b2");
    montar(<PantallaPeleador />);
    await userEvent.click(await screen.findByRole("button", { name: "Registrar" }));
    expect(await screen.findByText(/falta el texto del consentimiento/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Confirmar" })).toBeNull();
  });
});

describe("W32 · equipo", () => {
  it("CA-G09.0: en la version de prueba no hay emails, ni invitar, ni dar de baja", async () => {
    mundo.cuenta = cuentaGimnasio("titular");
    mundo.miembros = [
      { usuario_id: "u-tit", nombre: "Tita", rol: "titular", desde: "2026-10-26T10:00:00", baja_en: null },
      { usuario_id: "u-ana", nombre: "Ana", rol: "entrenador", desde: "2026-10-26T10:00:00", baja_en: null },
    ];
    enRuta("/equipo/");
    const { container } = montar(<PantallaEquipo />);
    expect(await screen.findByText("Ana")).toBeInTheDocument();
    expect(screen.getByText("En la versión de prueba el equipo viene cargado.")).toBeInTheDocument();
    expect(screen.getByText("Gimnasio · 50 h de video por mes")).toBeInTheDocument();
    expect(screen.getByText("El análisis en vivo no está incluido en esta versión.")).toBeInTheDocument();
    expect(container.textContent).not.toMatch(/@|Invitar|Dar de baja/);
    expect(screen.getByRole("button", { name: "Descargar el registro (CSV)" })).toBeInTheDocument();
  });

  it("un entrenador que abre Equipo ve que es del titular", async () => {
    mundo.cuenta = cuentaGimnasio("entrenador");
    enRuta("/equipo/");
    montar(<PantallaEquipo />);
    expect(await screen.findByText("Solo el titular del gimnasio administra el equipo.")).toBeInTheDocument();
    expect(mundo.pedidos.some((x) => x.ruta === "/cuenta/miembros")).toBe(false);
  });
});

describe("W35 · posicion en la cola", () => {
  it("dice cuantas van antes y cuanto falta, sin decir de quien son", () => {
    expect(textoDeCola({ delante: 3, espera_min: [12, 28] })).toBe("Hay 3 sesiones antes que la tuya. Unos 12 a 28 minutos para empezar.");
    expect(textoDeCola({ delante: 1, espera_min: [0, 0] })).toBe("Hay 1 sesión antes que la tuya.");
  });

  it("la espera de una sesion en cola la muestra", async () => {
    const s = { ...structuredClone(JOBS.en_cola), cola: { delante: 2, espera_min: [5, 9] } } as unknown as Sesion;
    montar(<EsperaSesion s={s} />);
    expect(await screen.findByText(/Hay 2 sesiones antes que la tuya\. Unos 5 a 9 minutos/)).toBeInTheDocument();
  });
});
