/*
 * Tipo de golpe v2 en pantalla (RT9 y RT10): la exactitud que se muestra es la que trae la
 * Fight-Card, y con la guardia supuesta de un lado aparece el aviso junto al tipo, en la
 * Fight-Card y en el analisis de video. Sin pantallas nuevas.
 */
import { describe, expect, it, beforeEach } from "vitest";
import { screen, within } from "@testing-library/react";
import { conSesion, enRuta, montar } from "./entorno";
import { ID, JOBS, mundo } from "./api-simulada";
import { videoSimulado } from "./video";
import { PantallaSesion } from "@/components/pantallas/sesion/PantallaSesion";
import { fcV2 } from "../fixtures/fc-v2";

const AVISO = /Jab y cross suponen guardia ortodoxa\. Cargá la guardia de .+ para corregirlo\./;

beforeEach(() => {
  conSesion();
  videoSimulado();
  mundo.job = structuredClone(JOBS.listo);
});

describe("tipo v2 · Fight-Card", () => {
  beforeEach(() => enRuta(`/sesion/?id=${ID}&vista=fightcard`));

  it("muestra la exactitud y el modelo que trae la Fight-Card", async () => {
    mundo.fightcard = fcV2();
    montar(<PantallaSesion />);
    const bloque = await screen.findByRole("region", { name: "Tipo de golpe" });
    expect(within(bloque).getByText(/clasificador_geo_v1/)).toBeInTheDocument();
    expect(within(bloque).getByText(/73%/)).toBeInTheDocument();
  });

  it("avisa solo del lado con la guardia supuesta", async () => {
    mundo.fightcard = fcV2({ A: false, B: true });
    montar(<PantallaSesion />);
    const bloque = await screen.findByRole("region", { name: "Tipo de golpe" });
    const avisos = within(bloque).getAllByText(AVISO);
    expect(avisos).toHaveLength(1);
    expect(avisos[0]).toHaveAttribute("data-aviso-guardia", "B");
  });

  it("sin guardia supuesta no hay aviso", async () => {
    mundo.fightcard = fcV2({ A: false, B: false });
    montar(<PantallaSesion />);
    await screen.findByRole("region", { name: "Tipo de golpe" });
    expect(screen.queryByText(AVISO)).toBeNull();
  });
});

describe("tipo v2 · analisis de video", () => {
  beforeEach(() => enRuta(`/sesion/?id=${ID}&vista=video`));

  it("el aviso aparece junto a la nota del tipo", async () => {
    mundo.fightcard = fcV2({ A: true, B: false });
    montar(<PantallaSesion />);
    const aviso = await screen.findByText(AVISO);
    expect(aviso).toHaveAttribute("data-aviso-guardia", "A");
  });
});
