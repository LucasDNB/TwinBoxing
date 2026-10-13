import "./navegacion-simulada";
import { afterAll, afterEach, beforeAll, beforeEach } from "vitest";
import { render } from "@testing-library/react";
import type { ReactElement } from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { crearClienteDeConsultas } from "@/components/boxtwin/Proveedores";
import { Toaster } from "@/components/ui/sonner";
import { reiniciarMundo, servidor } from "./api-simulada";
import { alVencerseLaSesion } from "@/lib/api/cliente";
import { navegacion } from "./navegacion-simulada";

// jsdom necesita una base para que las rutas relativas de fetch resuelvan.
beforeAll(() => servidor.listen({ onUnhandledRequest: "error" }));
afterAll(() => servidor.close());
beforeEach(() => {
  reiniciarMundo();
  navegacion.historial = [];
  localStorage.clear();
  alVencerseLaSesion(() => {
    const aca = window.location.pathname + window.location.search;
    window.history.pushState({}, "", `/ingresar/?siguiente=${encodeURIComponent(aca)}`);
    navegacion.historial.push(`/ingresar/?siguiente=${encodeURIComponent(aca)}`);
    window.dispatchEvent(new PopStateEvent("popstate"));
  });
});
afterEach(() => servidor.resetHandlers());

export function enRuta(url: string) {
  window.history.replaceState({}, "", url);
}

export function conSesion(token = "token-bueno") {
  localStorage.setItem("boxtwin.token", token);
  localStorage.setItem("boxtwin.usuario", "lucas@usal.edu.ar");
}

export function montar(ui: ReactElement) {
  const qc = crearClienteDeConsultas();
  return { qc, ...render(<QueryClientProvider client={qc}>{ui}<Toaster /></QueryClientProvider>) };
}
