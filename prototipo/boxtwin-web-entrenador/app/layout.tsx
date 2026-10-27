/*
 * Layout raiz: fuentes, tema oscuro unico, MotionConfig y el cliente de consultas.
 *
 * Las fuentes se sirven desde el build (next/font), asi que en produccion no se le pide
 * nada a Google y la CSP puede quedar en font-src 'self' (02 §8).
 */
import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import { Barlow_Condensed, IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import { Proveedores } from "@/components/boxtwin/Proveedores";
import "./globals.css";

const display = Barlow_Condensed({
  subsets: ["latin"], weight: ["500", "600", "700"], variable: "--fuente-display",
  display: "swap",
});
const sans = IBM_Plex_Sans({
  subsets: ["latin"], weight: ["400", "500", "600"], variable: "--fuente-sans",
  display: "swap",
});
const mono = IBM_Plex_Mono({
  subsets: ["latin"], weight: ["400", "500"], variable: "--fuente-mono", display: "swap",
});

export const metadata: Metadata = {
  title: "BoxTwin",
  description: "Análisis de sparring filmado: golpes detectados, con su margen a la vista.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#0E1113",
};

export default function Raiz({ children }: { children: ReactNode }) {
  return (
    <html lang="es-AR" className={`${display.variable} ${sans.variable} ${mono.variable}`}>
      <body>
        <Proveedores>{children}</Proveedores>
      </body>
    </html>
  );
}
