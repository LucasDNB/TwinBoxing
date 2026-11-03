/*
 * Layout raiz de BoxTwinZ: las mismas fuentes y el mismo tema oscuro que la web entrenador,
 * mas el manifest y el registro del service worker de la PWA.
 */
import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import { Barlow_Condensed, IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import { RegistroSW } from "@/components/boxtwinz/RegistroSW";
import "./globals.css";

const display = Barlow_Condensed({ subsets: ["latin"], weight: ["500", "600", "700"], variable: "--fuente-display", display: "swap" });
const sans = IBM_Plex_Sans({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--fuente-sans", display: "swap" });
const mono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--fuente-mono", display: "swap" });

export const metadata: Metadata = {
  title: "BoxTwinZ",
  description: "Golpes detectados de izquierda y derecha en sombra, bolsa y sparring.",
  manifest: "/manifest.webmanifest",
  icons: { icon: "/icono.svg", apple: "/icono-192.png" },
  appleWebApp: { capable: true, title: "BoxTwinZ", statusBarStyle: "black-translucent" },
};

export const viewport: Viewport = { width: "device-width", initialScale: 1, viewportFit: "cover", themeColor: "#0E1113" };

export default function Raiz({ children }: { children: ReactNode }) {
  return (
    <html lang="es-AR" className={`${display.variable} ${sans.variable} ${mono.variable}`}>
      <body>
        {children}
        <RegistroSW />
      </body>
    </html>
  );
}
