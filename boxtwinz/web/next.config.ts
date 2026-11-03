/*
 * Configuracion de Next, igual criterio que la web entrenador (02 §9 de A).
 *
 * Build: exportacion estatica, que la API sirve desde el mismo origen (BOXTWINZ_WEB).
 * Desarrollo: rewrites de /api hacia la API local, asi el codigo usa rutas relativas.
 *
 * Las rutas con id de la spec (/sesiones/[id]) van como query (/sesion?id=): una exportacion
 * estatica no puede generar paginas para ids que todavia no existen (desvio D5).
 */
import type { NextConfig } from "next";

const API = process.env.BOXTWINZ_API_DEV ?? "http://localhost:8003";
const desarrollo = process.env.NODE_ENV === "development";

const config: NextConfig = desarrollo
  ? {
      trailingSlash: true,
      async rewrites() {
        return { beforeFiles: [{ source: "/api/:resto*", destination: `${API}/api/:resto*` }], afterFiles: [], fallback: [] };
      },
    }
  : { output: "export", trailingSlash: true, images: { unoptimized: true } };

export default config;
