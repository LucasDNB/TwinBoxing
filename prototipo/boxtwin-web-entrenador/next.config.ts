/*
 * Configuracion de Next segun el modo (02 §9).
 *
 * En el build: output "export", sin rewrites. Next no admite rewrites en una exportacion
 * estatica y avisa si las encuentra; en produccion no hacen falta porque la API sirve la
 * web desde el mismo origen.
 *
 * En desarrollo: sin output, y rewrites hacia la API local para los prefijos reservados
 * de 02 §4. Asi el codigo usa siempre rutas relativas y no hay URL de API que configurar.
 */
import type { NextConfig } from "next";

// Los prefijos que son de la API y que ninguna pagina puede usar (02 §4).
export const PREFIJOS_API = [
  "/auth", "/jobs", "/videos", "/fightcards", "/boxeadores", "/salud", "/producto",
  "/panel", "/docs", "/openapi.json", "/sesiones",
];

const API = process.env.BOXTWIN_API_DEV ?? "http://localhost:8000";
const desarrollo = process.env.NODE_ENV === "development";

const config: NextConfig = desarrollo
  ? {
      trailingSlash: true,
      async rewrites() {
        // /sesiones/ es pagina y /sesiones/{id}/boxeadores es API: se reescribe solo la
        // forma de la API, con profundidad dos, para no tapar la pagina.
        const reglas = PREFIJOS_API.filter((p) => p !== "/sesiones").map((p) => ({
          source: `${p}/:resto*`,
          destination: `${API}${p}/:resto*`,
        }));
        reglas.push(
          ...PREFIJOS_API.filter((p) => p !== "/sesiones").map((p) => ({
            source: p,
            destination: `${API}${p}`,
          })),
        );
        reglas.push({
          source: "/sesiones/:id/boxeadores",
          destination: `${API}/sesiones/:id/boxeadores`,
        });
        return { beforeFiles: reglas, afterFiles: [], fallback: [] };
      },
    }
  : {
      output: "export",
      trailingSlash: true,
      images: { unoptimized: true },
    };

export default config;
