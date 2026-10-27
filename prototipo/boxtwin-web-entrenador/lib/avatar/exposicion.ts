/*
 * Las zonas de calor del avatar (27-10).
 *
 * POR QUE EXISTE
 *   El mapa de exposicion del perfil (W22) ya trae los golpes que le dirigen al peleador,
 *   partidos en cabeza o cuerpo y por la mano del rival que los tiro. El avatar los pinta sobre
 *   la piel. Es el mismo dato que la silueta 2D, con la misma estimacion detras: el detector no
 *   establece contacto, asi que lo que se pinta es hacia donde fueron, no donde pegaron.
 *
 * QUE HACE
 *   intensidades(zonas)       fraccion de cada zona normalizada a [0, 1] contra la mayor
 *   zonasCalor(anclas, int)   centro, radio e intensidad de cada zona en coordenadas de la figura
 *
 *   Ejes como en lib/maniqui.ts: x adelante, y arriba, z a la derecha del peleador. La mano
 *   izquierda del rival queda enfrente del lado derecho del peleador (z positivo), igual que
 *   en SiluetaZonas.
 */
import { ZONAS, type Zona } from "@/lib/api/esquemas";
import type { Vec3 } from "@/lib/maniqui";

export type Intensidades = Record<Zona, number>;

export interface ZonaCalor {
  zona: Zona;
  centro: Vec3;
  radio: number;
  intensidad: number;
}

/* Lo que hace falta del cuerpo para ubicar las zonas: el centro de la cabeza y el del pecho. */
export interface Anclas {
  cabeza: Vec3;
  torso: Vec3;
  altura: number;
}

type FilaZona = { n: number; fraccion: number | null; fraccion_corregida: number | null };

/* Normaliza contra la zona mas cargada: la mas dirigida queda en 1. Sin dato, todo en 0. */
export function intensidades(zonas: Partial<Record<Zona, FilaZona>> | null | undefined): Intensidades | null {
  if (!zonas) return null;
  const fr = ZONAS.map((z) => zonas[z]?.fraccion_corregida ?? zonas[z]?.fraccion ?? 0);
  const max = Math.max(...fr);
  if (!(max > 0)) return null;
  return Object.fromEntries(ZONAS.map((z, i) => [z, fr[i] / max])) as Intensidades;
}

// Desplazamientos en alturas: hacia adelante (la cara y el pecho miran al rival) y al costado.
const CABEZA = { adelante: 0.035, costado: 0.035, radio: 0.075 };
const CUERPO = { adelante: 0.06, costado: 0.07, radio: 0.12 };

export function zonasCalor(a: Anclas, int: Intensidades): ZonaCalor[] {
  const H = a.altura;
  return ZONAS.map((zona) => {
    const cabeza = zona.startsWith("cabeza");
    const base = cabeza ? a.cabeza : a.torso;
    const k = cabeza ? CABEZA : CUERPO;
    // Mano izquierda del rival: lado derecho del peleador.
    const lado = zona.endsWith("mano_izq") ? 1 : -1;
    return {
      zona,
      centro: [base[0] + k.adelante * H, base[1], base[2] + lado * k.costado * H] as Vec3,
      radio: k.radio * H,
      intensidad: int[zona],
    };
  });
}
