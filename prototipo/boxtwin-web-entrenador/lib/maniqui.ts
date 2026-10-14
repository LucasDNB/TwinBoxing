/*
 * La geometria del maniqui en guardia (cambios-14-10, W23 y W25).
 *
 * POR QUE EXISTE
 *   El avatar es ilustrativo (DC-6): no sale de ningun modelo de elevacion, sale de la altura y
 *   la envergadura cargadas a mano. Las cuentas que lo ponen a escala, y el alcance que el Rival
 *   Fantasma escribe al lado de cada control, viven aca y no en el componente de Three.js, para
 *   probarlas sin WebGL. El componente solo dibuja lo que esto devuelve.
 *
 * QUE HACE
 *   medidasDe(altura, envergadura)  largos de cada segmento, a escala o en proporciones estandar
 *   alcance(m, extension)           de hombro a punio, con el codo entre guardia y extendido
 *   esqueleto(m, guardia, ext)      puntos 3D de cada articulacion, mirando hacia +x
 *
 *   Ejes: x adelante (hacia el rival), y arriba, z a la derecha del peleador. Ortodoxo: pie y
 *   mano izquierdos adelante; zurdo, espejado.
 */

export type Vec3 = [number, number, number];

export interface MedidasManiqui {
  altura: number;
  envergadura: number;
  hombros: number;
  brazo: number;
  antebrazo: number;
  punio: number;
  aEscala: boolean;
}

// Proporciones de un adulto promedio cuando no hay medidas: la envergadura igual a la altura.
export const ALTURA_ESTANDAR = 1.75;
// Ancho biacromial sobre la altura (DC-6, spec 01 §4).
const HOMBROS = 0.23;
// Reparto del largo de cada brazo entre brazo, antebrazo y punio (de hombro a nudillos).
const REPARTO = { brazo: 0.44, antebrazo: 0.38, punio: 0.18 };
// El codo en guardia y extendido, en grados entre brazo y antebrazo.
export const CODO_GUARDIA = 40;
export const CODO_EXTENDIDO = 180;

export function medidasDe(altura_m: number | null | undefined, envergadura_m: number | null | undefined): MedidasManiqui {
  const aEscala = altura_m != null && envergadura_m != null;
  const altura = altura_m ?? ALTURA_ESTANDAR;
  const envergadura = envergadura_m ?? altura;
  const hombros = HOMBROS * altura;
  const largo = Math.max(envergadura - hombros, 0) / 2;
  return {
    altura, envergadura, hombros, aEscala,
    brazo: largo * REPARTO.brazo,
    antebrazo: largo * REPARTO.antebrazo,
    punio: largo * REPARTO.punio,
  };
}

function codo(extension: number): number {
  const e = Math.min(1, Math.max(0, extension));
  return CODO_GUARDIA + (CODO_EXTENDIDO - CODO_GUARDIA) * e;
}

/* De hombro a muneca, por el teorema del coseno con el angulo del codo. */
function hombroAMuneca(m: MedidasManiqui, extension: number): number {
  const t = (codo(extension) * Math.PI) / 180;
  return Math.sqrt(m.brazo ** 2 + m.antebrazo ** 2 - 2 * m.brazo * m.antebrazo * Math.cos(t));
}

/* El alcance de un brazo, de hombro a punio, en metros. Extendido es el largo entero del brazo. */
export function alcance(m: MedidasManiqui, extension: number): number {
  return hombroAMuneca(m, extension) + m.punio;
}

export interface Esqueleto {
  cabeza: Vec3;
  radioCabeza: number;
  cuello: Vec3;
  pelvis: Vec3;
  hombro: { izq: Vec3; der: Vec3 };
  codo: { izq: Vec3; der: Vec3 };
  muneca: { izq: Vec3; der: Vec3 };
  cadera: { izq: Vec3; der: Vec3 };
  rodilla: { izq: Vec3; der: Vec3 };
  tobillo: { izq: Vec3; der: Vec3 };
  radioGuante: number;
}

const suma = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const resta = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const por = (a: Vec3, k: number): Vec3 => [a[0] * k, a[1] * k, a[2] * k];
const largo = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
const unit = (a: Vec3): Vec3 => por(a, 1 / (largo(a) || 1));
const cruz = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/*
 * El codo por cinematica inversa de dos huesos: con el hombro, la muneca y los dos largos, el
 * codo queda en el circulo de soluciones del lado del polo (abajo y hacia afuera).
 */
function resolverCodo(hombro: Vec3, muneca: Vec3, a: number, b: number, polo: Vec3): Vec3 {
  const d = resta(muneca, hombro);
  const dist = Math.min(largo(d), a + b - 1e-6);
  const eje = unit(d);
  const x = (a * a - b * b + dist * dist) / (2 * dist);
  const h = Math.sqrt(Math.max(a * a - x * x, 0));
  const perp = unit(resta(polo, por(eje, polo[0] * eje[0] + polo[1] * eje[1] + polo[2] * eje[2])));
  return suma(suma(hombro, por(eje, x)), por(perp, h));
}

/*
 * Los puntos del maniqui en guardia. `extension` va de 0 (en guardia) a 1 (extendido) por
 * brazo: el adelantado y el atrasado. La muneca apunta al menton del rival: a la altura de los
 * hombros extendida y a la del propio menton en guardia.
 */
export function esqueleto(
  m: MedidasManiqui,
  guardia: "ortodoxa" | "zurda" | null | undefined,
  extension: { adelantada: number; atrasada: number },
): Esqueleto {
  const H = m.altura;
  const zurdo = guardia === "zurda";
  // Lado del cuerpo en z: izquierda -1, derecha +1. El pie adelantado es el izquierdo en ortodoxo.
  const ladoZ = { izq: -1, der: 1 } as const;
  const adelantado = zurdo ? "der" : "izq";
  const atrasado = zurdo ? "izq" : "der";
  // Perfilado: el hombro adelantado va hacia el rival y el atrasado hacia atras.
  const perfil = 0.06 * H;

  const pelvis: Vec3 = [0, 0.53 * H, 0];
  const cuello: Vec3 = [0.02 * H, 0.83 * H, 0];
  const radioCabeza = 0.065 * H;
  const cabeza: Vec3 = [0.03 * H, 0.83 * H + radioCabeza * 1.25, 0];

  const hombro = {} as Esqueleto["hombro"];
  const cadera = {} as Esqueleto["cadera"];
  const rodilla = {} as Esqueleto["rodilla"];
  const tobillo = {} as Esqueleto["tobillo"];
  const codoP = {} as Esqueleto["codo"];
  const muneca = {} as Esqueleto["muneca"];
  for (const lado of ["izq", "der"] as const) {
    const adelante = lado === adelantado ? 1 : -1;
    const z = ladoZ[lado];
    hombro[lado] = [cuello[0] + adelante * perfil, 0.81 * H, z * (m.hombros / 2) * 0.8];
    cadera[lado] = [adelante * perfil * 0.6, 0.52 * H, z * 0.09 * H];
    // Pies separados y rodillas apenas flexionadas: un paso de 0,35 alturas.
    tobillo[lado] = [adelante * 0.17 * H, 0.04 * H, z * 0.12 * H];
    rodilla[lado] = [adelante * 0.1 * H + 0.03 * H, 0.28 * H, z * 0.11 * H];

    const ext = lado === adelantado ? extension.adelantada : extension.atrasada;
    const d = hombroAMuneca(m, ext);
    // La muneca va hacia adelante y un poco hacia el centro; en guardia sube al menton.
    const altura = (1 - Math.min(1, Math.max(0, ext))) * 0.08 * H;
    const dir = unit([1, altura / Math.max(d, 1e-6), -z * 0.25]);
    muneca[lado] = suma(hombro[lado], por(dir, d));
    codoP[lado] = resolverCodo(hombro[lado], muneca[lado], m.brazo, m.antebrazo, [0, -1, z * 0.6]);
  }
  return {
    cabeza, radioCabeza, cuello, pelvis, hombro, codo: codoP, muneca, cadera, rodilla, tobillo,
    radioGuante: 0.04 * H,
  };
}

// Un uso en la prueba: confirmar que el codo cumple los dos largos.
export const _interno = { resolverCodo, hombroAMuneca, largo, resta, cruz };
