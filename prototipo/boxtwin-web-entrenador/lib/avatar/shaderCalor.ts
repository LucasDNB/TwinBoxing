/*
 * El mapa de calor sobre la piel del avatar, inyectado en el shader de MeshStandardMaterial.
 *
 * POR QUE EXISTE
 *   Un material propio perderia el PBR del GLB (normal, rugosidad, albedo). Inyectando en
 *   onBeforeCompile, Three.js arma su shader de siempre y aca solo se agregan unas lineas: el
 *   calor se mezcla en el albedo despues de leer el mapa y suma un poco de emision, y la luz,
 *   el normal map y la rugosidad siguen haciendo su trabajo encima.
 *
 * QUE HACE
 *   uniformesCalor()          los uniforms que comparte una figura (zonas, opacidad, umbral)
 *   cargarZonas(u, zonas)     copia las zonas de lib/avatar/exposicion.ts a los uniforms
 *   inyectarCalor(shader)     reescribe el vertex y el fragment; puro, se prueba sin WebGL
 *   aplicarCalor(mat, u)      engancha la inyeccion a un material
 *
 *   La posicion se lleva al espacio de la figura (uFiguraInv) despues del skinning, asi el
 *   calor sigue al cuerpo cuando el rig se mueve y no se corre cuando se gira la escena.
 */
import type { Material } from "three";
import type { ZonaCalor } from "@/lib/avatar/exposicion";

export const MAX_ZONAS = 4;

export interface UniformesCalor {
  uFiguraInv: { value: Float32Array };
  uZonas: { value: Float32Array };
  uRadios: { value: Float32Array };
  uIntensidad: { value: Float32Array };
  uOpacidad: { value: number };
  uUmbral: { value: number };
  uBrillo: { value: number };
}

export function uniformesCalor(): UniformesCalor {
  const identidad = new Float32Array(16);
  identidad[0] = identidad[5] = identidad[10] = identidad[15] = 1;
  return {
    uFiguraInv: { value: identidad },
    uZonas: { value: new Float32Array(MAX_ZONAS * 3) },
    uRadios: { value: new Float32Array(MAX_ZONAS).fill(1) },
    uIntensidad: { value: new Float32Array(MAX_ZONAS) },
    uOpacidad: { value: 0.85 },
    uUmbral: { value: 0.1 },
    uBrillo: { value: 1.2 },
  };
}

export function cargarZonas(u: UniformesCalor, zonas: ZonaCalor[] | null): void {
  u.uIntensidad.value.fill(0);
  if (!zonas) return;
  zonas.slice(0, MAX_ZONAS).forEach((z, i) => {
    u.uZonas.value.set(z.centro, i * 3);
    u.uRadios.value[i] = z.radio;
    u.uIntensidad.value[i] = z.intensidad;
  });
}

const VERTEX_DECL = /* glsl */ `
uniform mat4 uFiguraInv;
varying vec3 vCalorPos;`;

// Despues de skinning_vertex, `transformed` ya es la posicion deformada en espacio del objeto.
const VERTEX_POS = /* glsl */ `
vCalorPos = (uFiguraInv * modelMatrix * vec4(transformed, 1.0)).xyz;`;

const FRAGMENT_DECL = /* glsl */ `
#define CALOR_ZONAS ${MAX_ZONAS}
uniform vec3 uZonas[CALOR_ZONAS];
uniform float uRadios[CALOR_ZONAS];
uniform float uIntensidad[CALOR_ZONAS];
uniform float uOpacidad;
uniform float uUmbral;
uniform float uBrillo;
varying vec3 vCalorPos;

// Rampa termica: azul, cian, verde, amarillo, rojo.
vec3 calorRampa(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c = mix(vec3(0.05, 0.15, 0.9), vec3(0.0, 0.85, 0.95), smoothstep(0.0, 0.25, t));
  c = mix(c, vec3(0.2, 0.95, 0.3), smoothstep(0.25, 0.5, t));
  c = mix(c, vec3(1.0, 0.9, 0.1), smoothstep(0.5, 0.75, t));
  return mix(c, vec3(1.0, 0.12, 0.05), smoothstep(0.75, 1.0, t));
}

float calorEn(vec3 p) {
  float h = 0.0;
  for (int i = 0; i < CALOR_ZONAS; i++) {
    float d = distance(p, uZonas[i]) / max(uRadios[i], 1e-4);
    // Meseta hasta un tercio del radio y caida suave al borde; las zonas se pisan con max.
    h = max(h, uIntensidad[i] * (1.0 - smoothstep(0.35, 1.0, d)));
  }
  return h;
}`;

// Despues de map_fragment: el albedo ya trae su textura, el calor se mezcla encima.
const FRAGMENT_MEZCLA = /* glsl */ `
float calorH = calorEn(vCalorPos);
// El calor bajo se desvanece en vez de pintar de cian todo el torso: la opacidad sube con h.
float calorM = uOpacidad * smoothstep(uUmbral, uUmbral + 0.45, calorH);
vec3 calorC = calorRampa(calorH);
diffuseColor.rgb = mix(diffuseColor.rgb, calorC, calorM);`;

const FRAGMENT_EMISION = /* glsl */ `
totalEmissiveRadiance += calorC * calorM * calorH * uBrillo;`;

interface Shader {
  vertexShader: string;
  fragmentShader: string;
  uniforms: Record<string, unknown>;
}

function insertarDespues(fuente: string, chunk: string, codigo: string): string {
  const marca = `#include <${chunk}>`;
  if (!fuente.includes(marca)) throw new Error(`shader sin ${marca}`);
  return fuente.replace(marca, `${marca}\n${codigo}`);
}

export function inyectarCalor(shader: Shader, u: UniformesCalor): void {
  Object.assign(shader.uniforms, u);
  let vs = insertarDespues(shader.vertexShader, "common", VERTEX_DECL);
  vs = insertarDespues(vs, "skinning_vertex", VERTEX_POS);
  let fs = insertarDespues(shader.fragmentShader, "common", FRAGMENT_DECL);
  fs = insertarDespues(fs, "map_fragment", FRAGMENT_MEZCLA);
  fs = insertarDespues(fs, "emissivemap_fragment", FRAGMENT_EMISION);
  shader.vertexShader = vs;
  shader.fragmentShader = fs;
}

export function aplicarCalor(mat: Material, u: UniformesCalor): void {
  mat.onBeforeCompile = (shader) => inyectarCalor(shader, u);
  // El codigo es el mismo para todas las figuras: comparten el programa y difieren en uniforms.
  mat.customProgramCacheKey = () => "boxtwin-calor-v1";
  mat.needsUpdate = true;
}
