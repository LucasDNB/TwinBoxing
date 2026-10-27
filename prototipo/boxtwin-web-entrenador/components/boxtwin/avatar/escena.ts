/*
 * La escena de Three.js del avatar: luces, figuras, mapa de calor y giro (27-10).
 *
 * Vive aparte de Avatar3D.tsx para que todo Three.js (y el GLTFLoader) entre por un solo
 * import dinamico: nada de esto cuenta en el JS inicial de la ruta (03 §9, CA-23.1).
 *
 * Cada figura es un grupo con sus coordenadas de maniqui (x adelante, y arriba, z a la
 * derecha). Si public/models/boxer_mpfb2.glb esta y trae los huesos, se clona, se orienta y se
 * posa con lib/avatar/solverRig.ts; si no, se arma el maniqui de primitivas de siempre. Las dos
 * llevan el mismo mapa de calor inyectado en el material de la piel.
 */
import * as THREE from "three";
import type { GLTF } from "three/examples/jsm/loaders/GLTFLoader.js";
import { zonasCalor, type Anclas, type Intensidades } from "@/lib/avatar/exposicion";
import { aplicarCalor, cargarZonas, uniformesCalor, type UniformesCalor } from "@/lib/avatar/shaderCalor";
import { anclasDe, faltantes, orientar, posar } from "@/lib/avatar/solverRig";
import { esqueleto, type MedidasManiqui, type Vec3 } from "@/lib/maniqui";

export interface Figura {
  medidas: MedidasManiqui;
  guardia: "ortodoxa" | "zurda" | null | undefined;
  lado: "A" | "B";
  extension: { adelantada: number; atrasada: number };
  /* Posicion en x (metros) y hacia donde mira: 1 hacia +x, -1 hacia -x. */
  x: number;
  mira: 1 | -1;
  opacidad?: number;
  /* Golpes dirigidos por zona, normalizados (lib/avatar/exposicion.ts). Sin dato, sin calor. */
  calor?: Intensidades | null;
}

export interface AjustesCalor {
  activo: boolean;
  opacidad: number;
  umbral: number;
}

export interface Escena {
  ajustar: (a: AjustesCalor) => void;
  limpiar: () => void;
  /* "glb" si se dibujo el modelo MPFB2, "primitivas" si cayo al maniqui. */
  modelo: "glb" | "primitivas";
}

export const URL_MODELO = "/models/boxer_mpfb2.glb";
const CUERPO = "Boxer_Body_Mesh";
const GUANTES = "Boxer_Gloves_Mesh";
// Mas oscura que antes (0x8a8f96): sobre gris claro la rampa del calor se lavaba.
const COLOR_PIEL = 0x6b7078;

function colorDe(lado: "A" | "B"): number {
  const v = getComputedStyle(document.documentElement).getPropertyValue(lado === "A" ? "--color-a-trazo" : "--color-b-trazo").trim();
  return /^#[0-9a-f]{6}$/i.test(v) ? parseInt(v.slice(1), 16) : lado === "A" ? 0xe5563f : 0x4a90e2;
}

// Se baja una vez por pagina: el Rival Fantasma rearma la escena con cada control.
let modelo: Promise<GLTF | null> | null = null;
function cargarModelo(): Promise<GLTF | null> {
  modelo ??= import("three/examples/jsm/loaders/GLTFLoader.js")
    .then(({ GLTFLoader }) => new GLTFLoader().loadAsync(URL_MODELO))
    .then((gltf) => {
      const faltan = faltantes(gltf.scene);
      if (faltan.length) {
        console.warn(`boxer_mpfb2.glb sin los huesos ${faltan.join(", ")}: se dibuja el maniqui`);
        return null;
      }
      return gltf;
    })
    .catch(() => null);
  return modelo;
}

const v3 = (p: Vec3) => new THREE.Vector3(p[0], p[1], p[2]);

/* El maniqui de primitivas (W23). Devuelve las anclas de las zonas. */
function primitivas(grupo: THREE.Group, f: Figura, u: UniformesCalor, desechar: { dispose: () => void }[]): Anclas {
  const e = esqueleto(f.medidas, f.guardia, f.extension);
  const H = f.medidas.altura;
  const transparente = (f.opacidad ?? 1) < 1;
  const piel = new THREE.MeshStandardMaterial({ color: COLOR_PIEL, roughness: 0.55, transparent: transparente, opacity: f.opacidad ?? 1 });
  aplicarCalor(piel, u);
  const guante = new THREE.MeshStandardMaterial({ color: colorDe(f.lado), roughness: 0.35, transparent: transparente, opacity: f.opacidad ?? 1 });
  desechar.push(piel, guante);
  const segmento = (a: Vec3, b: Vec3, r: number, mat: THREE.Material) => {
    const pa = v3(a), pb = v3(b);
    const g = new THREE.CapsuleGeometry(r, Math.max(pa.distanceTo(pb), 1e-3), 6, 16);
    desechar.push(g);
    const m = new THREE.Mesh(g, mat);
    m.position.copy(pa.clone().add(pb).multiplyScalar(0.5));
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), pb.clone().sub(pa).normalize());
    grupo.add(m);
  };
  const esfera = (p: Vec3, r: number, mat: THREE.Material) => {
    const g = new THREE.SphereGeometry(r, 32, 24);
    desechar.push(g);
    const m = new THREE.Mesh(g, mat);
    m.position.copy(v3(p));
    grupo.add(m);
  };
  const r = 0.028 * H;
  // Tronco: de la pelvis al cuello, mas ancho que un miembro.
  segmento(e.pelvis, e.cuello, 0.09 * H, piel);
  segmento(e.hombro.izq, e.hombro.der, 0.035 * H, piel);
  segmento(e.cadera.izq, e.cadera.der, 0.045 * H, piel);
  esfera(e.cabeza, e.radioCabeza, piel);
  for (const l of ["izq", "der"] as const) {
    segmento(e.hombro[l], e.codo[l], r, piel);
    segmento(e.codo[l], e.muneca[l], r * 0.9, piel);
    esfera(e.muneca[l], e.radioGuante * 1.15, guante);
    segmento(e.cadera[l], e.rodilla[l], r * 1.35, piel);
    segmento(e.rodilla[l], e.tobillo[l], r * 1.1, piel);
  }
  const t = 0.55;
  const torso: Vec3 = [
    e.pelvis[0] + (e.cuello[0] - e.pelvis[0]) * t,
    e.pelvis[1] + (e.cuello[1] - e.pelvis[1]) * t,
    e.pelvis[2] + (e.cuello[2] - e.pelvis[2]) * t,
  ];
  return { cabeza: e.cabeza, torso, altura: H };
}

/* El modelo MPFB2, clonado para esta figura, con materiales propios (cada figura su calor). */
async function mpfb2(grupo: THREE.Group, gltf: GLTF, f: Figura, u: UniformesCalor, desechar: { dispose: () => void }[]): Promise<Anclas> {
  const { clone } = await import("three/examples/jsm/utils/SkeletonUtils.js");
  const H = f.medidas.altura;
  const raiz = clone(gltf.scene);
  grupo.add(raiz);
  orientar(raiz, H);
  posar(raiz, esqueleto(f.medidas, f.guardia, f.extension));
  const transparente = (f.opacidad ?? 1) < 1;
  // Si el export nombra el cuerpo, el calor va solo ahi (no en el short ni en los ojos).
  let conNombre = false;
  raiz.traverse((o) => { conNombre ||= o.name.includes(CUERPO) || !!o.parent?.name.includes(CUERPO); });
  raiz.traverse((o) => {
    if (!(o instanceof THREE.Mesh)) return;
    o.frustumCulled = false;
    const nombre = `${o.name} ${o.parent?.name ?? ""}`;
    const guantes = nombre.includes(GUANTES);
    const piel = conNombre ? nombre.includes(CUERPO) : o instanceof THREE.SkinnedMesh;
    const mats = (Array.isArray(o.material) ? o.material : [o.material]).map((m: THREE.Material) => {
      const c = m.clone();
      desechar.push(c);
      c.transparent = transparente;
      c.opacity = f.opacidad ?? 1;
      // El cuero toma el color del lado; el abrojo ("Detalle") queda como vino.
      if (guantes && c instanceof THREE.MeshStandardMaterial) {
        if (!c.name.includes("Detalle")) c.color.setHex(colorDe(f.lado));
      } else if (piel && c instanceof THREE.MeshStandardMaterial) aplicarCalor(c, u);
      return c;
    });
    o.material = Array.isArray(o.material) ? mats : mats[0];
  });
  return anclasDe(raiz, H) ?? primitivasAnclas(f);
}

function primitivasAnclas(f: Figura): Anclas {
  const e = esqueleto(f.medidas, f.guardia, f.extension);
  return { cabeza: e.cabeza, torso: [e.pelvis[0], (e.pelvis[1] + e.cuello[1]) / 2, 0], altura: f.medidas.altura };
}

export async function armarEscena(el: HTMLElement, figuras: Figura[], alto: number, ajustes: AjustesCalor): Promise<Escena | null> {
  let renderer: THREE.WebGLRenderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch {
    return null;
  }
  const gltf = await cargarModelo();

  const ancho = el.clientWidth || 320;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  // El tamano en CSS lo pone el contenedor (100%) y no Three.js: si el lienzo fija su ancho
  // en pixeles, empuja a su contenedor y el ResizeObserver lo agranda en bucle (en el
  // celular se veia scroll horizontal).
  renderer.setSize(ancho, alto, false);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  Object.assign(renderer.domElement.style, { display: "block", width: "100%", height: `${alto}px` });
  renderer.domElement.setAttribute("aria-hidden", "true");
  renderer.domElement.style.touchAction = "pan-y";
  el.appendChild(renderer.domElement);

  const escena = new THREE.Scene();
  escena.add(new THREE.HemisphereLight(0xdfe8ff, 0x2a2420, 1.6));
  const clave = new THREE.DirectionalLight(0xfff1e0, 2.4);
  clave.position.set(2.5, 4, 3);
  // Contraluz frio: recorta la silueta contra el fondo oscuro.
  const contra = new THREE.DirectionalLight(0x8fb8ff, 2.2);
  contra.position.set(-3, 2.5, -3);
  const relleno = new THREE.DirectionalLight(0xffffff, 0.6);
  relleno.position.set(-2, 1, 3);
  escena.add(clave, contra, relleno);

  const raiz = new THREE.Group();
  const desechar: { dispose: () => void }[] = [];
  const grupos: { grupo: THREE.Group; u: UniformesCalor }[] = [];

  let altoMax = 0;
  let xMin = Infinity, xMax = -Infinity;
  for (const f of figuras) {
    const H = f.medidas.altura;
    altoMax = Math.max(altoMax, H);
    const grupo = new THREE.Group();
    const u = uniformesCalor();
    // Se arma con el grupo en el origen: el solver trabaja en sus coordenadas.
    const anclas = gltf ? await mpfb2(grupo, gltf, f, u, desechar) : primitivas(grupo, f, u, desechar);
    cargarZonas(u, f.calor ? zonasCalor(anclas, f.calor) : null);
    // Girar y no espejar: espejar cambiaria la guardia (un ortodoxo se veria zurdo).
    grupo.rotation.y = f.mira === 1 ? 0 : Math.PI;
    grupo.position.x = f.x;
    raiz.add(grupo);
    grupos.push({ grupo, u });
    xMin = Math.min(xMin, f.x - 0.4 * H);
    xMax = Math.max(xMax, f.x + 0.4 * H);
  }
  // El centro de la escena en el origen, para rotar alrededor de los dos.
  raiz.position.x = -(xMin + xMax) / 2;
  const ancho3d = Math.max(xMax - xMin, altoMax * 0.8);

  const camara = new THREE.PerspectiveCamera(35, ancho / alto, 0.1, 50);
  // Margen de sobra: al girar, el que queda mas cerca de la camara no se corta por abajo.
  const dist = Math.max(altoMax * 2.1, (ancho3d / 2) / Math.tan((35 * Math.PI) / 360) / (ancho / alto) * 1.35);
  camara.position.set(0, altoMax * 0.6, dist);
  camara.lookAt(0, altoMax * 0.47, 0);

  const pivote = new THREE.Group();
  pivote.add(raiz);
  escena.add(pivote);
  // Uno solo, de tres cuartos de frente: se ve la cara y el pecho, que es donde va el calor.
  pivote.rotation.y = figuras.length === 1 ? -0.95 : -0.35;

  const inversa = new THREE.Matrix4();
  const dibujar = () => {
    escena.updateMatrixWorld(true);
    for (const { grupo, u } of grupos) u.uFiguraInv.value.set(inversa.copy(grupo.matrixWorld).invert().elements);
    renderer.render(escena, camara);
  };
  const ajustar = (a: AjustesCalor) => {
    for (const { u } of grupos) {
      u.uOpacidad.value = a.activo ? a.opacidad : 0;
      u.uUmbral.value = a.umbral;
    }
    dibujar();
  };
  ajustar(ajustes);

  // Arrastrar gira alrededor del eje vertical. Sin OrbitControls: es una sola cuenta.
  let x0: number | null = null;
  const abajo = (ev: PointerEvent) => { x0 = ev.clientX; renderer.domElement.setPointerCapture(ev.pointerId); };
  const mover = (ev: PointerEvent) => {
    if (x0 == null) return;
    pivote.rotation.y += (ev.clientX - x0) * 0.01;
    x0 = ev.clientX;
    dibujar();
  };
  const arriba = () => { x0 = null; };
  renderer.domElement.addEventListener("pointerdown", abajo);
  renderer.domElement.addEventListener("pointermove", mover);
  renderer.domElement.addEventListener("pointerup", arriba);
  renderer.domElement.addEventListener("pointercancel", arriba);

  const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => {
    const w = el.clientWidth || ancho;
    renderer.setSize(w, alto, false);
    camara.aspect = w / alto;
    camara.updateProjectionMatrix();
    dibujar();
  }) : null;
  ro?.observe(el);

  return {
    ajustar,
    modelo: gltf ? "glb" : "primitivas",
    limpiar: () => {
      ro?.disconnect();
      // Las texturas son del GLB cacheado y las comparten los clones: no se liberan.
      for (const d of desechar) d.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
