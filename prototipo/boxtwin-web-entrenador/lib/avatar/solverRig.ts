/*
 * Del esqueleto del maniqui a los huesos del rig MPFB2 (27-10).
 *
 * POR QUE EXISTE
 *   El GLB llega en su pose de reposo (A o T). La pose en guardia ya la calcula lib/maniqui.ts
 *   como puntos 3D; aca se giran los huesos del rig para que cada segmento apunte hacia donde
 *   apunta el del maniqui. Solo direcciones: el largo de cada hueso es el de la malla, que no
 *   coincide con el del maniqui, y estirarlo deformaria la piel.
 *
 *   La pose no sale del video: es la misma guardia ilustrativa del maniqui (DC-6).
 *
 * QUE HACE
 *   HUESOS                  los nombres que el GLB tiene que traer (rig "Game engine" de MPFB2)
 *   orientar(raiz, H)       gira y escala el modelo a los ejes y la altura del maniqui
 *   posar(raiz, e)          apunta cada hueso a su segmento, de padres a hijos
 *   anclasDe(raiz, H)       cabeza y pecho del modelo ya posado, para las zonas de calor
 *
 *   Las coordenadas son las del padre de `raiz` (el grupo de la figura), como en el maniqui.
 */
import * as THREE from "three";
import type { Anclas } from "@/lib/avatar/exposicion";
import type { Esqueleto, Vec3 } from "@/lib/maniqui";

export const HUESOS = [
  "pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "head",
  "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
  "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
  "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r",
] as const;

type Punto = (e: Esqueleto) => Vec3;

/*
 * Cada hueso se apunta desde su origen hacia el de `fin` (en reposo), y la direccion buscada va
 * de `desde` a `hasta` en el maniqui. Ordenado de padres a hijos: girar el padre mueve al hijo.
 */
const APUNTES: { hueso: string; fin: string; desde: Punto; hasta: Punto }[] = [
  { hueso: "spine_01", fin: "neck_01", desde: (e) => e.pelvis, hasta: (e) => e.cuello },
  { hueso: "neck_01", fin: "head", desde: (e) => e.cuello, hasta: (e) => e.cabeza },
  ...(["l", "r"] as const).flatMap((s) => {
    const l = s === "l" ? "izq" : "der";
    return [
      { hueso: `upperarm_${s}`, fin: `lowerarm_${s}`, desde: (e: Esqueleto) => e.hombro[l], hasta: (e: Esqueleto) => e.codo[l] },
      { hueso: `lowerarm_${s}`, fin: `hand_${s}`, desde: (e: Esqueleto) => e.codo[l], hasta: (e: Esqueleto) => e.muneca[l] },
      { hueso: `thigh_${s}`, fin: `calf_${s}`, desde: (e: Esqueleto) => e.cadera[l], hasta: (e: Esqueleto) => e.rodilla[l] },
      { hueso: `calf_${s}`, fin: `foot_${s}`, desde: (e: Esqueleto) => e.rodilla[l], hasta: (e: Esqueleto) => e.tobillo[l] },
    ];
  }),
];

export function huesoDe(raiz: THREE.Object3D, nombre: string): THREE.Object3D | undefined {
  // Blender a veces exporta con prefijo de la armadura ("Armature_upperarm_l") o con puntos.
  let hallado: THREE.Object3D | undefined;
  raiz.traverse((o) => {
    if (hallado) return;
    const n = o.name.replace(/\./g, "_");
    if (n === nombre || n.endsWith(`_${nombre}`) || n.endsWith(`:${nombre}`)) hallado = o;
  });
  return hallado;
}

export function faltantes(raiz: THREE.Object3D): string[] {
  return HUESOS.filter((h) => !huesoDe(raiz, h));
}

const pos = (o: THREE.Object3D) => o.getWorldPosition(new THREE.Vector3());

/*
 * Gira el modelo para que su derecha quede en +z y su frente en +x, lo escala a la altura `H` y
 * apoya los pies en y = 0 con la pelvis sobre el origen. La derecha se lee de los hombros, no
 * de una convencion de exportacion: asi no importa hacia donde miraba en Blender.
 */
export function orientar(raiz: THREE.Object3D, H: number): void {
  raiz.position.set(0, 0, 0);
  raiz.quaternion.identity();
  raiz.scale.setScalar(1);
  raiz.updateMatrixWorld(true);
  const padre = raiz.parent;
  const aLocal = (v: THREE.Vector3) => (padre ? padre.worldToLocal(v) : v);
  const izq = huesoDe(raiz, "upperarm_l"), der = huesoDe(raiz, "upperarm_r");
  if (izq && der) {
    const derecha = aLocal(pos(der)).sub(aLocal(pos(izq)));
    derecha.y = 0;
    if (derecha.lengthSq() > 1e-10) {
      // Un giro sobre y que lleva la derecha a +z. setFromUnitVectors podria elegir otro eje
      // cuando la derecha viene justo al reves (-z).
      raiz.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.atan2(-derecha.x, derecha.z));
      raiz.updateMatrixWorld(true);
    }
  }
  const caja = new THREE.Box3().setFromObject(raiz, true);
  const alto = caja.max.y - caja.min.y;
  const k = alto > 1e-6 ? H / alto : 1;
  raiz.scale.setScalar(k);
  raiz.updateMatrixWorld(true);
  caja.setFromObject(raiz, true);
  const pelvis = huesoDe(raiz, "pelvis");
  const centro = pelvis ? aLocal(pos(pelvis)) : caja.getCenter(new THREE.Vector3());
  raiz.position.set(-centro.x, -caja.min.y, -centro.z);
  raiz.updateMatrixWorld(true);
}

export function posar(raiz: THREE.Object3D, e: Esqueleto): void {
  const padre = raiz.parent;
  const mundo = (p: Vec3) => {
    const v = new THREE.Vector3(...p);
    return padre ? padre.localToWorld(v) : v;
  };
  const qPadre = new THREE.Quaternion();
  const qHueso = new THREE.Quaternion();
  const giro = new THREE.Quaternion();
  // Girar `hueso` en el mundo: el giro se pasa al espacio de su padre.
  const girarEnMundo = (hueso: THREE.Object3D, g: THREE.Quaternion) => {
    hueso.getWorldQuaternion(qHueso);
    if (hueso.parent) hueso.parent.getWorldQuaternion(qPadre);
    else qPadre.identity();
    hueso.quaternion.copy(qPadre.invert().multiply(g).multiply(qHueso));
    hueso.updateMatrixWorld(true);
  };
  const cabeza = huesoDe(raiz, "head");
  // La cara en reposo mira a +x de la figura (orientar la dejo asi): se guarda como arranca.
  const qCabezaReposo = cabeza ? cabeza.getWorldQuaternion(new THREE.Quaternion()) : null;
  for (const a of APUNTES) {
    const hueso = huesoDe(raiz, a.hueso), fin = huesoDe(raiz, a.fin);
    if (!hueso || !fin) continue;
    hueso.updateMatrixWorld(true);
    const actual = pos(fin).sub(pos(hueso));
    const buscada = mundo(a.hasta(e)).sub(mundo(a.desde(e)));
    if (actual.lengthSq() < 1e-12 || buscada.lengthSq() < 1e-12) continue;
    giro.setFromUnitVectors(actual.normalize(), buscada.normalize());
    girarEnMundo(hueso, giro);
  }
  // Menton metido: apuntar cuello y columna deja la cabeza como venga, y en el rig de MPFB2
  // queda mirando para arriba. Se gira la cabeza para que la cara mire al rival, hacia abajo.
  if (cabeza && qCabezaReposo) {
    const qFigura = padre ? padre.getWorldQuaternion(new THREE.Quaternion()) : new THREE.Quaternion();
    const frenteReposo = new THREE.Vector3(1, 0, 0).applyQuaternion(qFigura);
    const ahora = cabeza.getWorldQuaternion(new THREE.Quaternion());
    const frente = frenteReposo.applyQuaternion(ahora.multiply(qCabezaReposo.clone().invert()));
    const t = (MENTON_GRADOS * Math.PI) / 180;
    const buscado = new THREE.Vector3(Math.cos(t), -Math.sin(t), 0).applyQuaternion(qFigura);
    girarEnMundo(cabeza, giro.setFromUnitVectors(frente.normalize(), buscado));
  }
}

/* Cuanto baja la cara en guardia: el menton contra el pecho, los ojos arriba mirando al rival. */
export const MENTON_GRADOS = 15;

/* Hacia donde mira la cara del modelo posado, en coordenadas de la figura. Para las pruebas. */
export function frenteDeCabeza(raiz: THREE.Object3D, qReposo: THREE.Quaternion): THREE.Vector3 | null {
  const cabeza = huesoDe(raiz, "head");
  if (!cabeza) return null;
  const padre = raiz.parent;
  const qFigura = padre ? padre.getWorldQuaternion(new THREE.Quaternion()) : new THREE.Quaternion();
  const ahora = cabeza.getWorldQuaternion(new THREE.Quaternion()).multiply(qReposo.clone().invert());
  return new THREE.Vector3(1, 0, 0).applyQuaternion(qFigura).applyQuaternion(ahora).applyQuaternion(qFigura.invert());
}

/* Cabeza: un poco arriba del hueso `head`, que en MPFB2 nace en la base del craneo. */
export function anclasDe(raiz: THREE.Object3D, H: number): Anclas | null {
  const head = huesoDe(raiz, "head"), pecho = huesoDe(raiz, "spine_03") ?? huesoDe(raiz, "spine_02");
  if (!head || !pecho) return null;
  const padre = raiz.parent;
  const local = (o: THREE.Object3D) => {
    const v = pos(o);
    return (padre ? padre.worldToLocal(v) : v).toArray() as Vec3;
  };
  const c = local(head);
  return { cabeza: [c[0], c[1] + 0.06 * H, c[2]], torso: local(pecho), altura: H };
}
