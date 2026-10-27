import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { intensidades, zonasCalor } from "@/lib/avatar/exposicion";
import { cargarZonas, inyectarCalor, uniformesCalor } from "@/lib/avatar/shaderCalor";
import { HUESOS, anclasDe, faltantes, huesoDe, orientar, posar } from "@/lib/avatar/solverRig";
import { esqueleto, medidasDe } from "@/lib/maniqui";
// @ts-expect-error modulo js sin tipos
import { HUESOS as HUESOS_VALIDADOR, validarGlb } from "../../scripts/validar-glb.mjs";

const fila = (fraccion_corregida: number | null, n = 10) => ({ n, fraccion: fraccion_corregida, fraccion_corregida });

describe("zonas de calor (lib/avatar/exposicion.ts)", () => {
  it("normaliza contra la zona mas dirigida", () => {
    const i = intensidades({
      cabeza_mano_izq: fila(0.4), cabeza_mano_der: fila(0.2), cuerpo_mano_izq: fila(0.1), cuerpo_mano_der: fila(null),
    })!;
    expect(i.cabeza_mano_izq).toBe(1);
    expect(i.cabeza_mano_der).toBeCloseTo(0.5);
    expect(i.cuerpo_mano_der).toBe(0);
  });

  it("sin zonas o todo en cero no hay calor", () => {
    expect(intensidades(null)).toBeNull();
    expect(intensidades({ cabeza_mano_izq: fila(0) })).toBeNull();
  });

  it("la mano izquierda del rival cae del lado derecho del peleador (z positivo)", () => {
    const z = zonasCalor({ cabeza: [0, 1.6, 0], torso: [0, 1.2, 0], altura: 1.75 }, {
      cabeza_mano_izq: 1, cabeza_mano_der: 0.5, cuerpo_mano_izq: 0.2, cuerpo_mano_der: 0,
    });
    const de = (n: string) => z.find((x) => x.zona === n)!;
    expect(de("cabeza_mano_izq").centro[2]).toBeGreaterThan(0);
    expect(de("cabeza_mano_der").centro[2]).toBeLessThan(0);
    expect(de("cuerpo_mano_izq").centro[1]).toBeLessThan(de("cabeza_mano_izq").centro[1]);
    expect(de("cabeza_mano_der").intensidad).toBe(0.5);
  });
});

describe("inyeccion del shader (lib/avatar/shaderCalor.ts)", () => {
  // El shader real de MeshStandardMaterial en Three 0.170: si un chunk cambia de nombre, falla aca.
  const fuente = () => ({
    vertexShader: THREE.ShaderLib.physical.vertexShader,
    fragmentShader: THREE.ShaderLib.physical.fragmentShader,
    uniforms: {} as Record<string, unknown>,
  });

  it("inserta el calor despues del skinning, del albedo y de la emision", () => {
    const s = fuente();
    const u = uniformesCalor();
    inyectarCalor(s, u);
    expect(s.vertexShader.indexOf("vCalorPos =")).toBeGreaterThan(s.vertexShader.indexOf("#include <skinning_vertex>"));
    expect(s.fragmentShader.indexOf("diffuseColor.rgb = mix")).toBeGreaterThan(s.fragmentShader.indexOf("#include <map_fragment>"));
    expect(s.fragmentShader.indexOf("totalEmissiveRadiance += calorC")).toBeGreaterThan(s.fragmentShader.indexOf("#include <emissivemap_fragment>"));
    // El PBR sigue entero: la inyeccion agrega, no reemplaza.
    for (const c of ["normal_fragment_maps", "roughnessmap_fragment", "lights_fragment_begin"]) {
      expect(s.fragmentShader).toContain(`#include <${c}>`);
    }
    expect(s.uniforms.uZonas).toBe(u.uZonas);
  });

  it("falla fuerte si el shader no trae el chunk", () => {
    expect(() => inyectarCalor({ vertexShader: "void main(){}", fragmentShader: "", uniforms: {} }, uniformesCalor())).toThrow(/common/);
  });

  it("cargarZonas copia centros, radios e intensidades y limpia lo que sobra", () => {
    const u = uniformesCalor();
    u.uIntensidad.value.fill(9);
    cargarZonas(u, [{ zona: "cabeza_mano_izq", centro: [1, 2, 3], radio: 0.2, intensidad: 0.7 }]);
    expect(Array.from(u.uZonas.value.slice(0, 3))).toEqual([1, 2, 3]);
    expect(u.uRadios.value[0]).toBeCloseTo(0.2);
    expect(Array.from(u.uIntensidad.value)).toEqual([expect.closeTo(0.7), 0, 0, 0]);
  });
});

/*
 * Un rig de juguete en T, mirando a +z como exporta Blender a glTF: la derecha del personaje
 * queda en -x. Tiene los huesos del contrato y una malla de una caja para que haya altura.
 */
function rigEnT(): THREE.Group {
  const raiz = new THREE.Group();
  const huesos: Record<string, THREE.Bone> = {};
  const hueso = (nombre: string, padre: THREE.Object3D, p: [number, number, number]) => {
    const b = new THREE.Bone();
    b.name = nombre;
    b.position.set(...p);
    padre.add(b);
    huesos[nombre] = b;
    return b;
  };
  hueso("pelvis", raiz, [0, 1, 0]);
  hueso("spine_01", huesos.pelvis, [0, 0.1, 0]);
  hueso("spine_02", huesos.spine_01, [0, 0.15, 0]);
  hueso("spine_03", huesos.spine_02, [0, 0.15, 0]);
  hueso("neck_01", huesos.spine_03, [0, 0.15, 0]);
  hueso("head", huesos.neck_01, [0, 0.1, 0]);
  for (const [s, x] of [["l", 1], ["r", -1]] as const) {
    hueso(`clavicle_${s}`, huesos.spine_03, [x * 0.05, 0.12, 0]);
    hueso(`upperarm_${s}`, huesos[`clavicle_${s}`], [x * 0.15, 0, 0]);
    hueso(`lowerarm_${s}`, huesos[`upperarm_${s}`], [x * 0.3, 0, 0]);
    hueso(`hand_${s}`, huesos[`lowerarm_${s}`], [x * 0.27, 0, 0]);
    hueso(`thigh_${s}`, huesos.pelvis, [x * 0.1, -0.05, 0]);
    hueso(`calf_${s}`, huesos[`thigh_${s}`], [0, -0.45, 0]);
    hueso(`foot_${s}`, huesos[`calf_${s}`], [0, -0.45, 0]);
  }
  const caja = new THREE.Mesh(new THREE.BoxGeometry(0.4, 1.8, 0.3));
  caja.position.y = 0.9;
  raiz.add(caja);
  return raiz;
}

describe("solver del rig (lib/avatar/solverRig.ts)", () => {
  it("la lista de huesos es la misma que revisa el validador", () => {
    expect([...HUESOS]).toEqual(HUESOS_VALIDADOR);
  });

  it("encuentra huesos con el prefijo de la armadura", () => {
    const g = new THREE.Group();
    const b = new THREE.Bone();
    b.name = "Armature_upperarm_l";
    g.add(b);
    expect(huesoDe(g, "upperarm_l")).toBe(b);
    expect(faltantes(g)).not.toContain("upperarm_l");
  });

  it("orienta, escala y apunta cada segmento hacia el del maniqui", () => {
    const figura = new THREE.Group();
    const raiz = rigEnT();
    figura.add(raiz);
    const m = medidasDe(1.8, 1.85);
    orientar(raiz, m.altura);

    const pos = (n: string) => huesoDe(raiz, n)!.getWorldPosition(new THREE.Vector3());
    // La derecha del modelo quedo en +z y los pies en el piso a la altura pedida.
    expect(pos("upperarm_r").z).toBeGreaterThan(pos("upperarm_l").z);
    const caja = new THREE.Box3().setFromObject(raiz, true);
    expect(caja.min.y).toBeCloseTo(0, 5);
    expect(caja.max.y - caja.min.y).toBeCloseTo(1.8, 5);

    const e = esqueleto(m, "ortodoxa", { adelantada: 0.3, atrasada: 0 });
    posar(raiz, e);
    const dir = (a: THREE.Vector3, b: THREE.Vector3) => b.clone().sub(a).normalize();
    const v = (p: [number, number, number]) => new THREE.Vector3(...p);
    for (const [hueso, fin, desde, hasta] of [
      ["upperarm_l", "lowerarm_l", e.hombro.izq, e.codo.izq],
      ["lowerarm_l", "hand_l", e.codo.izq, e.muneca.izq],
      ["lowerarm_r", "hand_r", e.codo.der, e.muneca.der],
      ["calf_r", "foot_r", e.rodilla.der, e.tobillo.der],
      ["spine_01", "neck_01", e.pelvis, e.cuello],
    ] as const) {
      expect(dir(pos(hueso), pos(fin)).dot(dir(v(desde), v(hasta)))).toBeCloseTo(1, 4);
    }

    const a = anclasDe(raiz, m.altura)!;
    expect(a.cabeza[1]).toBeGreaterThan(a.torso[1]);
  });
});

/* Un GLB minimo: cabecera, chunk JSON con relleno a 4 bytes, sin binario. */
function glb(json: object): Buffer {
  let texto = JSON.stringify(json);
  while (texto.length % 4) texto += " ";
  const cuerpo = Buffer.from(texto, "utf8");
  const b = Buffer.alloc(20 + cuerpo.length);
  b.writeUInt32LE(0x46546c67, 0);
  b.writeUInt32LE(2, 4);
  b.writeUInt32LE(b.length, 8);
  b.writeUInt32LE(cuerpo.length, 12);
  b.writeUInt32LE(0x4e4f534a, 16);
  cuerpo.copy(b, 20);
  return b;
}

function contratoCompleto(triangulos = 20000) {
  const nodos = HUESOS_VALIDADOR.map((n: string) => ({ name: `Armature_${n}` }));
  return {
    asset: { version: "2.0" },
    nodes: [...nodos, { name: "Boxer_Body_Mesh", mesh: 0, skin: 0 }, { name: "Boxer_Gloves_Mesh", mesh: 1, skin: 0 }],
    meshes: [
      { name: "Boxer_Body_Mesh", primitives: [{ attributes: { POSITION: 0 }, indices: 1, material: 0 }] },
      { name: "Boxer_Gloves_Mesh", primitives: [{ attributes: { POSITION: 0 }, indices: 2, material: 1 }] },
    ],
    accessors: [{ count: 9000 }, { count: (triangulos - 1000) * 3 }, { count: 1000 * 3 }],
    skins: [{ joints: nodos.map((_: unknown, i: number) => i) }],
    materials: [
      { pbrMetallicRoughness: { baseColorTexture: { index: 0 }, metallicRoughnessTexture: { index: 2 } }, normalTexture: { index: 1 } },
      {},
    ],
  };
}

describe("validador del GLB (scripts/validar-glb.mjs)", () => {
  it("un GLB con el contrato completo pasa sin avisos", () => {
    const r = validarGlb(glb(contratoCompleto()));
    expect(r.errores).toEqual([]);
    expect(r.avisos).toEqual([]);
    expect(r.triangulos).toBe(20000);
  });

  it("marca los huesos que faltan y la malla del cuerpo", () => {
    const j = contratoCompleto();
    j.nodes[0].name = "Bone";
    j.nodes[HUESOS_VALIDADOR.length].name = "Body";
    j.meshes[0].name = "Body";
    const r = validarGlb(glb(j));
    expect(r.errores.join(" ")).toMatch(/Boxer_Body_Mesh/);
    expect(r.errores.join(" ")).toMatch(/faltan huesos en el skin: pelvis/);
  });

  it("frena una malla sin decimar y avisa fuera del rango", () => {
    expect(validarGlb(glb(contratoCompleto(80000))).errores.join(" ")).toMatch(/decimar/);
    expect(validarGlb(glb(contratoCompleto(40000))).avisos.join(" ")).toMatch(/fuera de/);
  });

  it("rechaza lo que no es GLB", () => {
    expect(validarGlb(Buffer.from("hola mundo, no soy un glb")).errores[0]).toMatch(/magic/);
  });
});
