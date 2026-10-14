"use client";

/*
 * Avatar 3D ilustrativo (cambios-14-10, W23 y W25; DC-6).
 *
 * Un maniqui de primitivas en guardia, a escala de las medidas cargadas. No sale de ningun
 * modelo de elevacion: la profundidad no se mide, y el rotulo lo dice. La geometria vive en
 * lib/maniqui.ts; aca solo se dibuja.
 *
 * Three.js se importa adentro del efecto, asi queda en un chunk aparte que el navegador baja
 * solo cuando el bloque se monta y no cuenta en el JS inicial de la ruta (03 §9, CA-23.1).
 * Se dibuja a pedido (al armar y al arrastrar), sin un bucle de animacion: el avatar es
 * estatico y un bucle gastaria bateria en el celular para nada.
 *
 * Sin WebGL (o en las pruebas, con jsdom) se muestra `alternativa`, que lleva las medidas en
 * texto: es tambien lo que lee el lector de pantalla.
 */
import { useEffect, useRef, useState, type ReactNode } from "react";
import { esqueleto, type MedidasManiqui, type Vec3 } from "@/lib/maniqui";
import { textos } from "@/lib/textos";

export interface Figura {
  medidas: MedidasManiqui;
  guardia: "ortodoxa" | "zurda" | null | undefined;
  lado: "A" | "B";
  extension: { adelantada: number; atrasada: number };
  /* Posicion en x (metros) y hacia donde mira: 1 hacia +x, -1 hacia -x. */
  x: number;
  mira: 1 | -1;
  opacidad?: number;
}

const COLOR_PIEL = 0x8a8f96;

function colorDe(lado: "A" | "B"): number {
  const v = typeof window === "undefined" ? "" :
    getComputedStyle(document.documentElement).getPropertyValue(lado === "A" ? "--color-a-trazo" : "--color-b-trazo").trim();
  return /^#[0-9a-f]{6}$/i.test(v) ? parseInt(v.slice(1), 16) : lado === "A" ? 0xe5563f : 0x4a90e2;
}

export function Avatar3D({
  figuras, etiqueta, alternativa, alto = 300,
}: {
  figuras: Figura[];
  etiqueta: string;
  alternativa: ReactNode;
  alto?: number;
}) {
  const caja = useRef<HTMLDivElement>(null);
  const [sinWebgl, setSinWebgl] = useState(false);
  // Lo que cambia entre renders y obliga a rearmar la escena.
  const clave = JSON.stringify(figuras);

  useEffect(() => {
    const el = caja.current;
    if (!el) return;
    let vivo = true;
    let limpiar = () => {};
    import("three").then((THREE) => {
      if (!vivo) return;
      let renderer: InstanceType<typeof THREE.WebGLRenderer>;
      try {
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      } catch {
        setSinWebgl(true);
        return;
      }
      const ancho = el.clientWidth || 320;
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      // El tamano en CSS lo pone el contenedor (100%) y no Three.js: si el lienzo fija su ancho
      // en pixeles, empuja a su contenedor y el ResizeObserver lo agranda en bucle (en el
      // celular se veia scroll horizontal).
      renderer.setSize(ancho, alto, false);
      Object.assign(renderer.domElement.style, { display: "block", width: "100%", height: `${alto}px` });
      renderer.domElement.setAttribute("aria-hidden", "true");
      renderer.domElement.style.touchAction = "pan-y";
      el.appendChild(renderer.domElement);

      const escena = new THREE.Scene();
      escena.add(new THREE.HemisphereLight(0xffffff, 0x404040, 2.2));
      const sol = new THREE.DirectionalLight(0xffffff, 1.6);
      sol.position.set(2, 4, 3);
      escena.add(sol);

      const raiz = new THREE.Group();
      escena.add(raiz);
      const geometrias: { dispose: () => void }[] = [];

      const v = (p: Vec3) => new THREE.Vector3(p[0], p[1], p[2]);
      function segmento(grupo: InstanceType<typeof THREE.Group>, a: Vec3, b: Vec3, r: number, mat: InstanceType<typeof THREE.Material>) {
        const pa = v(a), pb = v(b);
        const largo = pa.distanceTo(pb);
        const g = new THREE.CapsuleGeometry(r, Math.max(largo, 1e-3), 4, 10);
        geometrias.push(g);
        const m = new THREE.Mesh(g, mat);
        m.position.copy(pa.clone().add(pb).multiplyScalar(0.5));
        m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), pb.clone().sub(pa).normalize());
        grupo.add(m);
      }
      function esfera(grupo: InstanceType<typeof THREE.Group>, p: Vec3, r: number, mat: InstanceType<typeof THREE.Material>) {
        const g = new THREE.SphereGeometry(r, 18, 14);
        geometrias.push(g);
        const m = new THREE.Mesh(g, mat);
        m.position.copy(v(p));
        grupo.add(m);
      }

      let altoMax = 0;
      let xMin = Infinity, xMax = -Infinity;
      for (const f of figuras) {
        const e = esqueleto(f.medidas, f.guardia, f.extension);
        const H = f.medidas.altura;
        altoMax = Math.max(altoMax, H);
        const transparente = (f.opacidad ?? 1) < 1;
        const piel = new THREE.MeshStandardMaterial({ color: COLOR_PIEL, roughness: 0.7, transparent: transparente, opacity: f.opacidad ?? 1 });
        const guante = new THREE.MeshStandardMaterial({ color: colorDe(f.lado), roughness: 0.45, transparent: transparente, opacity: f.opacidad ?? 1 });
        const grupo = new THREE.Group();
        const r = 0.028 * H;
        // Tronco: de la pelvis al cuello, mas ancho que un miembro.
        segmento(grupo, e.pelvis, e.cuello, 0.09 * H, piel);
        segmento(grupo, e.hombro.izq, e.hombro.der, 0.035 * H, piel);
        segmento(grupo, e.cadera.izq, e.cadera.der, 0.045 * H, piel);
        esfera(grupo, e.cabeza, e.radioCabeza, piel);
        for (const l of ["izq", "der"] as const) {
          segmento(grupo, e.hombro[l], e.codo[l], r, piel);
          segmento(grupo, e.codo[l], e.muneca[l], r * 0.9, piel);
          esfera(grupo, e.muneca[l], e.radioGuante * 1.15, guante);
          segmento(grupo, e.cadera[l], e.rodilla[l], r * 1.35, piel);
          segmento(grupo, e.rodilla[l], e.tobillo[l], r * 1.1, piel);
        }
        geometrias.push(piel, guante);
        // Girar y no espejar: espejar cambiaria la guardia (un ortodoxo se veria zurdo).
        grupo.rotation.y = f.mira === 1 ? 0 : Math.PI;
        grupo.position.x = f.x;
        raiz.add(grupo);
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
      escena.remove(raiz);
      pivote.add(raiz);
      escena.add(pivote);
      pivote.rotation.y = -0.35;
      const dibujar = () => renderer.render(escena, camara);
      dibujar();

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

      limpiar = () => {
        ro?.disconnect();
        for (const g of geometrias) g.dispose();
        renderer.dispose();
        renderer.domElement.remove();
      };
    }).catch(() => vivo && setSinWebgl(true));
    return () => {
      vivo = false;
      limpiar();
    };
    // `clave` resume `figuras`: rearmar solo cuando cambian de verdad.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clave, alto]);

  return (
    <figure className="flex min-w-0 flex-col gap-2">
      <div ref={caja} role="img" aria-label={etiqueta} className="w-full overflow-hidden rounded-md bg-superficie-2" style={{ minHeight: sinWebgl ? undefined : alto }}>
        {sinWebgl && <p className="p-4 text-sm text-texto-3">{textos.avatarSinWebgl}</p>}
      </div>
      <figcaption className={sinWebgl ? "text-sm text-texto-2" : "sr-only"}>{alternativa}</figcaption>
    </figure>
  );
}
