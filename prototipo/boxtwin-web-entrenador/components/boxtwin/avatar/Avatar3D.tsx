"use client";

/*
 * Avatar 3D ilustrativo (cambios-14-10, W23 y W25; DC-6; modelo MPFB2 y calor del 27-10).
 *
 * Un cuerpo en guardia, a escala de las medidas cargadas: el modelo MPFB2 de
 * public/models/boxer_mpfb2.glb si esta, o el maniqui de primitivas si no. No sale de ningun
 * modelo de elevacion: la profundidad no se mide, y el rotulo lo dice. Sobre la piel va el mapa
 * de calor de los golpes que le dirigen (lib/avatar/exposicion.ts). La geometria vive en
 * lib/maniqui.ts y la escena en ./escena.ts; aca solo se monta.
 *
 * Three.js se importa adentro del efecto, asi queda en un chunk aparte que el navegador baja
 * solo cuando el bloque se monta y no cuenta en el JS inicial de la ruta (03 §9, CA-23.1).
 * Se dibuja a pedido (al armar, al arrastrar y al mover un ajuste del calor), sin un bucle de
 * animacion: el avatar es estatico y un bucle gastaria bateria en el celular para nada.
 *
 * Sin WebGL (o en las pruebas, con jsdom) se muestra `alternativa`, que lleva las medidas en
 * texto: es tambien lo que lee el lector de pantalla.
 */
import { useEffect, useRef, useState, type ReactNode } from "react";
import type { AjustesCalor, Escena, Figura } from "@/components/boxtwin/avatar/escena";
import { textos } from "@/lib/textos";

export type { AjustesCalor, Figura } from "@/components/boxtwin/avatar/escena";

export const AJUSTES_CALOR: AjustesCalor = { activo: true, opacidad: 0.85, umbral: 0.1 };

export function Avatar3D({
  figuras, etiqueta, alternativa, alto = 300, ajustes = AJUSTES_CALOR,
}: {
  figuras: Figura[];
  etiqueta: string;
  alternativa: ReactNode;
  alto?: number;
  ajustes?: AjustesCalor;
}) {
  const caja = useRef<HTMLDivElement>(null);
  const escena = useRef<Escena | null>(null);
  const ultimos = useRef(ajustes);
  const [sinWebgl, setSinWebgl] = useState(false);
  // Lo que cambia entre renders y obliga a rearmar la escena.
  const clave = JSON.stringify(figuras);

  useEffect(() => {
    const el = caja.current;
    if (!el) return;
    let vivo = true;
    import("@/components/boxtwin/avatar/escena")
      .then(({ armarEscena }) => armarEscena(el, figuras, alto, ultimos.current))
      .then((e) => {
        if (!e) return vivo && setSinWebgl(true);
        if (!vivo) return e.limpiar();
        escena.current = e;
        el.dataset.modelo = e.modelo;
      })
      .catch(() => vivo && setSinWebgl(true));
    return () => {
      vivo = false;
      escena.current?.limpiar();
      escena.current = null;
    };
    // `clave` resume `figuras`: rearmar solo cuando cambian de verdad.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clave, alto]);

  // Los ajustes del calor solo cambian uniforms: no se rearma la escena.
  useEffect(() => {
    ultimos.current = ajustes;
    escena.current?.ajustar(ajustes);
  }, [ajustes]);

  return (
    <figure className="flex min-w-0 flex-col gap-2">
      <div ref={caja} role="img" aria-label={etiqueta} className="w-full overflow-hidden rounded-md bg-superficie-2" style={{ minHeight: sinWebgl ? undefined : alto }}>
        {sinWebgl && <p className="p-4 text-sm text-texto-3">{textos.avatarSinWebgl}</p>}
      </div>
      <figcaption className={sinWebgl ? "text-sm text-texto-2" : "sr-only"}>{alternativa}</figcaption>
    </figure>
  );
}
