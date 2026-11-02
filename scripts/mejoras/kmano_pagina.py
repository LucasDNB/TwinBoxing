"""
BoxTwin - Pagina para marcar K-mano de E12 (L2, 31-10-2026).

POR QUE EXISTE
  K-mano (docs/specs/2026-10-30-mejoras-post-e8/04-e12-guantes-en-detector.md §4) pide que
  Lucas marque, en 200 cuadros de sparring-3-rounds con guante asignado, si el guante va a la
  mano correcta; con menos de 90% E12 no corre. Mirarlos en el anotador uno por uno es lento y
  no deja registro. Esto arma una pagina con los 200 recortes y una respuesta por clic.

QUE HACE
  Lee kmano.json (lo escribe guantes_por_cuadro.py, semilla 0), y por cada item lee el cuadro
  del video de raiz-v8, recorta la caja del peleador con un margen, marca el centro del guante
  y lo embebe como JPEG. La pregunta es la que decide K-mano: "este guante es la mano
  IZQUIERDA (o DERECHA) del peleador", la mano propia de la persona, no la de la pantalla. La
  pose no se dibuja, para no inducir la respuesta. Escribe un HTML; las respuestas quedan en la
  base del artifact (coleccion `respuestas`, un documento por item).

USO
  python scripts/mejoras/kmano_pagina.py --out kmano.html
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rutas  # noqa: E402

FUENTE = "sparring-3-rounds"
ALTO = 380
MARGEN = 0.15


def recortes(items: list[dict], raiz: Path) -> list[str]:
    import cv2

    import comun
    from boxtwin.core.annotations import load
    from boxtwin.core.identity import IdentityResolver
    from boxtwin.core.posecache import PoseCache
    from boxtwin.core.types import FighterId

    d = comun.FUENTES[FUENTE]
    doc, _ = load(raiz / d / "annotations" / f"{FUENTE}.annot.json")
    cache = PoseCache.open(raiz / d / "cache" / f"{FUENTE}.pose.npz")
    resolver = IdentityResolver(doc, cache)
    cap = cv2.VideoCapture(str(raiz / d / "videos" / Path(doc.video.path).name))
    salida = []
    for it in items:
        cap.set(cv2.CAP_PROP_POS_FRAMES, it["cuadro"])
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"no se pudo leer el cuadro {it['cuadro']}")
        pose = resolver.by_fighter(it["cuadro"])[(FighterId.A, FighterId.B)[it["peleador"]]]
        x0, y0, x1, y1 = (float(v) for v in pose.bbox)
        mx, my = (x1 - x0) * MARGEN, (y1 - y0) * MARGEN
        H, W = frame.shape[:2]
        a, b = int(max(0, x0 - mx)), int(max(0, y0 - my))
        c, e = int(min(W, x1 + mx)), int(min(H, y1 + my))
        img = frame[b:e, a:c].copy()
        gx, gy = it["xy"][0] - a, it["xy"][1] - b
        r = max(10, int((c - a) * 0.06))
        cv2.circle(img, (int(gx), int(gy)), r + 3, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(img, (int(gx), int(gy)), r, (0, 230, 255), 3, cv2.LINE_AA)
        k = ALTO / img.shape[0]
        img = cv2.resize(img, (max(1, int(img.shape[1] * k)), ALTO), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
        salida.append("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode())
    cap.release()
    return salida


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kmano", type=Path, default=rutas.MEJORAS / "e12" / "guantes" / "kmano.json")
    ap.add_argument("--raiz", type=Path, default=rutas.RAIZ_V8)
    ap.add_argument("--plantilla", type=Path, default=Path(__file__).with_name("kmano_plantilla.html"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    items = json.loads(args.kmano.read_text())
    imgs = recortes(items, args.raiz)
    datos = [{"id": f"{i:03d}", "cuadro": it["cuadro"], "peleador": "AB"[it["peleador"]],
              "mano": ("izquierda", "derecha")[it["mano"]], "img": img}
             for i, (it, img) in enumerate(zip(items, imgs))]
    html = args.plantilla.read_text().replace("/*DATOS*/[]", json.dumps(datos))
    args.out.write_text(html)
    print(f"{len(datos)} items -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
