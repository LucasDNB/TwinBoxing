"""
BoxTwin - Recortes de persona para entrenar el detector de guantes.

POR QUE EXISTE
  El detector se va a usar sobre video de ring, donde el guante mide 4 o 5 pixeles en el
  cuadro completo. En el dataset publico mide el 39% del alto de la imagen: son fotos de
  producto. Entrenar sobre el cuadro entero es entrenar para otra escala.

  Recortar la persona normaliza eso y no por comodidad: el guante es siempre cerca de 0,167
  del alto de un cuerpo, no importa a que distancia este la camara. Dentro de un recorte
  llevado a 320x320 el guante mide lo mismo en una foto de estudio que en un plano lejano de
  ringside, asi que la escala deja de ser una variable. Ademas le da al detector la
  asociacion guante->persona gratis, que es lo que despues necesita el filtro de identidad:
  con deteccion sobre el cuadro entero habria que resolver de quien es cada guante, y en un
  clinch eso es el mismo problema dificil que estamos tratando de evitar.

  El filtro por proporcion no es cosmetico. Medido sobre la v3: los cuadros de video dan
  guante/persona 0,155, que es la proporcion real, y las fotos web dan 0,287, casi el doble,
  porque son retratos donde alguien muestra el guante a camara. Ese recorte ensena una
  proporcion que en un ring no existe.

QUE HACE
  Corre YOLOv8-pose sobre cada imagen, recorta las personas que contienen algun guante,
  remapea las cajas a coordenadas del recorte y descarta lo que queda fuera de la banda de
  proporcion. Conserva el split del dataset de entrada, que ya viene agrupado por clip.

USO
  boxtwin-guantes recortes data/roboflow-v3 --out data/recortes
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ConfigRecortes", "recorte_de", "construir_recortes"]


@dataclass(frozen=True)
class ConfigRecortes:
    """
    Parametros del recorte.

    `ratio_min` y `ratio_max` acotan el alto del guante como fraccion del alto de la persona.
    La referencia anatomica es 0,167; la banda deja pasar variacion real de pose y encuadre y
    corta los retratos de primer plano, que son el grueso de lo que el dataset publico tiene
    de mas.
    """

    margen: float = 0.08          # se agranda la caja de persona para no cortar el guante
    alto_minimo: int = 64         # personas mas chicas no dan un recorte con detalle
    ratio_min: float = 0.05
    ratio_max: float = 0.35
    conf_persona: float = 0.25


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def recorte_de(
    caja_persona: tuple[float, float, float, float],
    cajas_guante: list[tuple[int, float, float, float, float]],
    ancho: int,
    alto: int,
    cfg: ConfigRecortes,
) -> tuple[tuple[int, int, int, int], list[tuple[int, float, float, float, float]]] | None:
    """
    Recorte de una persona y sus guantes en coordenadas del recorte.

    Las cajas entran normalizadas a la imagen y salen normalizadas al recorte. Devuelve None
    si el recorte no sirve: persona muy chica, sin guantes adentro, o con una proporcion
    guante/cuerpo fuera de la banda.
    """
    x0, y0, x1, y1 = caja_persona
    ph = y1 - y0
    if ph < cfg.alto_minimo:
        return None

    m = cfg.margen
    cx0 = int(_clamp(x0 - m * (x1 - x0), 0, ancho))
    cy0 = int(_clamp(y0 - m * ph, 0, alto))
    cx1 = int(_clamp(x1 + m * (x1 - x0), 0, ancho))
    cy1 = int(_clamp(y1 + m * ph, 0, alto))
    cw, ch = cx1 - cx0, cy1 - cy0
    if cw < 16 or ch < 16:
        return None

    salida = []
    for clase, gcx, gcy, gw, gh in cajas_guante:
        px, py = gcx * ancho, gcy * alto
        if not (x0 <= px <= x1 and y0 <= py <= y1):
            continue  # el guante es de otra persona
        if not (cfg.ratio_min <= (gh * alto) / ph <= cfg.ratio_max):
            return None  # proporcion de retrato: el recorte entero no sirve
        nx0 = _clamp((px - gw * ancho / 2 - cx0) / cw, 0.0, 1.0)
        nx1 = _clamp((px + gw * ancho / 2 - cx0) / cw, 0.0, 1.0)
        ny0 = _clamp((py - gh * alto / 2 - cy0) / ch, 0.0, 1.0)
        ny1 = _clamp((py + gh * alto / 2 - cy0) / ch, 0.0, 1.0)
        if nx1 - nx0 < 0.01 or ny1 - ny0 < 0.01:
            continue  # quedo pegado al borde y recortado a nada
        salida.append((clase, (nx0 + nx1) / 2, (ny0 + ny1) / 2, nx1 - nx0, ny1 - ny0))

    if not salida:
        return None
    return (cx0, cy0, cx1, cy1), salida


def _nombre_corto(stem: str, tope: int = 48) -> str:
    """
    Acorta el nombre conservando unicidad.

    El dataset trae nombres de scraping de hasta 200 caracteres, y con el sufijo del recorte
    pasan el limite del sistema de archivos. Se recorta y se le pega un hash del nombre
    completo, que es determinista: la misma imagen da el mismo recorte en cualquier corrida.
    """
    if len(stem) <= tope:
        return stem
    h = hashlib.sha1(stem.encode()).hexdigest()[:8]
    return f"{stem[:tope]}_{h}"


def _leer_yolo(txt: Path) -> list[tuple[int, float, float, float, float]]:
    cajas = []
    if not txt.exists():
        return cajas
    for linea in txt.read_text().splitlines():
        p = linea.split()
        if len(p) != 5:
            continue
        try:
            cajas.append((int(p[0]), *(float(v) for v in p[1:])))
        except ValueError:
            continue
    return cajas


def construir_recortes(
    entrada: Path,
    salida: Path,
    modelo_pose: str,
    cfg: ConfigRecortes | None = None,
    progreso=None,
) -> dict:
    """
    Recorre el dataset de entrada y escribe uno nuevo de recortes de persona.

    Conserva el split de la entrada: ya viene agrupado por clip de origen, y rehacerlo aca
    sobre recortes -que son varios por imagen- podria mandar dos recortes de la misma foto a
    lados distintos.
    """
    import cv2
    from ultralytics import YOLO

    cfg = cfg or ConfigRecortes()
    entrada, salida = Path(entrada), Path(salida)
    modelo = YOLO(modelo_pose)

    for s in ("train", "valid", "test"):
        (salida / "images" / s).mkdir(parents=True, exist_ok=True)
        (salida / "labels" / s).mkdir(parents=True, exist_ok=True)

    stats = {
        "imagenes_leidas": 0, "sin_persona": 0, "recortes": 0, "cajas": 0,
        "descartados_por_proporcion": 0, "descartados_sin_guante": 0,
        "descartados_por_tamano": 0, "ilegibles": 0,
    }
    conteos = {s: {"recortes": 0, "cajas": 0} for s in ("train", "valid", "test")}

    archivos = [
        (s, f)
        for s in ("train", "valid", "test")
        for f in sorted((entrada / "images" / s).iterdir())
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    for i, (split, f) in enumerate(archivos):
        img = cv2.imread(str(f))
        if img is None:
            stats["ilegibles"] += 1
            continue
        stats["imagenes_leidas"] += 1
        alto, ancho = img.shape[:2]
        guantes = _leer_yolo(entrada / "labels" / split / f"{f.stem}.txt")
        if not guantes:
            continue

        r = modelo.predict(img, conf=cfg.conf_persona, imgsz=640, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            stats["sin_persona"] += 1
            continue

        for j, caja in enumerate(r.boxes.xyxy.cpu().numpy().tolist()):
            res = recorte_de(tuple(caja), guantes, ancho, alto, cfg)
            if res is None:
                # Se distingue el motivo para poder ajustar la banda con un numero y no a ojo.
                if (caja[3] - caja[1]) < cfg.alto_minimo:
                    stats["descartados_por_tamano"] += 1
                elif any(caja[0] <= g[1] * ancho <= caja[2] and caja[1] <= g[2] * alto <= caja[3]
                         for g in guantes):
                    stats["descartados_por_proporcion"] += 1
                else:
                    stats["descartados_sin_guante"] += 1
                continue
            (cx0, cy0, cx1, cy1), cajas = res
            nombre = f"{_nombre_corto(f.stem)}_p{j}"
            cv2.imwrite(str(salida / "images" / split / f"{nombre}.jpg"), img[cy0:cy1, cx0:cx1])
            (salida / "labels" / split / f"{nombre}.txt").write_text(
                "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in cajas)
            )
            stats["recortes"] += 1
            stats["cajas"] += len(cajas)
            conteos[split]["recortes"] += 1
            conteos[split]["cajas"] += len(cajas)

        if progreso is not None and (i + 1) % 250 == 0:
            progreso(i + 1, len(archivos))

    clases = json.loads((entrada / "manifiesto.json").read_text())["clases"] \
        if (entrada / "manifiesto.json").exists() else ["Boxing-Glove"]
    # La ruta va absoluta: ultralytics resuelve `path` contra su propio directorio de
    # datasets configurado, no contra la ubicacion del yaml, asi que un "." relativo lo manda
    # a buscar las imagenes a otro lado.
    (salida / "data.yaml").write_text(
        "# Generado por boxtwin-guantes. Recortes de persona.\n"
        f"path: {salida.resolve()}\n"
        "train: images/train\nval: images/valid\ntest: images/test\n"
        f"nc: {len(clases)}\nnames: {json.dumps(clases)}\n"
    )
    manifiesto = {
        "kind": "boxtwin.guantes.recortes",
        "entrada": str(entrada),
        "modelo_pose": str(modelo_pose),
        "config": {
            "margen": cfg.margen, "alto_minimo": cfg.alto_minimo,
            "ratio_min": cfg.ratio_min, "ratio_max": cfg.ratio_max,
            "conf_persona": cfg.conf_persona,
        },
        "clases": clases,
        "conteos": conteos,
        "stats": stats,
    }
    (salida / "manifiesto.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False) + "\n"
    )
    return manifiesto
