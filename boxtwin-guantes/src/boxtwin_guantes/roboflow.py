"""
BoxTwin - Bajada del dataset de guantes desde Roboflow.

POR QUE EXISTE
  El dataset publicado es `boxing-uuhxl/boxing-gloves-detection`, una clase, 4050
  instancias, licencia de dominio publico. Tiene cinco versiones y las cinco hornean
  decisiones que no queremos: v4 y v5 pasan todo a escala de grises, v2 a v5 ecualizan el
  contraste, y las cinco reescalan con "Stretch to" 640x640 sobre imagenes que son 1920x1080.
  Ese estirado no es cosmetico: comprime el eje horizontal a 0,5625 del vertical, asi que un
  guante redondo queda elipse de 1,78 a 1, y despues nuestro recorte de persona le aplica
  OTRO factor distinto. Entrenar sobre eso y aplicar sobre video sin estirar es un cambio de
  dominio que nos meteriamos solos.

  Las cinco versiones incluyen ademas flip vertical en la aumentacion. Un guante de boxeo
  dado vuelta no existe, y ensenarle esa orientacion al modelo le gasta capacidad en algo
  que nunca va a ver.

  Por eso esto no baja un export de version sino los ORIGINALES, por el endpoint de busqueda,
  a resolucion nativa y con las cajas en coordenadas de la imagen original. La aumentacion la
  hace ultralytics en entrenamiento, que la hace mejor y sin horneala en disco.

QUE HACE
  Pagina el endpoint de busqueda, se queda con las imagenes que tienen anotacion, baja el
  original de cada una y escribe las cajas en formato YOLO. Es reanudable: una imagen que ya
  esta en disco no se vuelve a pedir. Deja un manifiesto con la procedencia, que es lo que
  despues permite decir en la tesis sobre que se entreno.

  La api key se lee de ROBOFLOW_API_KEY y no se escribe en ningun archivo de salida.

USO
  ROBOFLOW_API_KEY=... python -m boxtwin_guantes.cli fetch --out data/roboflow --limite 50
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

API = "https://api.roboflow.com"
WORKSPACE = "boxing-uuhxl"
PROYECTO = "boxing-gloves-detection"

# Campos que pedimos por imagen. `url` es el original sin preprocesar, que es el motivo
# entero de bajar por aca en vez de por un export de version.
# Solo lo que se usa. `annotations` se pedia antes creyendo que traia las cajas: trae un
# resumen, {"count": 2, "classes": {...}}, asi que no sirve para etiquetar y se saco.
CAMPOS = ["id", "name", "width", "height", "url"]

__all__ = ["ImagenRemota", "clave_api", "buscar", "descargar_dataset",
           "clip_de_origen", "repartir"]


class ErrorRoboflow(RuntimeError):
    """Falla al hablar con la API. Se distingue para que el CLI no muestre un traceback."""


def clave_api() -> str:
    k = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not k:
        raise ErrorRoboflow(
            "falta ROBOFLOW_API_KEY en el entorno. La clave no se guarda en el repo: "
            "se pasa por variable de entorno en cada corrida"
        )
    return k


@dataclass(frozen=True)
class ImagenRemota:
    """
    Una imagen del catalogo. Sin cajas, a proposito.

    El endpoint de busqueda no devuelve coordenadas: su campo `annotations` es un resumen
    de cuantas cajas hay y de que clase. Lo unico que este registro aporta es el TAMANO
    ORIGINAL, que es lo que permite deshacerle el estirado al export.
    """

    id: str
    nombre: str
    ancho: int
    alto: int
    url: str


def _pedir(ruta: str, payload: dict[str, Any], key: str, timeout: int = 60) -> dict:
    url = f"{API}/{ruta}?api_key={key}"
    datos = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=datos, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise ErrorRoboflow(f"HTTP {e.code} en {ruta}: {e.read()[:200]!r}") from e
    except urllib.error.URLError as e:
        raise ErrorRoboflow(f"no se pudo hablar con {API}: {e.reason}") from e


def _imagen_de(r: dict) -> ImagenRemota | None:
    ancho, alto = int(r.get("width") or 0), int(r.get("height") or 0)
    if not (r.get("id") and r.get("url") and ancho > 0 and alto > 0):
        return None
    return ImagenRemota(
        id=str(r["id"]),
        nombre=str(r.get("name") or r["id"]),
        ancho=ancho,
        alto=alto,
        url=str(r["url"]),
    )


def buscar(key: str, por_pagina: int = 100, tope: int | None = None) -> Iterator[ImagenRemota]:
    """
    Recorre el catalogo entero, pagina por pagina.

    El filtro por split del endpoint se ignora del lado del servidor -probado, devuelve el
    catalogo entero igual- asi que se pagina todo y se filtra aca. Son 8334 registros, de
    los cuales 1625 tienen anotacion.
    """
    offset, vistos = 0, 0
    while True:
        d = _pedir(
            f"{WORKSPACE}/{PROYECTO}/search",
            {"limit": por_pagina, "offset": offset, "fields": CAMPOS},
            key,
        )
        filas = d.get("results") or []
        if not filas:
            return
        for r in filas:
            img = _imagen_de(r)
            if img is not None:
                yield img
                vistos += 1
                if tope is not None and vistos >= tope:
                    return
        offset += len(filas)
        if offset >= int(d.get("total") or 0):
            return


# ---------------------------------------------------------------------------
# Reparto en train/valid/test
#
# Las imagenes son CUADROS DE VIDEO: se llaman "Trimed Box Match _mp4-2092.jpg". Los
# cuadros 2092 y 2093 de la misma pelea son casi la misma imagen, asi que un reparto al
# azar pone copias cuasi-identicas de los dos lados y la validacion mide de mas. El reparto
# tiene que ser POR CLIP DE ORIGEN, igual que el dataset de BoxingVI se reparte por video.
#
# El reparto que trae Roboflow no se usa por eso mismo: no hay manera de saber si respeta
# la procedencia, y si no la respeta sus numeros publicados estan inflados.

import hashlib
import re

_HASH_RF = re.compile(r"\.rf\.[0-9a-f]+", re.IGNORECASE)
_SUFIJO = re.compile(r"[_-]?(mp4|mov|avi|mkv)?[_-]+\d+\.\w+$", re.IGNORECASE)


def clip_de_origen(nombre: str) -> str:
    """
    El clip del que salio un cuadro, para no repartir cuadros vecinos entre splits.

    "Trimed Box Match _mp4-2092.jpg" -> "trimed box match". Si el nombre no tiene la forma
    de cuadro numerado se devuelve el nombre entero, que deja a esa imagen en un grupo
    propio: es el lado conservador, porque agrupar de menos reparte de mas pero nunca mezcla
    dos clips distintos en el mismo grupo.
    """
    # El hash del export va primero: "Trimed-Box-Match-_mp4-0020_jpg.rf.<hash>.jpg" tiene el
    # numero de cuadro en el medio, no pegado a la extension, y sin sacarlo cada cuadro
    # quedaba en un grupo propio. Eso repartia 853 cuadros vecinos de la misma pelea entre
    # train, valid y test: justo la filtracion que esta funcion existe para evitar.
    base = _HASH_RF.split(nombre.strip())[0]
    base = re.sub(r"_jpe?g$", "", base, flags=re.IGNORECASE)
    base = _SUFIJO.sub("", base)
    base = re.sub(r"[_-]?(mp4|mov|avi|mkv)[_-]*\d*$", "", base, flags=re.IGNORECASE)
    return re.sub(r"[\s_-]+", " ", base).strip().lower() or nombre.lower()


def repartir(
    nombres: dict[str, str], val: float = 0.2, test: float = 0.1
) -> dict[str, str]:
    """
    Asigna split a cada imagen agrupando por clip. Determinista por sha1 del clip.

    Devuelve id -> split. No usa random ni semilla: el hash del nombre del clip da el mismo
    reparto en cualquier maquina y en cualquier corrida, que es lo que permite comparar dos
    entrenamientos sin preguntarse si cambio el reparto.
    """
    grupos: dict[str, list[str]] = {}
    for ident, nombre in nombres.items():
        grupos.setdefault(clip_de_origen(nombre), []).append(ident)

    # Los grupos se ordenan por su hash, no por nombre, para que el corte no quede
    # correlacionado con el orden alfabetico de las peleas.
    orden = sorted(grupos, key=lambda g: hashlib.sha1(g.encode()).hexdigest())
    total = sum(len(grupos[g]) for g in orden)
    tope_val, tope_test = val * total, test * total

    salida: dict[str, str] = {}
    n_val = n_test = 0
    for g in orden:
        if n_val < tope_val:
            split, n_val = "valid", n_val + len(grupos[g])
        elif n_test < tope_test:
            split, n_test = "test", n_test + len(grupos[g])
        else:
            split = "train"
        for ident in grupos[g]:
            salida[ident] = split
    return salida


# ---------------------------------------------------------------------------
# Bajada del export
#
# El endpoint de busqueda da las imagenes originales pero NO las coordenadas: su campo
# `annotations` es un resumen, {"count": 2, "classes": {"Boxing-Glove": 2}}. Las cajas solo
# salen de un export de version, y los exports vienen con el preprocesado horneado.
#
# El estirado se deshace aca y no se sufre: las etiquetas YOLO son normalizadas y el
# "Stretch to" es un escalado lineal por eje, asi que las coordenadas normalizadas son
# invariantes. Reescalar la imagen de 640x640 a su relacion de aspecto original deja las
# cajas correctas sin tocar un solo numero. Lo que no vuelve es el detalle horizontal que el
# estirado ya tiro, y eso se declara en el manifiesto.


import shutil
import zipfile


def link_de_export(key: str, version: int, formato: str = "yolov8",
                   workspace: str = WORKSPACE, proyecto: str = PROYECTO) -> tuple[str, dict]:
    """Pide el link de descarga de una version y devuelve (link, ajustes de esa version)."""
    url = f"{API}/{workspace}/{proyecto}/{version}/{formato}?api_key={key}"
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise ErrorRoboflow(f"HTTP {e.code} pidiendo el export v{version}") from e
    except urllib.error.URLError as e:
        raise ErrorRoboflow(f"no se pudo hablar con {API}: {e.reason}") from e
    link = (d.get("export") or {}).get("link")
    if not link:
        raise ErrorRoboflow(f"la API no devolvio link de export para v{version}: {str(d)[:200]}")
    v = d.get("version") or {}
    return link, {
        "imagenes": v.get("images"),
        "splits": v.get("splits"),
        "preprocesado": v.get("preprocessing"),
        "aumentacion": v.get("augmentation"),
    }


def _bajar_archivo(url: str, destino: Path, timeout: int = 600) -> int:
    """Baja a un temporal y renombra: un corte de red no deja un archivo a medias en disco."""
    tmp = destino.with_suffix(destino.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, tmp.open("wb") as f:
            leido = 0
            while True:
                trozo = r.read(1 << 20)
                if not trozo:
                    break
                f.write(trozo)
                leido += len(trozo)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        tmp.unlink(missing_ok=True)
        raise ErrorRoboflow(f"no se pudo bajar {destino.name}: {e}") from e
    tmp.replace(destino)
    return leido


def _leer_labels(txt: Path) -> list[tuple[int, float, float, float, float]]:
    """
    Lee un archivo de etiquetas YOLO, sea de cajas o de poligonos.

    Las dos formas conviven en el mismo export y hay que soportar las dos. La v3 anota los
    cuadros de video con poligonos -`clase x1 y1 x2 y2 ...`- y el resto con cajas de cinco
    campos; medido, son 2404 lineas de poligono contra 0 de caja en los cuadros de video, y
    5478 cajas mas 1583 poligonos en el resto. Un lector que exija cinco campos tira el 100%
    del material de dominio sin decir nada, que es el modo de falla mas caro posible.

    Un poligono se reduce a su caja envolvente. Para un detector de cajas eso no pierde
    nada: la caja es justamente lo que se iba a entrenar.
    """
    cajas = []
    if not txt.exists():
        return cajas
    for linea in txt.read_text().splitlines():
        partes = linea.split()
        if len(partes) < 5:
            continue
        try:
            clase = int(partes[0])
            vals = [float(x) for x in partes[1:]]
        except ValueError:
            continue
        if len(vals) == 4:
            cajas.append((clase, *vals))
            continue
        if len(vals) < 6 or len(vals) % 2:
            continue  # un poligono necesita al menos tres vertices y pares completos
        xs, ys = vals[0::2], vals[1::2]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        if x1 <= x0 or y1 <= y0:
            continue
        cajas.append((clase, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return cajas


def clave_de_nombre(nombre: str) -> str:
    """
    Reduce un nombre a algo comparable entre el export y el catalogo.

    El export renombra: "Trimed Box Match _mp4-0431.jpg" sale como
    "Trimed-Box-Match-_mp4-0431_jpg.rf.<hash>.jpg". Se saca el hash, el sufijo `_jpg` que el
    renombrado deja pegado, la extension, y todo lo que no sea alfanumerico. Los dos lados
    caen en "trimedboxmatchmp40431".
    """
    n = _HASH_RF.split(nombre)[0]
    n = re.sub(r"\.(jpg|jpeg|png|bmp|webp)$", "", n, flags=re.IGNORECASE)
    n = re.sub(r"_jpe?g$", "", n, flags=re.IGNORECASE)
    return re.sub(r"[^0-9a-z]", "", n.lower())


def _catalogo_por_clave(key: str) -> dict[str, ImagenRemota]:
    """Indice del catalogo por clave de nombre, para recuperar el tamano original."""
    salida: dict[str, ImagenRemota] = {}
    for im in buscar(key):
        salida.setdefault(clave_de_nombre(im.nombre), im)
    return salida


def _desestirar(origen: Path, destino: Path, ancho: int, alto: int) -> bool:
    """
    Reescribe la imagen con la relacion de aspecto original.

    Las cajas no se tocan: son normalizadas y el estirado es un escalado lineal por eje, asi
    que sus coordenadas ya son las correctas para la imagen desestirada. Devuelve False si la
    imagen no se pudo leer.
    """
    import cv2

    img = cv2.imread(str(origen))
    if img is None:
        return False
    if img.shape[1] == ancho and img.shape[0] == alto:
        shutil.copyfile(origen, destino)
        return True
    # El lado largo se deja en el tamano que tenia el export: reescalar hacia arriba no
    # inventa detalle, solo pesa mas en disco.
    lado = max(img.shape[0], img.shape[1])
    escala = lado / max(ancho, alto)
    destino_wh = (max(1, round(ancho * escala)), max(1, round(alto * escala)))
    cv2.imwrite(str(destino), cv2.resize(img, destino_wh, interpolation=cv2.INTER_AREA))
    return True


def _clases_del_export(raiz: Path) -> list[str]:
    """Nombres de clase del data.yaml del export, sin depender de pyyaml."""
    y = raiz / "data.yaml"
    if not y.exists():
        return ["Boxing-Glove"]
    for linea in y.read_text().splitlines():
        if linea.strip().startswith("names:"):
            cuerpo = linea.split(":", 1)[1].strip()
            if cuerpo.startswith("["):
                return [c.strip().strip("'\"") for c in cuerpo.strip("[]").split(",") if c.strip()]
    return ["Boxing-Glove"]


def _pares_del_export(raiz: Path) -> list[tuple[Path, Path]]:
    """Todos los (imagen, labels) del export, de los tres splits que traiga."""
    pares = []
    for split in ("train", "valid", "test", "valids"):
        d = raiz / split / "images"
        if not d.is_dir():
            continue
        for img in sorted(d.iterdir()):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                pares.append((img, raiz / split / "labels" / f"{img.stem}.txt"))
    return pares


def descargar_dataset(
    key: str | None,
    salida: Path,
    version: int = 1,
    val: float = 0.2,
    test: float = 0.1,
    zip_local: Path | None = None,
    progreso=None,
) -> dict:
    """
    Toma el export de una version, le deshace el estirado y lo reparte por clip de origen.

    Con `zip_local` usa un export ya bajado a mano y no le pide nada a la API. Sin el, lo
    baja. Reanudable en el paso caro: si el zip ya esta en disco no se vuelve a bajar.

    La clave es opcional: sin ella no se puede consultar el catalogo, y entonces el tamano
    original de cada imagen se asume 16:9. Es lo que son todas las muestreadas, pero es un
    supuesto y el manifiesto lo cuenta en `sin_cruce_con_catalogo`.
    """
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    zip_path = salida / f"export-v{version}.zip"
    crudo = salida / f"_crudo-v{version}"

    if zip_local is not None:
        zip_local = Path(zip_local)
        if not zip_local.is_file():
            raise ErrorRoboflow(f"no existe el zip {zip_local}")
        ajustes = {"origen": "zip bajado a mano", "archivo": zip_local.name}
        if not zip_path.exists():
            shutil.copyfile(zip_local, zip_path)
    else:
        if not key:
            raise ErrorRoboflow("sin clave y sin --zip no hay de donde sacar el export")
        link, ajustes = link_de_export(key, version)
        if not zip_path.exists():
            _bajar_archivo(link, zip_path)
    if not crudo.exists():
        tmp = salida / f"_crudo-v{version}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        tmp.replace(crudo)

    pares = _pares_del_export(crudo)
    if not pares:
        raise ErrorRoboflow(f"el export de v{version} no trajo imagenes en {crudo}")
    clases = _clases_del_export(crudo)

    # El catalogo da el tamano ORIGINAL de cada imagen, que es lo que permite deshacer el
    # estirado. Lo que no aparece se asume 16:9, que es lo que son todas las muestreadas.
    catalogo = _catalogo_por_clave(key) if key else {}
    nombres = {img.stem: (catalogo[k].nombre if (k := clave_de_nombre(img.name)) in catalogo
                          else img.name) for img, _ in pares}
    splits = repartir(nombres, val=val, test=test)

    for s in ("train", "valid", "test"):
        (salida / "images" / s).mkdir(parents=True, exist_ok=True)
        (salida / "labels" / s).mkdir(parents=True, exist_ok=True)

    conteos = {s: {"imagenes": 0, "cajas": 0} for s in ("train", "valid", "test")}
    sin_catalogo = ilegibles = 0
    for i, (img, lbl) in enumerate(pares):
        ident = img.stem
        s = splits[ident]
        original = catalogo.get(clave_de_nombre(img.name))
        destino_img = salida / "images" / s / f"{ident}.jpg"
        destino_lbl = salida / "labels" / s / f"{ident}.txt"
        if not destino_img.exists():
            if original is None:
                # Sin el tamano original no se puede deshacer el estirado, y asumir una
                # relacion de aspecto seria deformar las que no la tenian. El dataset trae
                # fotos de producto cuadradas y 4:3 mezcladas con cuadros de video 16:9, asi
                # que un supuesto unico rompe mas de lo que arregla. Se copia tal cual.
                sin_catalogo += 1
                shutil.copyfile(img, destino_img)
            elif not _desestirar(img, destino_img, original.ancho, original.alto):
                ilegibles += 1
                continue
        cajas = _leer_labels(lbl)
        destino_lbl.write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in cajas)
        )
        conteos[s]["imagenes"] += 1
        conteos[s]["cajas"] += len(cajas)
        if progreso is not None and (i + 1) % 200 == 0:
            progreso(i + 1, len(pares))

    # Absoluta, por el mismo motivo que en recortes.py: ultralytics no resuelve `path`
    # contra la ubicacion del yaml.
    (salida / "data.yaml").write_text(
        "# Generado por boxtwin-guantes.\n"
        f"path: {salida.resolve()}\n"
        "train: images/train\nval: images/valid\ntest: images/test\n"
        f"nc: {len(clases)}\nnames: {json.dumps(clases)}\n"
    )

    clips: dict[str, set[str]] = {}
    for ident, nombre in nombres.items():
        clips.setdefault(clip_de_origen(nombre), set()).add(splits[ident])

    manifiesto = {
        "kind": "boxtwin.guantes.dataset",
        "fuente": {
            "workspace": WORKSPACE, "proyecto": PROYECTO, "version": version,
            "licencia": "Public Domain",
            "ajustes_de_la_version": ajustes,
        },
        "clases": clases,
        "conteos": conteos,
        "imagenes": sum(c["imagenes"] for c in conteos.values()),
        "cajas": sum(c["cajas"] for c in conteos.values()),
        "clips_de_origen": len(clips),
        "clips_en_mas_de_un_split": sorted(c for c, ss in clips.items() if len(ss) > 1),
        "reparto": {"por": "clip de origen", "val": val, "test": test, "determinista": "sha1"},
        "sin_cruce_con_catalogo": sin_catalogo,
        "ilegibles": ilegibles,
        "limitacion": (
            "el export viene estirado a 640x640. A las imagenes que cruzaron con el "
            "catalogo se les devolvio la relacion de aspecto original -las cajas "
            "normalizadas son invariantes a eso- pero el detalle que el estirado descarto "
            "no se recupera. A las que NO cruzaron se las dejo estiradas: sin el tamano "
            "original, asumir una relacion de aspecto deformaria las que no la tenian. "
            "La aumentacion horneada de la version, que incluye flip vertical, quedo adentro"
        ),
    }
    (salida / "manifiesto.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False) + "\n"
    )
    return manifiesto
