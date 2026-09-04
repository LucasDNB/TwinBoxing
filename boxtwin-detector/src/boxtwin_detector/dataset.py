"""
BoxTwin - Armado del dataset del detector, desde el export `sequence`.

POR QUE EXISTE
  Entre lo que el anotador exporta y lo que un modelo por cuadro puede consumir hay tres
  cosas que no son de forma sino de contenido, y cada una arruina el resultado en silencio
  si no se hace:

  1. REMUESTREO A UN FPS COMUN. Pacquiao corre a 59,94 fps y las otras dos fuentes a 30. En
     milisegundos los golpes duran parecido (200, 233 y 337 ms de mediana) pero EN CUADROS
     Pacquiao dura casi el doble. Un modelo con receptive field fijo en cuadros tendria que
     aprender dos escalas temporales para la misma accion fisica, y el fold que deja
     Pacquiao afuera es justo el que cambia el fps: se confundiria un problema de unidades
     con uno de generalizacion.

     El remuestreo se hace sobre los KEYPOINTS y las features se calculan despues. Al reves
     -features a 60 y despues submuestrear- las derivadas quedarian en cuadros de 60 fps y
     no habria arreglo posterior.

  2. LAS ETIQUETAS SE REMUESTREAN COMO SEGMENTOS, no como serie. Un submuestreo ingenuo de
     la serie tira la B, que ocupa un solo cuadro, y dos golpes pegados del mismo brazo se
     funden en uno.

  3. LOS AMAGUES NO SON FONDO. Un amague es movimiento de brazo disenado para parecer un
     golpe; es el negativo mas dificil que existe en este dominio. Etiquetarlo O le ensena
     al modelo que el gesto de golpe es fondo. Se enmascara, que es lo mismo que hace el
     anotador con los cuadros sin identidad resuelta: no es fondo, es "no sabemos".

  4. FUERA DEL TRAMO ANOTADO NO HAY FONDO, HAY IGNORANCIA. El export cubre el video entero,
     pero de Pacquiao solo esta anotado el round 1: 2,9 minutos de 87. Los cuadros del round
     7 con identidad resuelta entrarian como O, o sea que el modelo aprenderia "aca no hay
     golpe" sobre metraje donde nadie miro. Medido antes de acotarlo: 1228 cuadros usables
     de Pacquiao (7% de los suyos), 840 de Sparring y 336 de sparring-3.

     El tramo se deduce del primer y el ultimo cuadro etiquetado, que es lo unico que el
     export permite saber. Cuesta el fondo legitimo de antes del primer golpe y despues del
     ultimo, y ese error va en la direccion segura: se pierden negativos verdaderos en vez
     de inventarlos.

  5. LA POSE INTERPOLADA NO ES UNA POSE OBSERVADA. Cuando la identidad tiene un hueco, el
     anotador lo rellena interpolando linealmente entre los dos extremos. Es la mejor
     estimacion disponible, pero no es un dato medido, y una recta entre dos puntos no
     tiene la firma temporal que el detector busca.

     Entran por defecto, porque sacarlas quita cuadros positivos. Con `interpolados=False`
     se las trata como lo que son -no sabemos- y salen de la mascara, igual que los amagues
     y los cuadros fuera del tramo anotado.

     Importa desparejo. Medido: el 23,6% de los golpes de Sparring contiene algun cuadro
     interpolado, contra 3,3% en sparring-3 y 0% en las cuatro fuentes nuevas. Sparring es
     el peor fold en las cuatro rondas, y esa es la explicacion mas probable.

  LIMITACION CONOCIDA: los eventos `aborted` no llegan al export en ningun espacio de clases,
  asi que sus cuadros quedan como O. Son 9 sobre 676 (1,3%): se declara y no se corrige, que
  corregirlo pedia que el detector leyera el annot.json y dejara de consumir la interfaz.

QUE HACE
  Lee `<base>.sequence.npz` mas su meta, remuestrea, calcula features por carril y devuelve
  los tensores del detector con su mascara y su procedencia.

  Cuatro carriles por fuente: dos peleadores por dos brazos. Cada carril es una secuencia
  independiente, porque las features estan espejadas y el brazo derecho ya se ve como uno
  izquierdo.

USO
  fuente = construir(Path("exports/spar.sequence.npz"))
  escribir(fuente, Path("data/"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from boxtwin_detector.bio import Segmento, etiquetas_de, segmentos_de
from boxtwin_detector.features import N_FEATURES, NOMBRES, features_de

__all__ = ["Fuente", "FPS_DESTINO", "construir", "escribir", "leer", "indice_remuestreo",
           "remuestrear_segmentos"]

FPS_DESTINO = 30.0
CLASE_AMAGUE = "feint"


@dataclass
class Fuente:
    """Los tensores de una fuente, listos para el modelo."""

    nombre: str
    features: np.ndarray   # (carriles, T, F) float32
    labels: np.ndarray     # (carriles, T) int8, O/B/I
    usable: np.ndarray     # (carriles, T) bool
    carriles: list[str]
    fps: float
    conteos: dict = field(default_factory=dict)
    procedencia: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return self.features.shape[1]


def indice_remuestreo(T: int, fps_origen: float, fps_destino: float) -> np.ndarray:
    """Indices del video original que corresponden a cada cuadro del nuevo timebase."""
    if fps_origen <= 0 or fps_destino <= 0:
        raise ValueError("fps tiene que ser positivo")
    T_nuevo = max(1, int(np.floor(T * fps_destino / fps_origen)))
    idx = np.round(np.arange(T_nuevo) * fps_origen / fps_destino).astype(np.int64)
    return np.clip(idx, 0, T - 1)


def remuestrear_segmentos(
    segs: list[Segmento], fps_origen: float, fps_destino: float, T_nuevo: int
) -> tuple[list[Segmento], int]:
    """
    Lleva los segmentos al nuevo timebase, sin dejarlos solaparse.

    El intervalo se mapea como SEMIABIERTO y no extremo por extremo. Un golpe [100, 111] a
    60 fps son 12 cuadros; mapeando los dos extremos por separado da [50, 56], que son 7, y
    el golpe engorda un 17% al cambiar de fps. Tratado como [100, 112) da [50, 56) = [50,
    55], que son los 6 que corresponden. Con este mapeo dos segmentos que no se solapaban
    tampoco se solapan despues, asi que el ajuste de abajo solo entra en el caso extremo en
    que un golpe se comprime a cero cuadros.

    Cuando entra, se recorta el que empieza despues -igual que hace el export con los
    solapamientos del mismo brazo- y se cuenta: un ajuste silencioso aca es un golpe que
    cambia de duracion sin que nadie se entere.
    """
    k = fps_destino / fps_origen
    ajustados = 0
    salida: list[Segmento] = []
    fin_previo = -1
    for s in sorted(segs, key=lambda x: x.inicio):
        a = int(round(s.inicio * k))
        b = int(round((s.fin + 1) * k)) - 1
        if b < a:
            b = a
        if a <= fin_previo:
            a = fin_previo + 1
            ajustados += 1
            if b < a:
                b = a
        if a >= T_nuevo:
            ajustados += 1
            continue
        b = min(b, T_nuevo - 1)
        salida.append(Segmento(a, b, s.clase))
        fin_previo = b
    return salida, ajustados


def construir(
    npz: Path, fps_destino: float = FPS_DESTINO, interpolados: bool = True
) -> Fuente:
    """
    Arma los tensores de una fuente a partir de su export `sequence`.

    Con `interpolados=False`, los cuadros de pose rellenada salen de la mascara: no son un
    dato observado y el modelo no deberia responder por ellos.
    """
    npz = Path(npz)
    meta_path = npz.with_name(npz.name.replace(".npz", ".meta.json"))
    if not meta_path.is_file():
        raise FileNotFoundError(f"falta la metadata del export: {meta_path.name}")
    meta = json.loads(meta_path.read_text())
    if meta.get("channels") != "per-arm":
        raise ValueError(
            "el export tiene que ser --channels per-arm: en un solo carril un 1-2 pierde "
            "uno de los dos golpes"
        )

    d = np.load(npz)
    kp, sc = d["keypoints"], d["kp_score"]
    labels, valid, interp = d["labels"], d["valid"], d["interpolated"]
    lanes = [str(x) for x in d["lanes"]]
    clases = [str(x) for x in d["classes"]]
    fps = float(meta["video"]["fps"])
    T = kp.shape[1]

    i_amague = clases.index(CLASE_AMAGUE) if CLASE_AMAGUE in clases else None
    if i_amague is None:
        raise ValueError(
            "el export no distingue amagues: hace falta --classes 14, si no un amague "
            "queda como fondo y le ensena al modelo que el gesto de golpe es fondo"
        )

    idx = indice_remuestreo(T, fps, fps_destino)
    T_nuevo = len(idx)
    kp_r, sc_r, valid_r = kp[:, idx], sc[:, idx], valid[:, idx]
    interp_r = interp[:, idx]

    n_carriles = labels.shape[0] * labels.shape[1]
    F = np.zeros((n_carriles, T_nuevo, N_FEATURES), np.float32)
    L = np.zeros((n_carriles, T_nuevo), np.int8)
    U = np.zeros((n_carriles, T_nuevo), bool)
    nombres_carril: list[str] = []

    n_ev = n_amagues = n_ajustados = 0
    for p, peleador in enumerate(["A", "B"]):
        for c, brazo in enumerate(lanes):
            i = p * len(lanes) + c
            nombres_carril.append(f"{peleador}-{brazo}")

            f, ok = features_de(kp_r[p], sc_r[p], brazo)
            F[i] = f
            U[i] = ok & valid_r[p]
            if not interpolados:
                U[i] &= ~interp_r[p]

            segs = segmentos_de(labels[p, c])
            golpes = [s for s in segs if s.clase != i_amague]
            amagues = [s for s in segs if s.clase == i_amague]
            n_ev += len(golpes)
            n_amagues += len(amagues)

            golpes_r, aj = remuestrear_segmentos(golpes, fps, fps_destino, T_nuevo)
            n_ajustados += aj
            L[i] = etiquetas_de(golpes_r, T_nuevo)

            # los amagues no son fondo: salen de la perdida, no se etiquetan O
            amagues_r, _ = remuestrear_segmentos(amagues, fps, fps_destino, T_nuevo)
            for s in amagues_r:
                U[i, s.inicio : s.fin + 1] = False

    # -- cobertura: fuera del tramo anotado no se sabe si hay golpe, asi que no es fondo
    etiquetados = np.where((L != 0).any(axis=0))[0]
    if len(etiquetados):
        cob_ini, cob_fin = int(etiquetados[0]), int(etiquetados[-1])
    else:
        cob_ini, cob_fin = 0, -1
    fuera = np.ones(T_nuevo, bool)
    fuera[cob_ini : cob_fin + 1] = False
    n_fuera = int(U[:, fuera].sum())
    U[:, fuera] = False

    conteos = {
        "carriles": n_carriles,
        "cobertura": [cob_ini, cob_fin],
        "cuadros_fuera_de_cobertura": n_fuera,
        "cuadros": int(T_nuevo),
        "cuadros_originales": int(T),
        "golpes": n_ev,
        "amagues_enmascarados": n_amagues,
        "segmentos_ajustados_por_colision": n_ajustados,
        "interpolados_en_mascara": interpolados,
        "cuadros_interpolados": int(interp_r.sum()),
        "cuadros_usables": int(U.sum()),
        "cuadros_O": int(((L == 0) & U).sum()),
        "cuadros_B": int(((L == 1) & U).sum()),
        "cuadros_I": int(((L == 2) & U).sum()),
        "etiquetados_no_usables": int(((L != 0) & ~U).sum()),
    }
    procedencia = {
        "video": meta["video"]["name"],
        "video_sha256": meta["video"]["sha256"],
        "annot_sha256": meta["annot_sha256"],
        "fps_origen": fps,
        "fps_destino": fps_destino,
        "label_space": meta.get("label_space"),
        "classes": meta.get("classes"),
        "features": NOMBRES,
    }
    return Fuente(
        nombre=Path(meta["video"]["name"]).stem,
        features=F, labels=L, usable=U,
        carriles=nombres_carril, fps=fps_destino,
        conteos=conteos, procedencia=procedencia,
    )


def escribir(f: Fuente, out_dir: Path) -> tuple[Path, Path]:
    """Escribe el npz del detector y su manifiesto al lado."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{f.nombre}.det.npz"
    man = out_dir / f"{f.nombre}.det.json"
    np.savez_compressed(
        npz,
        features=f.features, labels=f.labels, usable=f.usable,
        carriles=np.array(f.carriles), fps=np.array(f.fps),
    )
    man.write_text(json.dumps(
        {"kind": "boxtwin.detector.dataset", "fuente": f.nombre,
         "conteos": f.conteos, "procedencia": f.procedencia},
        indent=2, ensure_ascii=False,
    ) + "\n")
    return npz, man


def leer(npz: Path) -> Fuente:
    """Relee lo que escribio `escribir`."""
    npz = Path(npz)
    man = json.loads(npz.with_suffix(".json").read_text())
    d = np.load(npz)
    return Fuente(
        nombre=man["fuente"],
        features=d["features"], labels=d["labels"], usable=d["usable"],
        carriles=[str(x) for x in d["carriles"]], fps=float(d["fps"]),
        conteos=man["conteos"], procedencia=man["procedencia"],
    )
