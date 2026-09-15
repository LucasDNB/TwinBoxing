#!/usr/bin/env python3
"""
BoxTwin - Arnes de medicion de la cadena completa sobre video, para el Capitulo 5.

POR QUE EXISTE
  El Capitulo 5 declara tres numeros sin medicion detras: 1,8 h de computo por hora de
  video (S-19), 250 W por instancia (S-17) y 1,25 USD por hora de video (S-04). El unico
  dato propio que habia -2,3x tiempo real- cubria solo pose y seguimiento, o sea un
  pedazo de la cadena, y ademas esta en tension con el 1,8.

  Este arnes mide el tiempo de pared de extremo a extremo y el tiempo por etapa, sobre el
  equipo propio y con la cadena que existe, para que S-19 deje de ser un supuesto.

  NO TOCA EL CODIGO DE PRODUCCION. Las marcas de tiempo se ponen envolviendo funciones ya
  existentes en tiempo de ejecucion -_inferir, el callback del tracker, detectar_cortes- en
  vez de editarlas. Asi la logica de procesamiento es bit a bit la misma que corre en
  serio, y el criterio de la especificacion -"la instrumentacion no debe cambiar el
  resultado"- se cumple por construccion y no por inspeccion.

  ADEMAS SE VERIFICA. Cada corrida compara el cache de pose recien calculado contra el que
  ya vive en el proyecto: si los track_id o los keypoints no coinciden, la corrida queda
  marcada y el numero no se usa. Sin eso, un cambio silencioso de resultado pasaria por
  una medicion valida.

QUE HACE
  Corre la cadena sobre un video, N veces, cronometrando cada etapa por separado:

    preparacion     carga de modelo, hash del video, deteccion de cortes de plano
    decodificacion  lectura de cuadros
    pose            YOLOv8l-pose
    seguimiento     BoT-SORT
    identidad       resolucion de track_id a peleador y export de la serie por brazo
    deteccion       features, ensamble de 5 semillas y decodificacion a segmentos
    clasificacion   PoseConv3D sobre cada segmento (subproceso en boxtwin_mmaction)

  La etapa de estimacion de resultado -conectado, bloqueado, esquivado- NO SE MIDE porque
  NO EXISTE como codigo: hoy es un campo que completa a mano el que anota. Ver el apartado
  de desvios en mediciones/protocolo.md.

  Escribe una fila por corrida a mediciones/corridas.csv y no pisa lo ya escrito.

USO
  python ../scripts/medicion_cadena.py --video anotacion-spar-01/videos/01-sparring.mp4 \
      --corridas 3 --out mediciones/corridas.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "boxtwin-annotator" / "src"))
sys.path.insert(0, str(RAIZ / "boxtwin-detector" / "src"))

PY_MMACTION = Path.home() / "miniforge3" / "envs" / "boxtwin_mmaction" / "bin" / "python"

ETAPAS = ["preparacion", "decodificacion", "pose", "seguimiento",
          "identidad", "deteccion", "clasificacion"]

COLUMNAS = [
    "ts_iso", "video", "duracion_video_s", "cuadros", "fps_video",
    "corrida", "estado", "commit", "tiempo_total_s",
    *[f"t_{e}_s" for e in ETAPAS],
    "ratio_computo", "cache_identico", "n_segmentos", "n_detecciones",
    "half", "imgsz", "detect_cuts", "notas",
]


class Reloj:
    """Acumulador de tiempo por etapa. Un solo reloj para que las marcas sean alineables."""

    def __init__(self) -> None:
        self.acum: dict[str, float] = {e: 0.0 for e in ETAPAS}
        self.marcas: list[dict] = []

    @contextmanager
    def etapa(self, nombre: str):
        t0 = time.perf_counter()
        w0 = time.time()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.acum[nombre] = self.acum.get(nombre, 0.0) + dt
            # epoch real, para poder cruzar con las muestras de potencia
            self.marcas.append({"etapa": nombre, "inicio_epoch": w0, "dur_s": dt})

    def sumar(self, nombre: str, dt: float) -> None:
        self.acum[nombre] = self.acum.get(nombre, 0.0) + dt


# ---------------------------------------------------------------------------
# Etapas 1 a 3: pose, seguimiento y decodificacion
# ---------------------------------------------------------------------------


def config_del_proyecto(meta_ref: Path, modelo_pose: Path):
    """
    Adopta la configuracion de inferencia con la que se construyo el cache del proyecto.

    POR QUE NO SE USAN LOS DEFAULTS. PreprocessConfig trae half=False, pero los caches
    reales de este proyecto se construyeron con half=True: siete de nueve lo declaran en su
    meta.json. Medir con FP32 mediria una configuracion que no es la que produjo las
    anotaciones ni la que corre en serio, y sobreestimaria el tiempo de la etapa de pose.

    Lo mismo con detect_cuts. En 01-sparring el cache de referencia se hizo con la
    deteccion de cortes apagada; correrla igual encuentra cortes, reinicia el tracker,
    renumera los track_id y deja las asignaciones de identidad apuntando a otra persona.
    El resultado no seria comparable con el cache de referencia ni con la anotacion.

    Los caches viejos no declaran detect_cuts -es posterior-, y ahi se asume apagada, que
    es lo que hacia el codigo cuando se generaron.
    """
    from boxtwin.preprocess.runner import PreprocessConfig

    if not meta_ref.is_file():
        return PreprocessConfig(make_proxy=False, model=str(modelo_pose)), "sin-referencia"
    inf = json.loads(meta_ref.read_text(encoding="utf-8"))["inference"]
    cortes = inf.get("detect_cuts")
    return PreprocessConfig(
        make_proxy=False,
        model=str(modelo_pose),
        imgsz=int(inf.get("imgsz", 640)),
        conf=float(inf.get("conf", 0.25)),
        iou=float(inf.get("iou", 0.7)),
        half=bool(inf.get("half", False)),
        detect_cuts=bool(cortes) if cortes is not None else False,
        cut_threshold=float(inf.get("cut_threshold", 0.3)),
    ), ("del-proyecto" if cortes is not None else "del-proyecto-sin-cortes")


def correr_preproceso(video: Path, destino: Path, reloj: Reloj, cfg_base,
                      max_cuadros: int | None = None) -> dict:
    """
    Corre el preproceso real -pose + seguimiento- envolviendo sus funciones para cronometrar.

    El proxy queda apagado: es el video liviano que usa la interfaz del anotador para
    reproducir, no una etapa del analisis. Dejarlo prendido sumaria un ffmpeg en paralelo
    al tiempo de pared sin que aporte nada al informe tactico.
    """
    from boxtwin.preprocess import runner as R

    cfg = cfg_base

    # -- limite de cuadros. Se implementa envolviendo VideoCapture.read para que devuelva
    #    fin de archivo pasado el corte: el runner no se entera y su logica queda intacta.
    #    Hace falta para pacquiao, que dura 38 minutos y tiene anotacion sobre los tres
    #    primeros; procesarlo entero gastaria una hora y tres cuartos por corrida para que
    #    las etapas 3 a 5 trabajen sobre el 8% de los cuadros.
    #    cv2.VideoCapture es un tipo inmutable y no deja reemplazarle read, asi que se
    #    sustituye la clase por una fabrica que devuelve un proxy.
    #
    #    OJO CON LO QUE ESTO NO ACOTA: la deteccion de cortes de plano corre por ffmpeg en
    #    un subproceso y recorre el archivo entero igual. Sobre un video largo eso deja el
    #    costo de esa pasada atribuido a un tramo corto, y el ratio sale inflado. Para
    #    medir un tramo de un video largo conviene recortar el archivo con ffmpeg y correr
    #    sobre el recorte, que es lo que se hizo con pacquiao; ver el desvio D-8.
    restaurar_read = None
    if max_cuadros:
        import cv2 as _cv2

        _Cap = _cv2.VideoCapture

        class _CapLimitada:
            def __init__(self, *a, **k):
                self._c = _Cap(*a, **k)
                self._n = 0

            def read(self, *a, **k):
                if self._n >= max_cuadros:
                    return False, None
                self._n += 1
                return self._c.read(*a, **k)

            def __getattr__(self, n):
                return getattr(self._c, n)

        _cv2.VideoCapture = _CapLimitada
        restaurar_read = lambda: setattr(_cv2, "VideoCapture", _Cap)  # noqa: E731

    # -- envoltura de la inferencia por cuadro. Todo lo que no cae adentro de _inferir
    #    durante el bucle es decodificacion: se deduce por diferencia al final.
    inferir_original = R._inferir
    t_infer = [0.0]

    def inferir_medido(modelo, frame, cfg_):
        t0 = time.perf_counter()
        r = inferir_original(modelo, frame, cfg_)
        t_infer[0] += time.perf_counter() - t0
        return r

    # -- envoltura del callback del tracker, que es la costura donde BoT-SORT hace la
    #    asociacion. Separa seguimiento de pose sin tocar ultralytics.
    t_track = [0.0]

    def envolver_tracker(modelo) -> None:
        cbs = getattr(getattr(modelo, "predictor", None), "callbacks", None)
        if not cbs:
            return
        lista = cbs.get("on_predict_postprocess_end") or []
        nuevas = []
        for fn in lista:
            def medido(*a, _fn=fn, **k):
                t0 = time.perf_counter()
                try:
                    return _fn(*a, **k)
                finally:
                    t_track[0] += time.perf_counter() - t0
            nuevas.append(medido)
        cbs["on_predict_postprocess_end"] = nuevas

    cargar_original = R._cargar_modelo
    t_prep = [0.0]

    def cargar_medido(path, cfg_):
        t0 = time.perf_counter()
        m = cargar_original(path, cfg_)
        # se fuerza una inferencia para que el predictor exista y se pueda envolver el
        # callback; ese costo es carga de modelo, no procesamiento de video
        t_prep[0] += time.perf_counter() - t0
        return _EnvolturaModelo(m, envolver_tracker)

    cortes_medido = _envolver_cortes(t_prep)
    hash_original = R.sha256_file

    def hash_medido(p):
        t0 = time.perf_counter()
        r = hash_original(p)
        t_prep[0] += time.perf_counter() - t0
        return r

    R._inferir = inferir_medido
    R._cargar_modelo = cargar_medido
    R.sha256_file = hash_medido
    _parchear_cortes(cortes_medido)

    try:
        t0 = time.perf_counter()
        res = R.preprocess(video, destino, cfg, restart=True, force=True)
        total = time.perf_counter() - t0
    finally:
        R._inferir = inferir_original
        R._cargar_modelo = cargar_original
        R.sha256_file = hash_original
        _desparchear_cortes()
        if restaurar_read is not None:
            restaurar_read()

    pose = max(0.0, t_infer[0] - t_track[0])
    # lo que queda del bucle despues de sacar inferencia y preparacion es leer cuadros
    decod = max(0.0, total - t_infer[0] - t_prep[0])

    reloj.sumar("preparacion", t_prep[0])
    reloj.sumar("pose", pose)
    reloj.sumar("seguimiento", t_track[0])
    reloj.sumar("decodificacion", decod)
    reloj.marcas.append({"etapa": "preproceso_total", "inicio_epoch": time.time() - total,
                         "dur_s": total})
    return {"npz": res.npz_path, "cuadros": res.total_frames,
            "n_detecciones": res.n_detections}


class _EnvolturaModelo:
    """Deja envolver el callback del tracker despues del primer track(), que es cuando existe."""

    def __init__(self, modelo, al_crear_predictor) -> None:
        self._m = modelo
        self._hook = al_crear_predictor
        self._listo = False

    def track(self, *a, **k):
        r = self._m.track(*a, **k)
        if not self._listo:
            self._hook(self._m)
            self._listo = True
        return r

    def __getattr__(self, n):
        return getattr(self._m, n)


_cortes_original = None


def _envolver_cortes(acum: list):
    from boxtwin.preprocess import cuts

    global _cortes_original
    _cortes_original = cuts.detectar_cortes
    orig = _cortes_original

    def medido(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            acum[0] += time.perf_counter() - t0

    return medido


def _parchear_cortes(fn) -> None:
    from boxtwin.preprocess import cuts

    cuts.detectar_cortes = fn


def _desparchear_cortes() -> None:
    from boxtwin.preprocess import cuts

    if _cortes_original is not None:
        cuts.detectar_cortes = _cortes_original


# ---------------------------------------------------------------------------
# Etapa 4: identidad
# ---------------------------------------------------------------------------


def correr_identidad(video: Path, npz: Path, annot: Path, salida: Path, reloj: Reloj) -> Path:
    """
    Resuelve track_id a peleador y escribe la serie por brazo que consume el detector.

    OJO CON QUE ES ESTA ETAPA. Las asignaciones las decidio una persona en el anotador y
    viven en el annot.json; aca solo se aplican. El costo medido es el de aplicarlas, no el
    de inferirlas, porque hoy no hay nada que las infiera.
    """
    from boxtwin.core.annotations import load as load_doc
    from boxtwin.core.export import ExportContext
    from boxtwin.core.export.sequence import exportar
    from boxtwin.core.identity import IdentityResolver
    from boxtwin.core.posecache import PoseCache

    with reloj.etapa("identidad"):
        cache = PoseCache.open(npz)
        doc, _ = load_doc(annot)
        ctx = ExportContext(
            doc=doc, cache=cache, resolver=IdentityResolver(doc, cache),
            video_path=video, out_dir=salida,
            opciones={"label_space": "side", "classes": 14, "channels": "per-arm",
                      "keypoints": "coco17"},
        )
        res = exportar(ctx)
    return next(p for p in res.archivos if p.suffix == ".npz")


# ---------------------------------------------------------------------------
# Etapa 5: deteccion de golpes
# ---------------------------------------------------------------------------


def correr_deteccion(seq_npz: Path, ensamble: Path, umbral: float, reloj: Reloj) -> list[dict]:
    """
    Features, ensamble y decodificacion a segmentos.

    El ensamble se CARGA de disco, no se entrena. Entrenar es costo de desarrollo y ocurre
    una vez; meterlo en el costo por hora de video inflaria S-19 con trabajo que en
    produccion no se repite.
    """
    import numpy as np
    import torch

    from boxtwin_detector.dataset import construir
    from boxtwin_detector.decodificacion import decodificar_score
    from boxtwin_detector.ensamble import cargar, probabilidad

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with reloj.etapa("preparacion"):
        ens = cargar(ensamble, device)

    segmentos: list[dict] = []
    with reloj.etapa("deteccion"):
        fuente = construir(seq_npz)
        probs = probabilidad(ens, fuente, device)
        a, b = fuente.conteos.get("cobertura", [0, fuente.T - 1])
        valido = np.zeros(fuente.T, bool)
        valido[a : b + 1] = True
        for c, carril in enumerate(fuente.carriles):
            for s in decodificar_score(probs[c], umbral, valido=valido):
                peleador, brazo = carril.split("-")
                segmentos.append({
                    "carril": carril, "fighter": f"fighter_{peleador}", "side": brazo,
                    "inicio": int(s.inicio), "fin": int(s.fin),
                    "prob": round(float(probs[c][s.inicio : s.fin + 1].max()), 4),
                })
    return segmentos


# ---------------------------------------------------------------------------
# Etapa 6: clasificacion de familia (otro entorno, otro proceso)
# ---------------------------------------------------------------------------


def correr_clasificacion(segmentos: list[dict], seq_npz: Path, geometria: dict,
                         config: Path, checkpoint: Path, trabajo: Path, reloj: Reloj) -> int:
    """
    Clasifica cada segmento con PoseConv3D, en el entorno boxtwin_mmaction.

    Va por subproceso porque el clasificador vive con torch 2.1 y OpenMMLab pinneados y el
    resto de la cadena con torch 2.6. El tiempo del subproceso es tiempo de pared de la
    cadena y cuenta entero, incluida la carga del modelo.
    """
    if not segmentos:
        return 0
    entrada = trabajo / "segmentos_para_clasificar.json"
    entrada.write_text(json.dumps({
        "segmentos": segmentos, "secuencia": str(seq_npz), **geometria,
    }), encoding="utf-8")
    script = Path(__file__).resolve().parent / "medicion_clasifica.py"

    with reloj.etapa("clasificacion"):
        r = subprocess.run(
            [str(PY_MMACTION), str(script), str(entrada), "--config", str(config),
             "--checkpoint", str(checkpoint), "--out", str(trabajo / "clasificado.json")],
            capture_output=True, text=True,
        )
    if r.returncode != 0:
        raise RuntimeError(f"la clasificacion fallo:\n{r.stderr[-2000:]}")
    return len(segmentos)


# ---------------------------------------------------------------------------
# Verificacion: el cache nuevo tiene que ser el mismo que el que ya existe
# ---------------------------------------------------------------------------


def cache_coincide(nuevo: Path, viejo: Path) -> str:
    """Compara track_id y keypoints. Devuelve 'si', 'no' o el motivo de no haber podido."""
    if not viejo.is_file():
        return "sin-referencia"
    try:
        import numpy as np

        a, b = np.load(nuevo), np.load(viejo)
        for clave in ("track_id", "keypoints"):
            if clave not in a or clave not in b:
                return "sin-referencia"
            x, y = a[clave], b[clave]
            # con --max-cuadros el cache nuevo es un prefijo del de referencia
            n = min(len(x), len(y))
            if x.shape[1:] != y.shape[1:]:
                return "no"
            x, y = x[:n], y[:n]
            if not np.array_equal(x, y) and not np.allclose(x, y, atol=1e-4):
                return "no"
        return "si" if len(a["track_id"]) == len(b["track_id"]) else "si-prefijo"
    except Exception as e:  # noqa: BLE001
        return f"error:{type(e).__name__}"


# ---------------------------------------------------------------------------


def duracion_video(video: Path) -> tuple[float, float]:
    """Duracion y fps del contenedor, con ffprobe. No se le cree al nombre del archivo."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, check=True,
    )
    partes = [x for x in r.stdout.split() if x]
    fps_txt = next(x for x in partes if "/" in x)
    n, d = fps_txt.split("/")
    fps = float(n) / float(d)
    dur = float(next(x for x in partes if "/" not in x))
    return dur, fps


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("USO")[0])
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--annot", type=Path, help="por defecto, el del proyecto del video")
    p.add_argument("--corridas", type=int, default=3)
    p.add_argument("--max-cuadros", type=int, default=None,
                   help="procesar solo los primeros N cuadros. Exige que la anotacion "
                        "declare ese mismo total_frames; ver la nota al rechazarlo")
    p.add_argument("--out", type=Path, default=Path("mediciones/corridas.csv"))
    p.add_argument("--trabajo", type=Path, default=Path("/tmp/boxtwin_medicion"))
    p.add_argument("--modelo-pose", type=Path, default=RAIZ / "yolov8l-pose.pt")
    p.add_argument("--ensamble", type=Path, default=RAIZ / "modelos" / "detector_7fuentes.ens.pt")
    p.add_argument("--config", type=Path, default=RAIZ / "modelos" / "poseC3D_7fuentes.py")
    p.add_argument("--checkpoint", type=Path, default=RAIZ / "modelos" / "poseC3D_7fuentes.pth")
    p.add_argument("--umbral", type=float, default=0.80)
    p.add_argument("--commit", default="")
    a = p.parse_args()

    video = a.video.resolve()
    if not video.is_file():
        print(f"error: no existe {video}", file=sys.stderr)
        return 1
    proyecto = video.parent.parent
    annot = a.annot or (proyecto / "annotations" / f"{video.stem}.annot.json")
    if not annot.is_file():
        print(f"error: no hay anotacion en {annot}", file=sys.stderr)
        return 1

    faltan = [str(x) for x in (a.modelo_pose, a.ensamble, a.config, a.checkpoint)
              if not Path(x).is_file()]
    if faltan:
        print("error: faltan checkpoints:\n  " + "\n  ".join(faltan), file=sys.stderr)
        return 1

    # --max-cuadros solo sirve si la anotacion acompana. El export dimensiona sus arreglos
    # con doc.video.total_frames y despues pide cada cuadro al cache: si el cache tiene
    # menos, la etapa de identidad se cae con IndexError a mitad de la corrida. Se rechaza
    # antes de gastar el pase de pose.
    if a.max_cuadros:
        _tf = json.loads(annot.read_text(encoding="utf-8"))["video"]["total_frames"]
        if _tf != a.max_cuadros:
            print(
                f"error: --max-cuadros {a.max_cuadros} pero la anotacion declara {_tf} "
                "cuadros.\n"
                "  El export dimensiona con la anotacion y le pide al cache cuadros que no\n"
                "  tiene. Para medir un tramo de un video largo, recortar el archivo con\n"
                "  ffmpeg y recortar la anotacion en consecuencia, que es lo que se hizo\n"
                "  con pacquiao (protocolo.md, desvio D-8). Asi ademas la deteccion de\n"
                "  cortes de plano, que corre por ffmpeg sobre el archivo entero, se acota\n"
                "  al tramo y el ratio no sale inflado.",
                file=sys.stderr,
            )
            return 1

    dur, fps = duracion_video(video)
    commit = a.commit or subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=RAIZ,
    ).stdout.strip()[:12]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not a.out.is_file()
    fh = a.out.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=COLUMNAS)
    if nuevo:
        w.writeheader()

    cache_ref = proyecto / "cache" / f"{video.stem}.pose.npz"
    meta_ref = proyecto / "cache" / f"{video.stem}.meta.json"
    cfg_base, origen_cfg = config_del_proyecto(meta_ref, a.modelo_pose)
    print(f"configuracion de inferencia ({origen_cfg}): half={cfg_base.half} "
          f"imgsz={cfg_base.imgsz} detect_cuts={cfg_base.detect_cuts}")
    _doc = json.loads(annot.read_text(encoding="utf-8"))
    geometria = {"alto": _doc["video"]["height"], "ancho": _doc["video"]["width"],
                 "fps_video": fps}

    for i in range(1, a.corridas + 1):
        trabajo = a.trabajo / f"{video.stem}_c{i}"
        shutil.rmtree(trabajo, ignore_errors=True)
        trabajo.mkdir(parents=True, exist_ok=True)

        reloj = Reloj()
        notas = ""
        estado = "caliente" if i > 1 else "frio"
        t0 = time.perf_counter()
        try:
            pre = correr_preproceso(video, trabajo, reloj, cfg_base, a.max_cuadros)
            seq = correr_identidad(video, pre["npz"], annot, trabajo / "export", reloj)
            segs = correr_deteccion(seq, a.ensamble, a.umbral, reloj)
            n_seg = correr_clasificacion(segs, seq, geometria, a.config, a.checkpoint,
                                         trabajo, reloj)
            total = time.perf_counter() - t0
            ident = cache_coincide(pre["npz"], cache_ref)
            # el ratio se calcula contra el video efectivamente procesado
            dur_ef = pre["cuadros"] / fps if fps else dur
        except Exception as e:  # noqa: BLE001
            total = time.perf_counter() - t0
            print(f"corrida {i} FALLO: {type(e).__name__}: {e}", file=sys.stderr)
            w.writerow({"ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), "video": video.name,
                        "duracion_video_s": round(dur, 3), "cuadros": "", "fps_video": round(fps, 4),
                        "corrida": i, "estado": estado + "-fallo", "commit": commit,
                        "tiempo_total_s": round(total, 3),
                        **{f"t_{e}_s": round(reloj.acum.get(e, 0.0), 3) for e in ETAPAS},
                        "ratio_computo": "", "cache_identico": "", "n_segmentos": "",
                        "n_detecciones": "", "half": cfg_base.half, "imgsz": cfg_base.imgsz,
                        "detect_cuts": cfg_base.detect_cuts,
                        "notas": f"{type(e).__name__}: {e}"[:300]})
            fh.flush()
            continue

        (trabajo / "marcas.json").write_text(json.dumps(reloj.marcas, indent=2), encoding="utf-8")
        fila = {
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), "video": video.name,
            "duracion_video_s": round(dur_ef, 3), "cuadros": pre["cuadros"],
            "fps_video": round(fps, 4), "corrida": i, "estado": estado, "commit": commit,
            "tiempo_total_s": round(total, 3),
            **{f"t_{e}_s": round(reloj.acum.get(e, 0.0), 3) for e in ETAPAS},
            "ratio_computo": round(total / dur_ef, 4) if dur_ef else "",
            "cache_identico": ident, "n_segmentos": n_seg,
            "n_detecciones": pre["n_detecciones"],
            "half": cfg_base.half, "imgsz": cfg_base.imgsz,
            "detect_cuts": cfg_base.detect_cuts, "notas": notas,
        }
        w.writerow(fila)
        fh.flush()
        suma = sum(reloj.acum.get(e, 0.0) for e in ETAPAS)
        print(f"[{video.name}] corrida {i}/{a.corridas} ({estado}): {total:.1f} s  "
              f"ratio {total/dur_ef:.2f}x  etapas suman {suma:.1f} s  cache={ident}  "
              f"segmentos={n_seg}")

    fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
