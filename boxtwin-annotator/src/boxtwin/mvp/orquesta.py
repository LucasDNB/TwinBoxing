"""
BoxTwin - De un video crudo a una Fight-Card, en dos tiempos.

POR QUE EXISTE
  Hasta ahora no habia forma de correr el sistema sobre un video sin armarle antes un
  proyecto de anotacion: el detector entra por `Fuente`, que se construye desde un export, y
  el export sale de un `annot.json` que alguien tiene que haber tocado. Todo el circuito
  medido pasa por ahi. Para un producto eso no sirve, y para la tesis tampoco: el numero de
  punta a punta sobre una sesion nueva no se puede tomar si tomarlo requiere anotarla.

  Va en dos comandos y no en uno por una razon que no es de ingenieria. Entre la pose y el
  detector hay una pregunta que el sistema contesta bien el 82,4% de las veces y una persona
  el 99,1%: cual de los dos cuerpos es cual. Cuesta dos clicks y es la unica intervencion
  humana del producto.

QUE HACE
  procesar   pose, tracking, evidencia de guante y propuesta de los dos candidatos.
  completar  identidad por color desde las semillas, detector, guardia y Fight-Card.

  Las dos etapas dejan todo en el directorio de la sesion, que ademas es un proyecto de
  anotacion valido: se puede abrir con el anotador y ver exactamente lo que vio el sistema.
  Eso no es un lujo, es como se audita un resultado que salio sin que nadie mirara.

USO
  boxtwin-annotator procesar video.mp4 --out sesion/ --modelo-guantes guantes-v2.pt
  boxtwin-annotator completar sesion/ --semilla-a 3 --semilla-b 7 --detector ens.pt
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from boxtwin.core.identidad_auto import (
    AsignarIdentidadAuto,
    ConfigIdentidadAuto,
    analizar,
    cargar_evidencia,
    guardar_evidencia,
    proponer,
)
from boxtwin.core.posecache import PoseCache
from boxtwin.core.project import ProjectPaths, cargar_o_crear, project_paths
from boxtwin.mvp.candidatos import elegir, recortar
from boxtwin.mvp.fightcard import construir
from boxtwin.mvp.guardia import ConfigGuardia, medir
from boxtwin.mvp.sesion import Sesion

__all__ = [
    "CORRECCIONES",
    "EVIDENCIA",
    "FIGHTCARD",
    "POSES",
    "SEGMENTOS",
    "TIPOS",
    "aplicar_tipos_a_sesion",
    "completar",
    "leer_fightcard",
    "procesar",
    "registrar_correccion",
    "series_por_peleador",
]

EVIDENCIA = "evidencia.json"
POSES = "poses.npz"
SEGMENTOS = "segmentos.json"
FIGHTCARD = "fightcard.json"
CANDIDATOS = "candidatos"


class SesionEnEstadoEquivocado(RuntimeError):
    """Se pidio una etapa que no corresponde al estado de la sesion."""


# ---------------------------------------------------------------------------
# Etapa 1
# ---------------------------------------------------------------------------


def procesar(
    video: Path,
    salida: Path,
    modelo_guantes: Path,
    round_s: float | None = None,
    descanso_s: float = 60.0,
    cfg_identidad: ConfigIdentidadAuto | None = None,
    device: str = "0",
    modelo_pose: str = "yolov8l-pose.pt",
    imgsz: int = 640,
    rehacer: bool = False,
    progreso=None,
) -> Sesion:
    """
    Pose, tracking y evidencia de identidad. Deja la sesion esperando la siembra.

    Es reanudable en el punto que importa, que es el caro: el preproceso ya lo es por
    shards, y la evidencia de guante se guarda apenas termina de recorrer el video. Volver a
    correr el comando sobre una sesion que ya la tiene no vuelve a pasar el detector de
    guantes por 30 minutos de video.
    """
    from boxtwin.core.deteccion_guantes import DetectorGuantes
    from boxtwin.preprocess.runner import PreprocessConfig, default_tracker_path, preprocess

    salida = Path(salida)
    cfg_identidad = cfg_identidad or ConfigIdentidadAuto()
    video_en_sesion = _ubicar_video(Path(video), salida)
    paths = project_paths(video_en_sesion)

    # -- pose y tracking ----------------------------------------------------
    t0 = time.perf_counter()
    cfg_pre = PreprocessConfig(
        model=modelo_pose, imgsz=imgsz, device=device, tracker=default_tracker_path(),
    )
    r = preprocess(video_en_sesion, salida, cfg_pre, on_progress=progreso)
    cache = PoseCache.open(paths.npz)
    doc, _ = cargar_o_crear(paths, cache.meta)

    ses = Sesion.nueva(
        salida,
        video={
            "nombre": video_en_sesion.name,
            "ruta": str(video_en_sesion),
            "sha256": doc.video.sha256,
            "fps": doc.video.fps,
            "total_frames": doc.video.total_frames,
            "duracion_s": doc.video.duration_s,
            "ancho": doc.video.width,
            "alto": doc.video.height,
        },
        round_s=round_s,
        descanso_s=descanso_s,
    )
    ses.anotar_etapa("preproceso", time.perf_counter() - t0, cuadros=r.total_frames)
    if r.suspected_vfr:
        ses.avisos.append(
            "el fps declarado no concuerda con el conteo real de cuadros: puede ser "
            "framerate variable, y en ese caso los segundos de cada evento se corren"
        )
    if r.seams:
        ses.avisos.append(
            f"{len(r.seams)} costura(s) de reanudacion del tracker: ningun track cruza esos "
            "cuadros, asi que la identidad se fragmenta ahi"
        )

    # -- evidencia de guante ------------------------------------------------
    ruta_ev = salida / EVIDENCIA
    t0 = time.perf_counter()
    if ruta_ev.is_file() and not rehacer:
        ev = cargar_evidencia(ruta_ev)
        ses.anotar_etapa("evidencia", 0.0, reutilizada=True, tracks=len(ev))
    else:
        detector = DetectorGuantes(str(modelo_guantes), device=None if device == "0" else device)
        ev = analizar(
            cache, video_en_sesion, detector, cfg_identidad,
            alto_imagen=doc.video.height, fps=doc.video.fps,
            total_frames=doc.video.total_frames, progreso=progreso,
        )
        guardar_evidencia(ev, ruta_ev)
        ses.anotar_etapa("evidencia", time.perf_counter() - t0, tracks=len(ev))

    # -- filtros y candidatos -----------------------------------------------
    prop = proponer(ev, cfg_identidad, doc.video.total_frames, doc.video.fps)
    # Se aplica la propuesta automatica aunque despues la siembra la reemplace: lo que
    # aporta son los descartes -publico, arbitro, entrenador- que la siembra no cambia y
    # que quedan escritos para que el anotador no los tenga que repetir si abre la sesion.
    if prop.roles:
        AsignarIdentidadAuto(prop, annotator="boxtwin-procesar").do(doc)
        from boxtwin.core.annotations import save as save_doc

        save_doc(doc, paths.annot)

    pareja, candidatos = elegir(
        ev, prop.diagnostico.get("nucleo_tracks", []),
        min_coexistencia=cfg_identidad.min_coexistencia,
    )
    recortar(video_en_sesion, cache, candidatos, salida / CANDIDATOS)

    ses.candidatos = [c.a_dict() for c in candidatos]
    ses.identidad = {
        "diagnostico_automatico": _jsonable(prop.diagnostico),
        "avisos_automaticos": list(prop.avisos),
        "pareja_sugerida": list(pareja) if pareja else None,
    }
    if pareja is None:
        ses.avisos.append(
            "ningun par de tracks coexiste lo suficiente como para afirmar que son dos "
            "personas distintas: hay que elegir las dos semillas a mano entre los candidatos"
        )
    ses.estado = "espera_siembra"
    ses.guardar()
    return ses


def _ubicar_video(video: Path, salida: Path) -> Path:
    """
    Deja el video adentro de la sesion sin copiarlo si se puede.

    El enlace simbolico alcanza y evita duplicar gigabytes por sesion. `project_paths` no
    resuelve symlinks a proposito, asi que el proyecto queda en la sesion y no en el
    directorio del video original, que es lo que uno esperaria y lo que evita escribir
    resultados al lado del material de otra persona.
    """
    video = Path(video).absolute()
    if not video.is_file():
        raise FileNotFoundError(f"no existe el video {video}")
    destino_dir = Path(salida) / "videos"
    if video.parent == destino_dir:
        return video
    destino_dir.mkdir(parents=True, exist_ok=True)
    destino = destino_dir / video.name
    if not destino.exists():
        try:
            destino.symlink_to(video)
        except OSError:
            import shutil

            shutil.copy2(video, destino)
    return destino


# ---------------------------------------------------------------------------
# Etapa 2
# ---------------------------------------------------------------------------


def completar(
    directorio: Path,
    semilla_a: int,
    semilla_b: int,
    detector: Path,
    umbral: float = 0.80,
    cfg_identidad: ConfigIdentidadAuto | None = None,
    cfg_guardia: ConfigGuardia | None = None,
    device: str | None = None,
    progreso=None,
) -> dict:
    """
    Identidad por color desde las semillas, detector, guardia y Fight-Card.

    Devuelve la Fight-Card ya escrita en disco. El tipo de golpe NO se resuelve aca: lo pone
    la etapa de clasificacion, que corre en el otro entorno conda y lee `segmentos.json`.
    """
    import torch

    from boxtwin.core.annotations import save as save_doc
    from boxtwin_detector.ensamble import cargar
    from boxtwin_detector.inferencia import carriles_de, detectar

    directorio = Path(directorio)
    ses = Sesion.cargar(directorio)
    if ses.estado not in ("espera_siembra", "listo", "completando"):
        raise SesionEnEstadoEquivocado(
            f"la sesion esta en '{ses.estado}' y la siembra se aplica sobre "
            "'espera_siembra'. Corre primero 'procesar'"
        )
    cfg_identidad = cfg_identidad or ConfigIdentidadAuto()
    paths = _paths_de(ses, directorio)
    cache = PoseCache.open(paths.npz)
    doc, _ = cargar_o_crear(paths, cache.meta)
    ev = cargar_evidencia(directorio / EVIDENCIA)

    ses.estado = "completando"
    ses.siembra = {"track_a": int(semilla_a), "track_b": int(semilla_b)}
    ses.guardar()

    # -- identidad con la siembra -------------------------------------------
    t0 = time.perf_counter()
    prop = proponer(
        ev, cfg_identidad, doc.video.total_frames, doc.video.fps,
        semillas=(int(semilla_a), int(semilla_b)),
    )
    if prop.diagnostico.get("semillas_en_conflicto"):
        ses.estado = "espera_siembra"
        ses.error = prop.avisos[-1] if prop.avisos else "semillas en conflicto"
        ses.guardar()
        raise ValueError(
            "las dos semillas caen del mismo lado de la coexistencia: "
            "o son la misma persona, o hay una coexistencia espuria. Elegi otro par"
        )
    AsignarIdentidadAuto(prop, annotator="boxtwin-completar", forzar=True).do(doc)
    save_doc(doc, paths.annot)
    ses.anotar_etapa("identidad", time.perf_counter() - t0,
                     asignados=prop.diagnostico.get("asignados_A", 0)
                     + prop.diagnostico.get("asignados_B", 0))

    # -- series por peleador -------------------------------------------------
    t0 = time.perf_counter()
    kp, sc, valid, cobertura = series_por_peleador(doc, cache)
    np.savez_compressed(directorio / POSES, keypoints=kp, kp_score=sc, valid=valid)
    ses.anotar_etapa("series", time.perf_counter() - t0)

    # -- detector ------------------------------------------------------------
    t0 = time.perf_counter()
    dev = torch.device(device) if device else None
    ens = cargar(Path(detector), dev)
    carriles = carriles_de(kp, sc, valid, fps_origen=doc.video.fps)
    golpes = detectar(ens, carriles, umbral=umbral, device=dev)
    ses.anotar_etapa("detector", time.perf_counter() - t0, golpes=len(golpes))

    # -- guardia -------------------------------------------------------------
    t0 = time.perf_counter()
    eventos = medir(kp, sc, golpes, fps=doc.video.fps, cfg=cfg_guardia or ConfigGuardia())
    ses.anotar_etapa("guardia", time.perf_counter() - t0)

    # -- salidas -------------------------------------------------------------
    info_detector = {
        "checkpoint": Path(detector).name,
        "semillas_del_ensamble": list(getattr(ens, "semillas", [])),
        "umbral": umbral,
    }
    (directorio / SEGMENTOS).write_text(
        json.dumps(
            {
                "version": "0.1",
                "video": ses.video,
                "detector": info_detector,
                "poses": POSES,
                "segmentos": [g.a_dict() for g in golpes],
            },
            indent=2, ensure_ascii=False,
        )
        + "\n"
    )

    cobertura["separacion_de_perfiles"] = prop.diagnostico.get("separacion_de_perfiles")
    cobertura["piso_de_separacion"] = cfg_identidad.separacion_minima
    cobertura["tracks_sin_asignar"] = len(prop.sin_asignar)
    ses.identidad.update({"diagnostico_con_siembra": _jsonable(prop.diagnostico),
                          "avisos_con_siembra": list(prop.avisos)})
    for a in prop.avisos:
        if a not in ses.avisos:
            ses.avisos.append(a)

    fc = construir(ses, golpes, eventos, cobertura, detector=info_detector)
    (directorio / FIGHTCARD).write_text(json.dumps(fc, indent=2, ensure_ascii=False) + "\n")

    ses.estado = "listo"
    ses.error = None
    ses.guardar()
    return fc


def _paths_de(ses: Sesion, directorio: Path) -> ProjectPaths:
    ruta = ses.video.get("ruta")
    video = Path(ruta) if ruta else (directorio / "videos" / str(ses.video.get("nombre")))
    if not video.exists():
        # La sesion se pudo haber movido de maquina, que es exactamente lo que pasa cuando
        # el worker corre adentro de un contenedor y el directorio viene montado.
        video = directorio / "videos" / Path(str(ses.video.get("nombre"))).name
    return project_paths(video)


def series_por_peleador(doc, cache) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    De tracks a dos series de keypoints, una por peleador, y cuanto se cubrio de cada uno.

    Es la misma resolucion que usa el export `sequence`, sin las etiquetas: un cuadro sin
    identidad resuelta no es un cuadro sin golpe, es un cuadro sin dato, y sale en la
    mascara para que el detector no pueda marcar nada ahi.

    La cobertura se reporta porque RF5 lo pide: cuando el sistema se abstiene, el usuario
    tiene que ver cuanto tiempo quedo sin mirar, no un conteo que parece completo.
    """
    from boxtwin.core.identity import IdentityResolver
    from boxtwin.core.types import FighterId

    T = int(doc.video.total_frames)
    n_kp = 17
    kp = np.zeros((2, T, n_kp, 2), np.float32)
    sc = np.zeros((2, T, n_kp), np.float32)
    valid = np.zeros((2, T), bool)

    resolver = IdentityResolver(doc, cache)
    peleadores = [FighterId.A, FighterId.B]
    cuadros_con_sin_asignar = 0
    for f in range(T):
        porf = resolver.by_fighter(f)
        for p, fid in enumerate(peleadores):
            pose = porf.get(fid)
            if pose is None:
                continue
            valid[p, f] = pose.reliable
            kp[p, f] = pose.keypoints[:n_kp]
            sc[p, f] = pose.kp_score[:n_kp]
        dets = cache.detections(f)
        if len(dets) and any(
            resolver.role_of(f, int(dets.track_id[i])) is None for i in range(len(dets))
        ):
            cuadros_con_sin_asignar += 1

    cobertura = {
        "cobertura_A": round(float(valid[0].mean()), 4),
        "cobertura_B": round(float(valid[1].mean()), 4),
        "sin_asignar": round(cuadros_con_sin_asignar / T, 4) if T else 0.0,
        "cuadros": T,
    }
    return kp, sc, valid, cobertura


def _jsonable(d: dict) -> dict:
    """El diagnostico trae tuplas y claves int, que json convierte a string sin avisar."""
    salida = {}
    for k, v in d.items():
        if isinstance(v, dict):
            salida[k] = {str(kk): list(vv) if isinstance(vv, tuple) else vv
                         for kk, vv in v.items()}
        elif isinstance(v, tuple):
            salida[k] = list(v)
        else:
            salida[k] = v
    return salida


# ---------------------------------------------------------------------------
# Tipos y correcciones
# ---------------------------------------------------------------------------

TIPOS = "tipos.json"
CORRECCIONES = "correcciones.jsonl"


def leer_fightcard(directorio: Path) -> dict:
    ruta = Path(directorio) / FIGHTCARD
    if not ruta.is_file():
        raise FileNotFoundError(
            f"no hay Fight-Card en {directorio}: corre primero 'completar'"
        )
    return json.loads(ruta.read_text())


def escribir_fightcard(directorio: Path, fc: dict) -> Path:
    ruta = Path(directorio) / FIGHTCARD
    tmp = ruta.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(fc, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(ruta)
    return ruta


def leer_correcciones(directorio: Path) -> list[dict]:
    ruta = Path(directorio) / CORRECCIONES
    if not ruta.is_file():
        return []
    return [json.loads(l) for l in ruta.read_text().splitlines() if l.strip()]


def aplicar_tipos_a_sesion(directorio: Path, exactitud: float | None = None) -> dict:
    """
    Pega tipos.json sobre la Fight-Card, y vuelve a aplicar las correcciones encima.

    El orden importa y es el unico que tiene sentido: primero lo que dijo el modelo nuevo,
    despues lo que dijo la persona. Al reves, publicar un checkpoint le borraria al
    entrenador todas las correcciones que hizo, que son justo las etiquetas mas caras que
    tiene el proyecto.
    """
    from boxtwin.mvp.fightcard import aplicar_correcciones, aplicar_tipos

    directorio = Path(directorio)
    d = json.loads((directorio / TIPOS).read_text())
    fc = leer_fightcard(directorio)
    fc = aplicar_tipos(fc, d.get("tipos", {}), d.get("checkpoint", ""), exactitud)
    fc = aplicar_correcciones(fc, leer_correcciones(directorio))
    escribir_fightcard(directorio, fc)
    return fc


def registrar_correccion(
    directorio: Path, golpe: str, tipo: str, por: str | None = None
) -> dict:
    """
    Guarda la correccion del entrenador y la refleja en la Fight-Card. F14 y RF10.

    Se appendea a un jsonl y no se pisa nada: cada correccion es una etiqueta nueva sobre
    material que el modelo no vio, que es exactamente lo que le falta para generalizar. Un
    registro que se sobrescribe deja de ser un dataset.
    """
    from boxtwin.mvp.fightcard import aplicar_correcciones, correccion

    directorio = Path(directorio)
    fc = leer_fightcard(directorio)
    c = correccion(golpe, tipo, fc, por=por)
    with (directorio / CORRECCIONES).open("a") as f:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
    fc = aplicar_correcciones(fc, leer_correcciones(directorio))
    escribir_fightcard(directorio, fc)
    return fc
