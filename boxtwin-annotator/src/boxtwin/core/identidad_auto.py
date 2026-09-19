"""
BoxTwin - Asignacion automatica de identidad a partir de los guantes.

POR QUE EXISTE
  Asignar el rol de cada track es la ultima pieza enteramente manual del sistema y se lleva
  el 73% del tiempo de anotacion medido. No es un problema de dos decisiones: el tracker
  fragmenta a cada peleador en decenas de ids -38 para A y 36 para B en sparring-3-rounds- y
  son 390 relevos sobre las siete fuentes. La regla que hoy los propone, IoU contra la ultima
  caja, acierta el 45,1%.

  Este modulo los resuelve con tres filtros medidos, en este orden, y cada uno saca algo que
  el anterior no puede:

  1. ALTURA. Saca al publico, que esta lejos y es chico. En 04-sparring lleva 375 tracks a
     42. Lo que NO saca es la gente a distancia de ring: arbitro, entrenador, cronometrista,
     la otra pareja del gimnasio, que quedan entre 20 y 43 por video.
  2. GUANTE. Saca a esos, porque el arbitro es el unico adentro del ring sin guantes de
     boxeo. Medido sobre seis fuentes, los tracks de peleador dan fraccion de guante entre
     0,59 y 0,96 contra 0,22 a 0,33 del resto, y las medianas no se tocan en ninguna. En el
     video amateur el arbitro usa guantes de LATEX y aun asi se lleva la fraccion mas baja
     del video, 0,197: el detector no los confunde.
  3. COLOR DEL GUANTE. Decide A contra B. Sobre 8150 guantes, 87,7% por guante suelto y 112
     de 113 tracks por voto de mayoria, con el unico fallo siendo un empate en la fuente
     donde los dos llevan el mismo naranja.

  Los dos primeros son complementarios y no redundantes: combinados dejan un unico track
  residual en las seis fuentes.

QUE HACE
  Mide la evidencia por track, siembra los dos perfiles de color con los primeros segundos de
  video, y propone un rol para cada track. Lo que NO puede decidir lo deja SIN ASIGNAR, que
  no es lo mismo que adivinarlo: un track sin rol sale en el color de sin asignar del overlay
  y el anotador lo ve, mientras que un rol equivocado se dibuja sobre un cuerpo y se ve
  perfecto. Es la asimetria que manda en este modulo.

USO
  ev = analizar(cache, video, detector, cfg)
  prop = proponer(ev, cfg, fps=30.0, alto_imagen=1080)
  pila.do(AsignarIdentidadAuto(prop, annotator="lucas"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from boxtwin.core.annotations import new_id
from boxtwin.core.deteccion_guantes import (
    ACROMATICO,
    Color,
    DetectorGuantes,
    distancia_color,
    perfil_de,
)
from boxtwin.core.identity_ops import _SnapshotCommand, _origen
from boxtwin.core.posecache import PoseCache
from boxtwin.core.schema import AnnotationDoc, Assignment, Identity
from boxtwin.core.types import AssignmentOp, TrackRole

__all__ = [
    "ConfigIdentidadAuto", "EvidenciaTrack", "Propuesta", "analizar", "proponer",
    "AsignarIdentidadAuto",
]


@dataclass(frozen=True)
class ConfigIdentidadAuto:
    """
    Parametros de la asignacion.

    `fraccion_altura` es relativa al track mas alto del video y no absoluta: un plano cerrado
    de gimnasio y una transmision de television no comparten escala, y un umbral fijo que
    anda en uno deja pasar a todo el publico del otro.

    `umbral_guante` cae en el hueco medido entre las dos poblaciones: peleadores de 0,59 para
    arriba, el resto de 0,33 para abajo. 0,45 esta en el medio y lejos de los dos bordes.

    La ventana de siembra ARRANCA en `segundos_semilla` y se extiende sola hasta
    `segundos_maximos` mientras los dos perfiles no se separen. Cinco segundos alcanzan
    cuando los guantes son de colores distintos, y no alcanzan cuando no lo son: sobre
    01-sparring, con cinco segundos los perfiles salieron a tono 7,3 y 172,9 -separacion
    0,449- mientras que con el video entero daban 3,5 y 9,1. Es poca evidencia, no un color
    distinto.

    `separacion_minima` es un piso duro, no un aviso. Con 0,449 el reparto salio 13 tracks a
    A contra 1 a B, que es una asignacion equivocada con cara de asignacion. Preferimos no
    asignar: los filtros de descarte igual sacan 122 de 137 tracks, y lo que queda lo
    resuelve el anotador viendo cual es cual.
    """

    segundos_semilla: float = 5.0
    segundos_maximos: float = 60.0   # hasta donde se extiende la ventana si no separan
    separacion_minima: float = 0.55  # debajo de esto no se asigna A ni B
    paso: int = 5                    # se evalua un cuadro de cada N
    fraccion_altura: float = 0.55
    umbral_guante: float = 0.45
    min_recortes: int = 15           # menos que esto y la fraccion no significa nada
    min_guantes_voto: int = 10       # menos que esto y el voto se deja sin asignar
    min_coexistencia: int = 3        # cuadros en que las dos semillas tienen que coincidir
    iteraciones: int = 4             # vueltas de reestimar perfiles y volver a votar
    iou_max_aislado: float = 0.02    # para muestrear color sin cuerpo del rival adentro
    # Con quien hay que estar aislado: solo con gente de tamano comparable. Un espectador
    # lejano que se superpone con la caja no contamina el parche del guante, pero exigir
    # aislamiento de TODOS descalificaba casi todos los cuadros en videos con publico. Sobre
    # 02-sparring, que tiene 400 tracks, eso dejaba 45 de 49 candidatos sin evidencia
    # suficiente para votar. El riesgo real es el otro peleador, que es del mismo tamano.
    fraccion_alto_relevante: float = 0.6


@dataclass
class EvidenciaTrack:
    """Lo que se midio de un track. No decide nada: solo junta."""

    track_id: int
    primer_frame: int
    ultimo_frame: int
    alto_max: float = 0.0            # fraccion del alto de la imagen
    recortes: int = 0
    con_guante: int = 0
    # Cada medicion viaja con su cuadro. Asi la ventana de siembra se elige DESPUES de haber
    # recorrido el video, y extenderla no cuesta una segunda pasada del detector.
    colores: list[tuple[int, Color]] = field(default_factory=list)
    frames_con_guante: list[int] = field(default_factory=list)
    xs: list[tuple[int, float]] = field(default_factory=list)

    @property
    def fraccion_guante(self) -> float:
        return self.con_guante / self.recortes if self.recortes else 0.0

    def colores_hasta(self, frame: int) -> list[Color]:
        return [c for f, c in self.colores if f < frame]

    def guantes_hasta(self, frame: int) -> int:
        return sum(1 for f in self.frames_con_guante if f < frame)

    def x_hasta(self, frame: int) -> float:
        xs = [x for f, x in self.xs if f < frame]
        return float(np.median(xs)) if xs else 0.0


@dataclass
class Propuesta:
    """Lo que la asignacion automatica decidio, y por que."""

    roles: dict[int, TrackRole]              # track -> rol propuesto
    rangos: dict[int, tuple[int, int]]       # track -> [primer, ultimo+1)
    sin_asignar: list[int] = field(default_factory=list)
    perfiles: dict[str, Color] = field(default_factory=dict)
    semillas: dict[str, int] = field(default_factory=dict)
    avisos: list[str] = field(default_factory=list)
    diagnostico: dict = field(default_factory=dict)


def analizar(
    cache: PoseCache,
    video: Path,
    detector: DetectorGuantes,
    cfg: ConfigIdentidadAuto,
    alto_imagen: int,
    fps: float,
    total_frames: int | None = None,
    progreso=None,
) -> dict[int, EvidenciaTrack]:
    """
    Recorre el video y mide, por track, altura, fraccion de guante y colores.

    El color solo se muestrea en cuadros donde la deteccion esta AISLADA de las demas: en
    clinch las cajas se superponen casi por completo y el parche tendria guante del rival
    adentro. Sobre las fuentes medidas eso descarta entre el 35% y el 52% de los cuadros, y
    es lo que hace que el voto por color sea confiable.
    """
    import cv2

    fin = min(total_frames or len(cache), len(cache))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"no se pudo abrir el video {video}")

    ev: dict[int, EvidenciaTrack] = {}
    try:
        for f in range(0, fin, cfg.paso):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                break
            dets = cache.detections(f)
            cajas = [np.asarray(dets.bbox[i], dtype=np.float64) for i in range(len(dets))]

            for i in range(len(dets)):
                tid = int(dets.track_id[i])
                x0, y0, x1, y1 = cajas[i]
                e = ev.get(tid)
                if e is None:
                    e = ev[tid] = EvidenciaTrack(tid, f, f)
                e.primer_frame = min(e.primer_frame, f)
                e.ultimo_frame = max(e.ultimo_frame, f)
                e.alto_max = max(e.alto_max, (y1 - y0) / alto_imagen)
                if (y1 - y0) < 16 or (x1 - x0) < 8:
                    continue

                e.recortes += 1
                con_color = detector.detectar_con_color(frame, cajas[i])
                if con_color:
                    e.con_guante += 1

                # El color solo cuenta si esta aislado de la gente de tamano comparable.
                alto_i = y1 - y0
                aislado = True
                for j in range(len(dets)):
                    if j == i:
                        continue
                    alto_j = cajas[j][3] - cajas[j][1]
                    if alto_j < cfg.fraccion_alto_relevante * alto_i:
                        continue
                    if _iou(cajas[i], cajas[j]) > cfg.iou_max_aislado:
                        aislado = False
                        break
                if aislado:
                    for _, color in con_color:
                        e.colores.append((f, color))
                if con_color:
                    e.frames_con_guante.append(f)
                e.xs.append((f, (x0 + x1) / 2.0))

            if progreso is not None and f % (cfg.paso * 40) == 0:
                progreso(f, fin)
    finally:
        cap.release()
    return ev


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Propuesta


def _mejor_par(
    evidencia: dict[int, EvidenciaTrack],
    candidatos: set[int],
    limite: int,
    cfg: ConfigIdentidadAuto,
) -> tuple[int, int, set[int]] | None:
    """
    Los dos tracks que mas veces aparecen A LA VEZ y separados, y los cuadros que comparten.

    La primera version tomaba los dos tracks con mas guantes en la ventana, y eso no dice que
    sean dos personas: un peleador fragmentado en dos ids da dos semillas de la MISMA
    persona, y ahi el perfil de A y el de B describen al mismo tipo. El voto reparte entonces
    casi al azar. Medido contra la anotacion, las tres fuentes donde las semillas cayeron
    sobre el mismo peleador dieron 57,9%, 53,8% y 89,5%, contra 100% donde cayeron bien.

    Coexistir resuelve eso sin ninguna heuristica de apariencia: dos fragmentos del mismo
    cuerpo casi nunca estan en el mismo cuadro, y los dos peleadores si lo estan. Se cuentan
    solo los cuadros donde los DOS tienen color medido, que por construccion son cuadros
    donde cada uno estaba aislado de la gente de su tamano; si los dos lo estaban en el mismo
    cuadro, entonces tampoco se superponian entre si.
    """
    frames = {
        t: {f for f, _ in evidencia[t].colores if f < limite}
        for t in candidatos
    }
    frames = {t: fs for t, fs in frames.items() if fs}
    if len(frames) < 2:
        return None
    orden = sorted(frames)
    mejor, mejor_n = None, 0
    for i, t1 in enumerate(orden):
        for t2 in orden[i + 1:]:
            n = len(frames[t1] & frames[t2])
            if n > mejor_n:
                mejor, mejor_n = (t1, t2), n
    if mejor is None or mejor_n < cfg.min_coexistencia:
        return None
    return mejor[0], mejor[1], frames[mejor[0]] & frames[mejor[1]]


def proponer(
    evidencia: dict[int, EvidenciaTrack],
    cfg: ConfigIdentidadAuto,
    total_frames: int,
    fps: float,
) -> Propuesta:
    """
    Decide un rol por track a partir de la evidencia.

    El orden importa y cada filtro saca lo que el anterior no puede. Lo que no se puede
    decidir queda SIN ASIGNAR: un track sin rol se ve en el overlay, un rol equivocado no.
    """
    prop = Propuesta(roles={}, rangos={})
    if not evidencia:
        prop.avisos.append("el cache no tiene ningun track")
        return prop

    for tid, e in evidencia.items():
        prop.rangos[tid] = (e.primer_frame, min(e.ultimo_frame + 1, total_frames))

    # -- 1. altura ----------------------------------------------------------
    alto_maximo = max(e.alto_max for e in evidencia.values())
    umbral_alto = cfg.fraccion_altura * alto_maximo
    pasan_altura = {t for t, e in evidencia.items() if e.alto_max >= umbral_alto}

    # -- 2. guante ----------------------------------------------------------
    candidatos = {
        t for t in pasan_altura
        if evidencia[t].recortes >= cfg.min_recortes
        and evidencia[t].fraccion_guante >= cfg.umbral_guante
    }
    for t in evidencia:
        if t not in candidatos:
            prop.roles[t] = TrackRole.IGNORE

    prop.diagnostico = {
        "tracks": len(evidencia),
        "umbral_altura": round(umbral_alto, 4),
        "pasan_altura": len(pasan_altura),
        "candidatos_tras_guante": len(candidatos),
    }
    if not candidatos:
        prop.avisos.append(
            "ningun track pasa los filtros de altura y guante; no hay a quien asignar"
        )
        return prop

    # -- 3. semilla, con la ventana extendiendose hasta que los perfiles separen ------
    # Se arranca corto y se extiende. Cinco segundos alcanzan cuando los guantes son de
    # colores distintos; cuando no lo son, el problema es que hay poca evidencia y no que el
    # color no sirva, asi que se le da mas video antes de rendirse. Extender no cuesta otra
    # pasada del detector: cada medicion ya viaja con su cuadro.
    segundos_video = total_frames / fps if fps > 0 else 0.0
    ventanas: list[float] = []
    seg = cfg.segundos_semilla
    while seg < cfg.segundos_maximos:
        ventanas.append(seg)
        seg *= 2
    ventanas.append(min(cfg.segundos_maximos, segundos_video) if segundos_video else
                    cfg.segundos_maximos)
    ventanas = sorted({round(v, 3) for v in ventanas if v > 0})

    intentos: list[tuple[float, float]] = []
    elegida = None
    mejor = None
    for seg in ventanas:
        limite = min(int(seg * fps), total_frames) if fps > 0 else total_frames
        par = _mejor_par(evidencia, candidatos, limite, cfg)
        if par is None:
            continue
        a, b, comunes = par
        # Quien es A: el que arranca mas a la izquierda. Arbitrario pero deterministico, y le
        # da al anotador una expectativa estable en vez de un sorteo por id del tracker.
        if evidencia[a].x_hasta(limite) > evidencia[b].x_hasta(limite):
            a, b = b, a
        # Solo los cuadros compartidos: ahi los dos son dos personas distintas.
        col_a = [c for f, c in evidencia[a].colores if f in comunes]
        col_b = [c for f, c in evidencia[b].colores if f in comunes]
        if not col_a or not col_b:
            continue
        pa = perfil_de(col_a)
        pb = perfil_de(col_b)
        sep = distancia_color(pa, pb)
        intentos.append((seg, round(sep, 3)))
        if mejor is None or sep > mejor[-1]:
            mejor = (seg, a, b, pa, pb, sep)
        if sep >= cfg.separacion_minima:
            elegida = (seg, a, b, pa, pb, sep)
            break

    prop.diagnostico["ventanas_probadas"] = intentos

    if not intentos:
        prop.sin_asignar = sorted(candidatos)
        prop.avisos.append(
            "ningun par de tracks tiene guantes con color en el video: no alcanza para "
            "sembrar los dos perfiles. Los candidatos quedan sin asignar"
        )
        return prop

    if elegida is None:
        # Se probaron todas las ventanas y los perfiles nunca separaron. No se asigna: con
        # separacion 0,449 el reparto salio 13 tracks a A contra 1 a B, que es una asignacion
        # equivocada con cara de asignacion. Los filtros de descarte SI quedan aplicados.
        seg, a, b, pa, pb, sep = mejor
        prop.perfiles = {"fighter_A": pa, "fighter_B": pb}
        prop.diagnostico["separacion_de_perfiles"] = round(sep, 3)
        prop.sin_asignar = sorted(candidatos)
        prop.avisos.append(
            f"los dos perfiles de color nunca se separaron: el mejor intento fue {sep:.3f} "
            f"con {seg:g} s, y el piso es {cfg.separacion_minima}. Los guantes de los dos "
            "peleadores son del mismo color, asi que NO se asigna A ni B: hacerlo produciria "
            f"un reparto equivocado con cara de correcto. Quedan {len(candidatos)} candidatos "
            "sin asignar, y los filtros de descarte si se aplicaron"
        )
        return prop

    seg, s1, s2, perfil_a, perfil_b, separacion = elegida
    prop.perfiles = {"fighter_A": perfil_a, "fighter_B": perfil_b}
    prop.semillas = {"fighter_A": s1, "fighter_B": s2}
    prop.diagnostico["separacion_de_perfiles"] = round(separacion, 3)
    prop.diagnostico["segundos_de_siembra"] = seg
    if seg > cfg.segundos_semilla:
        prop.avisos.append(
            f"con {cfg.segundos_semilla:g} s los perfiles no separaban; hizo falta extender "
            f"la ventana a {seg:g} s"
        )
    if perfil_a[0] == ACROMATICO and perfil_b[0] == ACROMATICO:
        prop.avisos.append(
            "los dos peleadores tienen guantes sin color -negros o blancos-: el tono no "
            "aporta y la decision queda solo en saturacion y brillo"
        )

    # -- 4. voto, con los perfiles refinandose -----------------------------
    # Los cuadros compartidos garantizan que las dos semillas son dos personas, pero son
    # pocos y el perfil sale ruidoso: sobre Sparring, sembrar solo con ellos bajaba el
    # acierto de 7/9 a 5/9. La salida no es elegir entre las dos cosas sino encadenarlas.
    # Los compartidos deciden QUIENES son; una vez que hay tracks asignados, los perfiles se
    # reestiman con todos sus guantes y se vuelve a votar. Converge en dos o tres vueltas y
    # no puede irse a cualquier lado, porque el punto de partida ya es correcto.
    cols_por_track = {t: [c for _, c in evidencia[t].colores] for t in candidatos}
    pa, pb = perfil_a, perfil_b
    votos: dict[int, tuple[int, int]] = {}
    roles_previos: dict[int, TrackRole] = {}

    for iteracion in range(cfg.iteraciones):
        roles_iter: dict[int, TrackRole] = {}
        votos = {}
        for t in sorted(candidatos):
            cols = cols_por_track[t]
            if len(cols) < cfg.min_guantes_voto:
                continue
            a = sum(1 for c in cols if distancia_color(c, pa) < distancia_color(c, pb))
            b = len(cols) - a
            votos[t] = (a, b)
            if a != b:
                roles_iter[t] = TrackRole.A if a > b else TrackRole.B

        if roles_iter == roles_previos:
            break
        roles_previos = roles_iter

        nuevos_a = [c for t, r in roles_iter.items() if r is TrackRole.A
                    for c in cols_por_track[t]]
        nuevos_b = [c for t, r in roles_iter.items() if r is TrackRole.B
                    for c in cols_por_track[t]]
        if not nuevos_a or not nuevos_b:
            break   # un lado vacio: no hay con que reestimar y se deja el perfil anterior
        pa, pb = perfil_de(nuevos_a), perfil_de(nuevos_b)

    prop.diagnostico["iteraciones"] = iteracion + 1
    prop.diagnostico["separacion_final"] = round(distancia_color(pa, pb), 3)
    prop.perfiles = {"fighter_A": pa, "fighter_B": pb}

    for t in sorted(candidatos):
        if t in roles_previos:
            prop.roles[t] = roles_previos[t]
        else:
            prop.sin_asignar.append(t)   # pocos guantes o empate: no se adivina

    prop.diagnostico["votos"] = votos
    prop.diagnostico["asignados_A"] = sum(1 for r in prop.roles.values() if r is TrackRole.A)
    prop.diagnostico["asignados_B"] = sum(1 for r in prop.roles.values() if r is TrackRole.B)
    prop.diagnostico["sin_asignar"] = len(prop.sin_asignar)

    if not prop.diagnostico["asignados_A"] or not prop.diagnostico["asignados_B"]:
        prop.avisos.append(
            "quedo un peleador sin ningun track asignado. Suele significar que la semilla "
            "tomo dos fragmentos de la MISMA persona en los primeros segundos"
        )
    return prop


@dataclass
class AsignarIdentidadAuto(_SnapshotCommand):
    """
    Aplica una propuesta entera como un solo gesto.

    Va en lote y con una sola instantanea a proposito: es una operacion que el anotador
    ejecuta y despues revisa, y tener que deshacerla rol por rol con cincuenta Ctrl+Z seria
    peor que no poder deshacerla. Un solo Ctrl+Z la revierte completa.

    Los tracks sin asignar NO reciben assignment. Dejarlos en blanco es la decision: salen
    en el color de sin asignar del overlay y el anotador los ve. Ponerles un rol adivinado
    los haria invisibles.
    """

    propuesta: Propuesta
    annotator: str = "identidad-auto"
    label: str = "asignar identidad automaticamente"
    forzar: bool = False
    aplicados: int = 0
    _antes: Identity | None = field(default=None, repr=False)

    def _aplicar(self, doc: AnnotationDoc) -> None:
        # Pisar trabajo manual se deshace con Ctrl+Z, pero solo si el anotador se da cuenta
        # a tiempo. Un swap corregido a mano y borrado en silencio reaparece recien en el
        # export, con los keypoints del peleador equivocado adentro.
        manuales = [
            a for a in doc.identity.assignments
            if a.role in (TrackRole.A, TrackRole.B)
            and a.origin.op in (AssignmentOp.SWAP, AssignmentOp.RESEED)
        ]
        if manuales and not self.forzar:
            raise ValueError(
                f"el documento ya tiene {len(manuales)} correccion/es manual/es de identidad "
                "(swap o reseed) y esta operacion reemplaza el bloque entero. "
                "Usar forzar=True si de verdad se quiere descartarlas"
            )
        origen = _origen(doc, AssignmentOp.MANUAL, 0, self.annotator)
        nuevas: list[Assignment] = []
        for tid, rol in sorted(self.propuesta.roles.items()):
            ini, fin = self.propuesta.rangos.get(tid, (0, doc.video.total_frames))
            if fin <= ini:
                continue
            nuevas.append(
                Assignment(
                    id=new_id(doc, "assignment"),
                    track_id=tid,
                    role=rol,
                    start_frame=ini,
                    end_frame_excl=fin,
                    origin=origen,
                )
            )
        self.aplicados = len(nuevas)
        # Reemplaza el bloque entero: esto se corre sobre un documento sin identidad resuelta,
        # y conservar asignaciones previas dejaria intervalos solapados del mismo track cuyo
        # ganador dependeria del orden de la lista.
        doc.identity.assignments = nuevas


# ---------------------------------------------------------------------------
# Persistencia de la evidencia
#
# Recorrer el video con el detector cuesta minutos; elegir umbrales sobre la evidencia ya
# medida cuesta milisegundos. Separarlos deja ajustar la ventana, el piso de separacion o el
# umbral de guante sin volver a decodificar nada, que es lo que permite medir en serio en
# vez de una corrida por corazonada.


def guardar_evidencia(evidencia: dict[int, EvidenciaTrack], destino: Path) -> None:
    import json

    datos = [
        {
            "track_id": e.track_id, "primer_frame": e.primer_frame,
            "ultimo_frame": e.ultimo_frame, "alto_max": e.alto_max,
            "recortes": e.recortes, "con_guante": e.con_guante,
            "colores": [[f, list(c)] for f, c in e.colores],
            "frames_con_guante": e.frames_con_guante,
            "xs": [[f, x] for f, x in e.xs],
        }
        for e in evidencia.values()
    ]
    Path(destino).write_text(json.dumps({"kind": "boxtwin.identidad.evidencia",
                                         "tracks": datos}) + "\n")


def cargar_evidencia(origen: Path) -> dict[int, EvidenciaTrack]:
    import json

    d = json.loads(Path(origen).read_text())
    if d.get("kind") != "boxtwin.identidad.evidencia":
        raise ValueError(f"{origen} no es un archivo de evidencia de identidad")
    salida = {}
    for t in d["tracks"]:
        salida[t["track_id"]] = EvidenciaTrack(
            track_id=t["track_id"], primer_frame=t["primer_frame"],
            ultimo_frame=t["ultimo_frame"], alto_max=t["alto_max"],
            recortes=t["recortes"], con_guante=t["con_guante"],
            colores=[(f, tuple(c)) for f, c in t["colores"]],
            frames_con_guante=list(t["frames_con_guante"]),
            xs=[(f, x) for f, x in t["xs"]],
        )
    return salida


def puntuar_contra(propuesta: Propuesta, doc: AnnotationDoc) -> dict:
    """
    Compara la propuesta con las asignaciones manuales del documento.

    La etiqueta A o B que pone la propuesta es arbitraria -sale de quien arranca mas a la
    izquierda- asi que se prueban las dos correspondencias y gana la mejor. Lo que se esta
    midiendo es si la PARTICION es correcta, no si los nombres coinciden.
    """
    verdad: dict[int, TrackRole] = {}
    for a in doc.identity.assignments:
        if a.role in (TrackRole.A, TrackRole.B):
            verdad[a.track_id] = a.role

    comunes = [t for t in verdad if t in propuesta.roles
               and propuesta.roles[t] in (TrackRole.A, TrackRole.B)]
    if not comunes:
        return {"evaluables": 0}

    directo = sum(1 for t in comunes if propuesta.roles[t] is verdad[t])
    cruzado = len(comunes) - directo
    aciertos = max(directo, cruzado)

    # Peleadores que la propuesta mando a ignore: son falsos descartes, y es el error caro.
    descartados = [t for t in verdad if propuesta.roles.get(t) is TrackRole.IGNORE]
    sin_decidir = [t for t in verdad if t in propuesta.sin_asignar]
    return {
        "evaluables": len(comunes),
        "aciertos": aciertos,
        "acierto": round(aciertos / len(comunes), 4),
        "etiquetas_invertidas": cruzado > directo,
        "peleadores_descartados_por_los_filtros": len(descartados),
        "peleadores_sin_decidir": len(sin_decidir),
        "peleadores_en_la_verdad": len(verdad),
    }
