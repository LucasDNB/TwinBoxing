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

    `segundos_ancla` no recorta la evidencia: la particion usa el video entero. Lo que hace
    es elegir el componente de REFERENCIA, el que define cual lado es A y contra el que se
    orientan los demas, entre los que aparecen al principio. La primera version si recortaba
    -sembraba con los primeros cinco segundos- y era una fuente de errores: sobre 01-sparring
    los perfiles salian a tono 7,3 y 172,9 con cinco segundos, contra 3,5 y 9,1 con el video
    entero. Era poca evidencia, no un color distinto.

    `separacion_minima` es un piso duro, no un aviso, y aplica SOLO a los tracks que la
    coexistencia no resolvio. Con 0,449 el voto por color repartio 13 tracks a A contra 1 a
    B, que es una asignacion equivocada con cara de asignacion.
    """

    segundos_ancla: float = 5.0      # de donde tiene que salir el componente de referencia
    separacion_minima: float = 0.55  # debajo de esto no se asigna A ni B
    paso: int = 5                    # se evalua un cuadro de cada N
    fraccion_altura: float = 0.55
    umbral_guante: float = 0.45
    min_recortes: int = 15           # menos que esto y la fraccion no significa nada
    min_guantes_voto: int = 10       # menos que esto y el voto se deja sin asignar
    min_coexistencia: int = 3        # cuadros en que dos tracks tienen que coincidir
    min_guantes_rescate: int = 3     # guantes minimos para rescatar un fragmento corto
    # Diferencia minima entre las dos orientaciones posibles de un componente para animarse
    # a elegir una. Por debajo, el color no esta decidiendo y se deja sin asignar.
    margen_orientacion: float = 0.25
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

    # -- 2. guante, y que significa descartar ------------------------------
    # Hay tres situaciones y antes se mezclaban las tres en `ignore`:
    #
    #   MEDIDO Y NO ES PELEADOR: recortes suficientes y fraccion de guante baja. El arbitro,
    #   el entrenador, el cronometrista. Eso si es ignore, y es lo que el filtro sabe hacer.
    #
    #   MEDIDO Y ES PELEADOR: entra al nucleo, que es con lo que se arma la particion.
    #
    #   NO SE PUDO MEDIR: un fragmento de pocos cuadros, o un track que quedo chico en un
    #   plano abierto. Marcarlo ignore es afirmar algo que no se comprobo, y la afirmacion
    #   es invisible: el track desaparece del export y nadie lo ve faltar. Medido, eso se
    #   llevaba 61 de 185 tracks de peleador, el 33%, de los cuales 31 eran fragmentos de
    #   menos de quince recortes y 23 habian quedado bajos.
    #
    # Ahora el tercero queda SIN ASIGNAR, que es lo que el modulo hace con todo lo que no
    # puede decidir, y ademas se le da una segunda oportunidad mas abajo.
    nucleo = {
        t for t in pasan_altura
        if evidencia[t].recortes >= cfg.min_recortes
        and evidencia[t].fraccion_guante >= cfg.umbral_guante
    }
    # El rescate es para UN caso concreto y no para todo lo que tenga un guante: el track
    # que pasa altura y guante pero se quedo corto de cuadros. Son fragmentos de alguien que
    # ya se comprobo que es del tamano de un peleador y que lleva guantes, y descartarlos por
    # durar poco es descartar por no haber podido medir.
    #
    # La version amplia de esto -cualquier track con un guante- metia 104 tracks con rol de
    # peleador que la anotacion no tiene como tal, y es el error caro: keypoints de otro
    # cuerpo exportados como peleador, invisibles en el overlay. La causa era una falla
    # logica: coexistir con A prueba que NO es A, no que SEA B, y eso ultimo solo se sigue si
    # ya se sabe que es peleador.
    #
    # Los que quedan bajos NO se rescatan, y hay un numero atras: bajar el umbral de altura
    # de 0,55 a 0,35 rescata 7 peleadores y cuela 47 que no lo son. No hay umbral que separe,
    # asi que esos quedan sin asignar y los mira una persona.
    rescatables = {
        t for t in pasan_altura
        if t not in nucleo
        and evidencia[t].con_guante >= cfg.min_guantes_rescate
        and evidencia[t].fraccion_guante >= cfg.umbral_guante
    }
    for t in evidencia:
        if t in nucleo or t in rescatables:
            continue
        if evidencia[t].recortes >= cfg.min_recortes:
            prop.roles[t] = TrackRole.IGNORE      # medido: no es peleador
        else:
            prop.sin_asignar.append(t)            # no alcanzo para decir nada

    prop.diagnostico = {
        "tracks": len(evidencia),
        "umbral_altura": round(umbral_alto, 4),
        "pasan_altura": len(pasan_altura),
        "nucleo": len(nucleo),
        "rescatables": len(rescatables),
    }
    if not nucleo:
        prop.sin_asignar.extend(sorted(rescatables))
        prop.avisos.append(
            "ningun track pasa altura y guante con evidencia suficiente: no hay nucleo con "
            "que armar la particion"
        )
        return prop
    candidatos = nucleo

    # -- 3. particion por coexistencia -------------------------------------
    # La geometria primero, porque no se equivoca: dos tracks en el mismo cuadro, cada uno
    # aislado de la gente de su tamano, son dos personas distintas. Vale aunque los guantes
    # sean identicos y aunque un track haya cambiado de persona a mitad de camino.
    limite_ancla = int(cfg.segundos_ancla * fps) if fps > 0 else 0
    lados, diag_part = particionar(evidencia, candidatos, cfg, limite_ancla)
    prop.diagnostico.update(diag_part)

    if not lados:
        prop.sin_asignar = sorted(candidatos)
        prop.avisos.append(
            "ningun par de tracks coexiste lo suficiente: sin esa restriccion no hay como "
            "afirmar que dos tracks son dos personas distintas. Quedan sin asignar"
        )
        return prop

    cols = {t: [c for _, c in evidencia[t].colores] for t in candidatos}
    perfil_0 = perfil_de([c for t, v in lados.items() if v == 0 for c in cols[t]])
    perfil_1 = perfil_de([c for t, v in lados.items() if v == 1 for c in cols[t]])
    separacion = distancia_color(perfil_0, perfil_1)
    prop.diagnostico["separacion_de_perfiles"] = round(separacion, 3)

    # -- 4. los sueltos, por color -----------------------------------------
    # Un track que nunca coexiste con ninguno de los ya repartidos no tiene restriccion
    # geometrica que lo ate, y ahi si decide el color. Si los dos perfiles estan demasiado
    # juntos no se decide nada: con separacion 0,449 el reparto salio 13 a 1, una asignacion
    # equivocada con cara de correcta. Lo que la geometria SI decidio se conserva igual.
    votos: dict[int, tuple[int, int]] = {}
    color_sirve = separacion >= cfg.separacion_minima
    if not color_sirve:
        prop.avisos.append(
            f"los dos lados tienen guantes casi del mismo color (separacion {separacion:.3f}, "
            f"piso {cfg.separacion_minima}). Los {len(lados)} tracks que la coexistencia "
            "resolvio quedan asignados igual; los demas quedan sin asignar"
        )

    for t in sorted(candidatos):
        if t in lados:
            continue
        if not color_sirve or len(cols[t]) < cfg.min_guantes_voto:
            prop.sin_asignar.append(t)
            continue
        a = sum(1 for c in cols[t]
                if distancia_color(c, perfil_0) < distancia_color(c, perfil_1))
        b = len(cols[t]) - a
        votos[t] = (a, b)
        if a == b:
            prop.sin_asignar.append(t)   # empate: no se adivina
        else:
            lados[t] = 0 if a > b else 1

    # -- 4b. rescate -------------------------------------------------------
    # Los que no entraron al nucleo tienen guantes igual. Se los engancha con las dos reglas
    # que ya estan, en orden de confianza: si coexisten con un track ya repartido son la otra
    # persona -geometria, no se discute- y si no, decide el color. Lo que ninguna de las dos
    # resuelve queda sin asignar, no ignore: no se comprobo que no sea peleador.
    frames_lado = {t: {f for f, _ in evidencia[t].colores} for t in lados}
    rescatados_geo = rescatados_color = 0
    for t in sorted(rescatables):
        fs = {f for f, _ in evidencia[t].colores}
        vecino = None
        if fs:
            for u, v in lados.items():
                if len(fs & frames_lado.get(u, set())) >= cfg.min_coexistencia:
                    vecino = 1 - v
                    break
        if vecino is not None:
            lados[t] = vecino
            cols[t] = [c for _, c in evidencia[t].colores]
            rescatados_geo += 1
            continue
        propios = [c for _, c in evidencia[t].colores]
        if not color_sirve or len(propios) < cfg.min_guantes_voto:
            prop.sin_asignar.append(t)
            continue
        a = sum(1 for c in propios
                if distancia_color(c, perfil_0) < distancia_color(c, perfil_1))
        b = len(propios) - a
        if a == b:
            prop.sin_asignar.append(t)
        else:
            lados[t] = 0 if a > b else 1
            cols[t] = propios
            rescatados_color += 1

    prop.diagnostico["rescatados_por_coexistencia"] = rescatados_geo
    prop.diagnostico["rescatados_por_color"] = rescatados_color

    # -- 5. cual lado es A --------------------------------------------------
    # El que arranca mas a la izquierda. Arbitrario pero deterministico: le da al anotador
    # una expectativa estable en vez de un sorteo por id interno del tracker.
    def x_medio(valor: int) -> float:
        xs = [x for t, v in lados.items() if v == valor for _, x in evidencia[t].xs]
        return float(np.median(xs)) if xs else 0.0

    lado_a = 0 if x_medio(0) <= x_medio(1) else 1
    for t, v in lados.items():
        prop.roles[t] = TrackRole.A if v == lado_a else TrackRole.B

    pa, pb = (perfil_0, perfil_1) if lado_a == 0 else (perfil_1, perfil_0)
    prop.perfiles = {"fighter_A": pa, "fighter_B": pb}
    prop.semillas = {
        "fighter_A": min((t for t, v in lados.items() if v == lado_a), default=-1),
        "fighter_B": min((t for t, v in lados.items() if v != lado_a), default=-1),
    }

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


# ---------------------------------------------------------------------------
# Particion global por coexistencia
#
# Una semilla es un punto de apoyo: si esta contaminada -si el tracker le metio dos personas
# al mismo id- todo lo que cuelgue de ella sale torcido. Sobre Sparring y 03-sparring eso
# dejaba la particion en 5/9 y 11/19, y ninguna variante de "elegir mejor la semilla" lo
# arreglo, porque el problema no es cual se elige sino que haya UNA sola.
#
# La salida es no apoyarse en una. Si dos tracks aparecen en el mismo cuadro, cada uno
# aislado de la gente de su tamano, son dos personas distintas. Eso no es una heuristica de
# apariencia: es geometria, y no se equivoca ni cuando los guantes son del mismo color ni
# cuando un track cambio de persona. Cada par que coexiste es una restriccion de "estos dos
# van separados", y el conjunto de esas restricciones es un grafo que hay que pintar de dos
# colores.
#
# El color de guante queda para lo que la geometria no puede: orientar los pedazos sueltos
# del grafo entre si, porque dos tracks que nunca coexisten no tienen restriccion que los
# ate.


def _componentes(adyacencia: dict[int, set[int]], nodos: list[int]) -> list[list[int]]:
    vistos: set[int] = set()
    salida = []
    for n in nodos:
        if n in vistos:
            continue
        pila, comp = [n], []
        vistos.add(n)
        while pila:
            x = pila.pop()
            comp.append(x)
            for y in adyacencia.get(x, ()):  # noqa: SIM118
                if y not in vistos:
                    vistos.add(y)
                    pila.append(y)
        salida.append(sorted(comp))
    return salida


def _pintar(componente: list[int], adyacencia: dict[int, set[int]]) -> dict[int, int]:
    """
    Pinta de dos colores por BFS. Devuelve track -> 0 o 1, sin los que generan conflicto.

    Un conflicto es un ciclo impar: tres tracks que coexisten de a pares, lo que no puede
    pasar con dos peleadores. Significa que alguna coexistencia es espuria -una deteccion
    duplicada, alguien del publico del tamano de un peleador- y esos tracks se dejan afuera
    en vez de forzar una respuesta.
    """
    lado: dict[int, int] = {componente[0]: 0}
    cola = [componente[0]]
    conflictivos: set[int] = set()
    while cola:
        x = cola.pop(0)
        for y in sorted(adyacencia.get(x, ())):
            if y not in lado:
                lado[y] = 1 - lado[x]
                cola.append(y)
            elif lado[y] == lado[x]:
                conflictivos.add(y)
    for t in conflictivos:
        lado.pop(t, None)
    return lado


def particionar(
    evidencia: dict[int, EvidenciaTrack],
    candidatos: set[int],
    cfg: ConfigIdentidadAuto,
    limite_ancla: int = 0,
) -> tuple[dict[int, int], dict]:
    """
    Reparte los candidatos en dos lados: primero por coexistencia, despues por color.

    Devuelve track -> 0 o 1, y un diagnostico. El lado 0 y el 1 todavia no son A y B: cual es
    cual lo decide despues la posicion en pantalla.
    """
    frames = {t: {f for f, _ in evidencia[t].colores} for t in candidatos}
    frames = {t: fs for t, fs in frames.items() if fs}
    nodos = sorted(frames)
    if len(nodos) < 2:
        return {}, {"motivo": "menos de dos tracks con color medido"}

    adyacencia: dict[int, set[int]] = {t: set() for t in nodos}
    for i, t1 in enumerate(nodos):
        for t2 in nodos[i + 1:]:
            if len(frames[t1] & frames[t2]) >= cfg.min_coexistencia:
                adyacencia[t1].add(t2)
                adyacencia[t2].add(t1)

    comps = _componentes(adyacencia, nodos)
    # Solo sirven los pedazos que tienen los dos lados: un componente de un track suelto no
    # dice nada por si mismo y se resuelve despues por color.
    pintados = [(_pintar(c, adyacencia), c) for c in comps]
    con_dos = [(l, c) for l, c in pintados if len(set(l.values())) == 2]
    if not con_dos:
        return {}, {"motivo": "ningun grupo de tracks coexiste; no hay restriccion geometrica",
                    "componentes": len(comps)}

    def perfil_lado(lado_map, valor):
        cols = [c for t, v in lado_map.items() if v == valor
                for _, c in evidencia[t].colores]
        return perfil_de(cols) if cols else None

    # El ancla: el componente mas grande de entre los que ya aparecen al principio del
    # video. Su lado 0 es el lado 0 global y los demas se orientan contra el. Que salga del
    # principio no es una restriccion tecnica sino una decision de producto: el anotador
    # abre el video, ve los primeros segundos y sabe cual es A sin tener que adivinarlo.
    def empieza_temprano(lado_map) -> bool:
        return any(
            any(f < limite_ancla for f, _ in evidencia[t].colores) for t in lado_map
        )

    con_dos.sort(key=lambda x: -len(x[0]))
    tempranos = [x for x in con_dos if empieza_temprano(x[0])] if limite_ancla else []
    ancla_temprana = bool(tempranos)
    if not tempranos:
        tempranos = con_dos
    base, _ = tempranos[0]
    # El resto se orienta contra el ancla, empezando por los mas grandes.
    otros = [x for x in con_dos if x[0] is not base]
    salida = dict(base)
    p0, p1 = perfil_lado(base, 0), perfil_lado(base, 1)

    # Los demas componentes se orientan por color contra el de referencia. La geometria ya
    # dijo que adentro de cada uno van separados; falta saber cual de sus dos lados es cual.
    invertidos = sin_orientar = 0
    for lado_map, _ in otros:
        q0, q1 = perfil_lado(lado_map, 0), perfil_lado(lado_map, 1)
        if None in (p0, p1, q0, q1):
            sin_orientar += 1
            continue
        directo = distancia_color(q0, p0) + distancia_color(q1, p1)
        cruzado = distancia_color(q0, p1) + distancia_color(q1, p0)
        # Orientar un componente contra otro es lo unico que la geometria no puede, porque
        # por definicion no comparten un solo cuadro. Si las dos orientaciones dan casi lo
        # mismo, el color no esta decidiendo nada y ponerlo igual es tirar una moneda: se
        # deja el componente sin asignar y lo resuelve el anotador, que si puede mirarlo.
        if abs(directo - cruzado) < cfg.margen_orientacion:
            sin_orientar += 1
            continue
        invertir = cruzado < directo
        invertidos += int(invertir)
        for t, v in lado_map.items():
            salida[t] = (1 - v) if invertir else v

    diag = {
        "componentes": len(comps),
        "componentes_con_dos_lados": len(con_dos),
        "tracks_por_coexistencia": len(salida),
        "componentes_invertidos_por_color": invertidos,
        "componentes_sin_orientar": sin_orientar,
        "ancla_del_principio": ancla_temprana,
    }
    return salida, diag
