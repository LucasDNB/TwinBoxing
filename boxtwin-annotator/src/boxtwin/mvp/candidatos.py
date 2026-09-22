"""
BoxTwin - Los dos recortes que se le muestran al usuario para que diga cual es cual.

POR QUE EXISTE
  Es el paso 4 del flujo y la unica intervencion humana del producto. Existe porque esta
  medido lo que cuesta no tenerlo: la identidad automatica resuelve 82,4% de los tracks
  sola y 99,1% cuando los perfiles se siembran desde dos tracks con rol conocido. Toda esa
  diferencia es la siembra.

  Lo que se le pide a la persona no es que revise el video entero sino que conteste una
  pregunta de dos opciones sobre dos imagenes. Por eso los candidatos tienen que ser dos
  tracks que se vean AL MISMO TIEMPO y SEPARADOS: si se los muestra en cuadros distintos,
  nada le garantiza al que mira que no sea la misma persona dos veces, que es exactamente
  el error que la siembra tiene que evitar.

QUE HACE
  Ordena los tracks que pasaron los filtros por cuanto coexisten de a pares, propone el par
  que mas coexiste, y recorta a cada uno en un cuadro donde los dos estan en pantalla.
  Cuando ningun par coexiste lo suficiente devuelve los candidatos sueltos igual, que es el
  caso de "no pude separar dos: elegi vos" del flujo.

USO
  from boxtwin.mvp.candidatos import elegir, recortar
  pareja, cands = elegir(evidencia, nucleo, cfg)
  recortar(video, cache, cands, dir_sesion / "candidatos")
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = ["Candidato", "elegir", "recortar"]

# Margen alrededor de la caja, en fracciones de su tamano. El recorte justo al borde de la
# caja se lee mal: el usuario tiene que reconocer a una persona, no a un bounding box.
MARGEN = 0.12


@dataclass
class Candidato:
    """Un track que el sistema propone como peleador, con donde mirarlo."""

    track: int
    cuadro: int
    alto: float
    fraccion_guante: float
    cuadros_con_color: int
    coexiste_con: int | None = None
    cuadros_de_coexistencia: int = 0
    recorte: str | None = None
    avisos: list[str] = field(default_factory=list)

    def a_dict(self) -> dict:
        return asdict(self)


def elegir(
    evidencia: dict,
    nucleo: list[int] | set[int],
    min_coexistencia: int = 3,
    maximo: int = 6,
) -> tuple[tuple[int, int] | None, list[Candidato]]:
    """
    El par que mas coexiste, y hasta `maximo` candidatos ordenados por evidencia.

    La coexistencia se cuenta sobre los cuadros donde el color se pudo muestrear, que son
    justamente los cuadros donde el track esta AISLADO de la gente de su tamano. Dos tracks
    aislados en el mismo cuadro son dos personas distintas, y eso es lo que hace que la
    pregunta que se le hace al usuario tenga sentido.
    """
    nucleo = sorted(nucleo)
    frames = {t: {f for f, _ in evidencia[t].colores} for t in nucleo}

    mejor: tuple[int, int] | None = None
    mejor_n = 0
    for i, t1 in enumerate(nucleo):
        for t2 in nucleo[i + 1:]:
            n = len(frames[t1] & frames[t2])
            # El desempate es por id y no por cualquier orden de diccionario: dos corridas
            # sobre el mismo video tienen que proponer el mismo par.
            if n > mejor_n:
                mejor, mejor_n = (t1, t2), n
    if mejor_n < min_coexistencia:
        mejor = None

    orden = list(mejor) if mejor else []
    orden += sorted(
        (t for t in nucleo if t not in orden),
        key=lambda t: (-len(frames[t]), -evidencia[t].fraccion_guante, t),
    )

    cuadro_par = _cuadro_comun(frames, mejor) if mejor else None
    salida = []
    for t in orden[:maximo]:
        fs = sorted(frames[t])
        cuadro = cuadro_par if (mejor and t in mejor and cuadro_par is not None) else (
            fs[len(fs) // 2] if fs else evidencia[t].primer_frame
        )
        c = Candidato(
            track=t,
            cuadro=int(cuadro),
            alto=round(float(evidencia[t].alto_max), 4),
            fraccion_guante=round(float(evidencia[t].fraccion_guante), 4),
            cuadros_con_color=len(fs),
            coexiste_con=(mejor[1] if mejor and t == mejor[0]
                          else mejor[0] if mejor and t == mejor[1] else None),
            cuadros_de_coexistencia=mejor_n if mejor and t in mejor else 0,
        )
        salida.append(c)
    return mejor, salida


def _cuadro_comun(frames: dict[int, set[int]], par: tuple[int, int]) -> int | None:
    """
    Un cuadro donde los dos estan en pantalla y separados.

    Se toma la mediana de los compartidos y no el primero: al principio del video suele
    estar alguien entrando al plano, con media persona fuera del cuadro.
    """
    comunes = sorted(frames[par[0]] & frames[par[1]])
    return comunes[len(comunes) // 2] if comunes else None


def recortar(
    video: Path, cache, candidatos: list[Candidato], destino: Path
) -> list[Candidato]:
    """
    Escribe un jpg por candidato y le deja la ruta relativa adentro.

    Si un recorte no sale -el cuadro no se pudo leer, el track no esta en ese cuadro- se
    deja el candidato sin imagen con su aviso, en vez de sacarlo de la lista: el usuario
    tiene que poder elegirlo igual desde el numero de track.
    """
    import cv2

    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"no se pudo abrir el video {video}")
    try:
        for c in candidatos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, c.cuadro)
            ok, img = cap.read()
            if not ok:
                c.avisos.append(f"no se pudo leer el cuadro {c.cuadro}")
                continue
            dets = cache.detections(c.cuadro)
            i = dets.index_of_track(c.track)
            if i is None:
                c.avisos.append(f"el track {c.track} no esta en el cuadro {c.cuadro}")
                continue
            x0, y0, x1, y1 = (float(v) for v in dets.bbox[i])
            mx, my = (x1 - x0) * MARGEN, (y1 - y0) * MARGEN
            h, w = img.shape[:2]
            x0 = max(0, int(x0 - mx))
            y0 = max(0, int(y0 - my))
            x1 = min(w, int(x1 + mx))
            y1 = min(h, int(y1 + my))
            if x1 - x0 < 4 or y1 - y0 < 4:
                c.avisos.append("la caja quedo demasiado chica para recortar")
                continue
            nombre = f"track_{c.track}.jpg"
            cv2.imwrite(str(destino / nombre), img[y0:y1, x0:x1])
            c.recorte = f"{destino.name}/{nombre}"
    finally:
        cap.release()
    return candidatos
