"""
BoxTwin - Los dos indicadores de guardia.

POR QUE EXISTE
  Es lo primero que mira un entrenador y lo que la entrevista a Suarez puso arriba de todo:
  no cuantos golpes tiro, sino si la mano vuelve al menton y si la otra se le cae mientras
  pega. El volumen de golpes se cuenta; esto se corrige.

  Y es lo unico del MVP que no venia medido de antes, asi que entra condicionado: hay un
  criterio de aceptacion escrito antes de mirar los datos -acuerdo de 80% contra 100 golpes
  marcados a mano- y si no lo pasa, sale del producto en vez de quedarse como numero que
  nadie valido.

  Una limitacion que no se negocia: la profundidad no se ve. Una mano que en la imagen esta
  al lado del menton puede estar treinta centimetros adelante. Los dos indicadores son
  estimaciones geometricas sobre la proyeccion y salen rotulados asi.

QUE HACE
  Sobre cada golpe detectado mide dos cosas:

  RETORNO. Cuantos milisegundos tarda la mano que pego en volver a la zona de guardia. La
  zona es "muneca a menos de 0,6 anchos de hombro de la nariz", que es la definicion
  operativa del criterio de dominio. Si no vuelve antes del proximo golpe del mismo brazo,
  o antes del limite de busqueda, se reporta como "no vuelve" y no como un numero grande.

  MANO OPUESTA. Que fraccion del golpe la otra mano estuvo FUERA de esa zona. Es la que
  tiene que quedarse arriba tapando mientras la otra sale.

  El umbral de retorno lento no esta calibrado todavia y por eso es un parametro, no una
  constante: sale de la mediana medida sobre sparring-3 y esa medicion es parte de C3. Hasta
  entonces el indicador devuelve el numero y marca `calibrado: false`.

USO
  from boxtwin.mvp.guardia import medir
  eventos = medir(kp, sc, golpes, fps=30.0)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["ConfigGuardia", "EventoGuardia", "medir", "en_guardia"]

# COCO17
NARIZ = 0
L_HOM, R_HOM = 5, 6
L_MUN, R_MUN = 9, 10

MIN_SCORE = 0.3


@dataclass(frozen=True)
class ConfigGuardia:
    """
    Parametros de los dos indicadores.

    `radio_guardia` es la definicion operativa del criterio de dominio: 0,6 anchos de hombro
    entre la muneca y la nariz. Se mide en anchos de hombro y no en pixeles porque la
    distancia a la camara cambia entre un video y otro, y un umbral en pixeles mide el zoom.

    MEDIDO EL 22-09 SOBRE amateur_estatico Y HAY QUE DECIRLO: con 0,6 el indicador marca la
    mano opuesta caida en 40 de 41 golpes, que leido como tactica diria que dos amateurs de
    competencia pelean dos minutos con la guardia abajo. Sobre 3127 cuadros con pose
    confiable, la muneca esta a 1,4-1,75 anchos de hombro de la nariz de mediana y solo el
    2-6% de los cuadros cae bajo 0,6. Tiene sentido geometrico: en guardia el puno esta al
    lado del menton, y del menton a la nariz ya hay distancia. El 0,6 no describe "mano
    arriba" sino "mano tocandose la cara".

    NO se cambio el default, a proposito: la definicion esta pre-registrada en la spec como
    criterio de dominio a confirmar por Lucas, y moverla mirando estos datos seria elegir el
    criterio despues de ver el resultado. El p10 medido anda entre 0,75 y 1,09, asi que la
    calibracion de C3 probablemente termine cerca de 1,0. Ver
    docs/experiments/2026-09-22-mvp-sobre-video-real.md

    `retorno_lento_ms` es el unico numero sin calibrar del modulo y por eso es None por
    defecto: con un valor inventado el indicador diria "guardia baja" sin que nadie haya
    comprobado que eso es guardia baja. Con None se reporta el tiempo medido y nada mas.
    """

    radio_guardia: float = 0.6
    retorno_lento_ms: float | None = None
    # Hasta donde se busca el retorno despues del golpe. Un segundo y medio es mucho mas que
    # cualquier retorno legitimo; pasado eso, lo que haya que decir no es "tardo".
    limite_busqueda_ms: float = 1500.0
    # Fraccion del golpe con la otra mano afuera para marcarla caida. La mitad es un punto
    # de partida razonado y tambien entra en C3.
    fraccion_mano_caida: float = 0.5


@dataclass
class EventoGuardia:
    """Lo medido sobre un golpe. Los dos indicadores viajan con su numero, no solo con el si/no."""

    golpe: str
    peleador: str
    brazo: str
    t: float
    retorno_ms: float | None = None          # None = no volvio adentro del limite
    retorno_lento: bool | None = None        # None = sin umbral calibrado
    fraccion_opuesta_afuera: float | None = None
    mano_opuesta_caida: bool | None = None
    medible: bool = True
    motivo: str | None = None

    def a_dict(self) -> dict:
        return asdict(self)


def en_guardia(
    kp: np.ndarray, sc: np.ndarray, muneca: int, radio: float
) -> np.ndarray:
    """
    Por cuadro: la muneca esta adentro de la zona de guardia, y se pudo medir.

    Devuelve un array de tres estados y no un booleano, porque "la mano esta abajo" y "no se
    ve la mano" no son lo mismo y tratarlos igual convierte una oclusion en un descuido:
    1 adentro, 0 afuera, -1 no medible.
    """
    kp = np.asarray(kp, np.float64)
    sc = np.asarray(sc, np.float64)
    ancho = np.linalg.norm(kp[:, L_HOM] - kp[:, R_HOM], axis=-1)
    d = np.linalg.norm(kp[:, muneca] - kp[:, NARIZ], axis=-1)

    ok = (
        (sc[:, muneca] >= MIN_SCORE)
        & (sc[:, NARIZ] >= MIN_SCORE)
        & (sc[:, L_HOM] >= MIN_SCORE)
        & (sc[:, R_HOM] >= MIN_SCORE)
        & (ancho > 1e-6)
    )
    estado = np.where(d <= radio * np.where(ancho > 1e-6, ancho, 1.0), 1, 0)
    return np.where(ok, estado, -1).astype(np.int8)


def medir(
    kp: np.ndarray,
    sc: np.ndarray,
    golpes: list,
    fps: float,
    cfg: ConfigGuardia | None = None,
) -> list[EventoGuardia]:
    """
    Los dos indicadores sobre cada golpe detectado.

    `kp` es (2, T, 17, 2) con la identidad ya resuelta y `golpes` son los objetos que
    devuelve el detector, en cuadros del video original.
    """
    cfg = cfg or ConfigGuardia()
    kp = np.asarray(kp)
    sc = np.asarray(sc)
    if fps <= 0:
        raise ValueError("el fps tiene que ser positivo")

    indices = {"A": 0, "B": 1}
    muneca_de = {"left": L_MUN, "right": R_MUN}
    limite = int(round(cfg.limite_busqueda_ms * fps / 1000.0))

    # El retorno se busca hasta el proximo golpe del mismo carril: si el peleador vuelve a
    # tirar, lo que pase despues ya es otro golpe y no el retorno de este.
    proximo: dict[int, int] = {}
    por_carril: dict[tuple[str, str], list] = {}
    for i, g in enumerate(golpes):
        por_carril.setdefault((g.peleador, g.brazo), []).append(i)
    for ids in por_carril.values():
        ids.sort(key=lambda i: golpes[i].inicio)
        for a, b in zip(ids, ids[1:]):
            proximo[a] = golpes[b].inicio

    salida = []
    for i, g in enumerate(golpes):
        p = indices.get(g.peleador)
        m = muneca_de.get(g.brazo)
        opuesta = R_MUN if m == L_MUN else L_MUN
        ev = EventoGuardia(
            golpe=getattr(g, "id", "") or f"{g.peleador}-{g.brazo}-{g.inicio}",
            peleador=g.peleador, brazo=g.brazo, t=round(g.t_inicio, 3),
        )
        if p is None or m is None:
            ev.medible, ev.motivo = False, "carril desconocido"
            salida.append(ev)
            continue

        # -- mano opuesta, durante el golpe ---------------------------------
        est_op = en_guardia(kp[p, g.inicio : g.fin + 1], sc[p, g.inicio : g.fin + 1],
                            opuesta, cfg.radio_guardia)
        medibles = est_op >= 0
        if medibles.any():
            fuera = float((est_op[medibles] == 0).mean())
            ev.fraccion_opuesta_afuera = round(fuera, 3)
            ev.mano_opuesta_caida = fuera >= cfg.fraccion_mano_caida
        else:
            ev.motivo = "no se vio la mano opuesta durante el golpe"

        # -- retorno de la mano que pego ------------------------------------
        fin_busqueda = min(len(kp[p]) - 1, g.fin + limite)
        tope = proximo.get(i)
        if tope is not None:
            fin_busqueda = min(fin_busqueda, max(g.fin, tope - 1))
        est = en_guardia(kp[p, g.fin : fin_busqueda + 1], sc[p, g.fin : fin_busqueda + 1],
                         m, cfg.radio_guardia)
        vueltas = np.flatnonzero(est == 1)
        if len(vueltas):
            ev.retorno_ms = round(float(vueltas[0]) * 1000.0 / fps, 1)
            if cfg.retorno_lento_ms is not None:
                ev.retorno_lento = ev.retorno_ms > cfg.retorno_lento_ms
        elif (est >= 0).any():
            ev.retorno_ms = None          # se vio y no volvio
            if cfg.retorno_lento_ms is not None:
                ev.retorno_lento = True
        else:
            ev.medible = False
            ev.motivo = (ev.motivo or "") + " no se vio la mano que pego despues del golpe"
            ev.motivo = ev.motivo.strip()
        salida.append(ev)
    return salida
