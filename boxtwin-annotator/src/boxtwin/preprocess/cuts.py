"""
BoxTwin - Deteccion de cortes de plano.

POR QUE EXISTE
  BoT-SORT no sabe que hubo un corte de camara. Ante un cambio de plano ve dos cuerpos en
  posiciones nuevas y hace lo que sabe hacer: los asocia con los tracks que venia siguiendo.
  El resultado es que el track del peleador A pasa a contener el cuerpo del B, sin ningun
  aviso, y es el peor error de este sistema porque no se ve: el esqueleto sigue dibujandose
  sobre un cuerpo y el overlay se ve perfecto. El dano aparece en el export, cuando ya se
  anoto encima.

  Sobre sparring de camara fija esto no pasa nunca y por eso no hacia falta. Sobre una
  transmision profesional pasa decenas de veces por round.

  La solucion no es adivinar la correspondencia a traves del corte, que es justo lo que se
  equivoca: es declarar que ahi la identidad se corta. El tracker se reinicia, los ids nuevos
  arrancan de cero con un offset, y queda registrada una costura que el timeline dibuja. El
  anotador ve la marca y sabe que tiene que reasignar. Es el mismo mecanismo que ya existia
  para reanudar el preproceso, aplicado a otra causa.

  El detector se valido antes de usarlo, porque un umbral sin control no informa nada: sobre
  un video armado pegando tres clips distintos encuentra exactamente 2 cortes con umbrales
  0,2, 0,3 y 0,4, y sobre Sparring.mp4, que es camara fija, encuentra 0.

QUE HACE
  Corre el filtro de escena de ffmpeg en una pasada y devuelve los indices de cuadro donde
  cambia el plano.

USO
  from boxtwin.preprocess.cuts import detectar_cortes
  cortes = detectar_cortes(Path("pelea.mp4"), fps=29.97)
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

__all__ = ["detectar_cortes", "UMBRAL_ESCENA", "MIN_SEPARACION"]

# 0,3 es el valor de fabrica del filtro y el control lo respalda: con 0,2, 0,3 y 0,4 encuentra
# los mismos 2 cortes del video de prueba. Mas bajo empieza a disparar con los flashes de las
# camaras de prensa, que en boxeo profesional son constantes.
UMBRAL_ESCENA = 0.3

# Separacion minima entre dos cortes, en segundos.
#
# Una disolvencia o un flash de prensa mantienen el puntaje de escena alto durante varios
# cuadros y el filtro dispara en cada uno. Sobre 20 s de la pelea salieron cuatro cortes en
# los cuadros 195, 197, 199 y 201: es una sola transicion, no cuatro. Ninguna transmision
# tiene dos planos distintos separados por menos de dos decimas de segundo, y los cortes
# rapidos de verdad andan por el medio segundo, asi que agrupar por debajo de 0,2 no pierde
# ninguno real y evita reiniciar el tracker cuatro veces seguidas.
#
# Se encadena: se compara contra la deteccion ANTERIOR, no contra el ultimo corte guardado. Con
# la segunda forma una disolvencia que dispara cada tres cuadros durante medio segundo deja
# escapar uno cada doce, en vez de colapsar a uno. Encadenar agrupa la RACHA entera, que es lo
# que una transicion es, y no corre riesgo de juntar cortes reales porque dos planos distintos
# nunca estan a menos de 0,2 s.
MIN_SEPARACION = 0.2

_PTS = re.compile(r"pts_time:([0-9.]+)")


def detectar_cortes(
    video: Path, *, fps: float, umbral: float = UMBRAL_ESCENA, timeout: int = 3600
) -> list[int]:
    """
    Indices de cuadro donde cambia el plano. Vacio si es una sola toma.

    El filtro reporta el tiempo del cuadro, no su indice, asi que se convierte con los fps.
    El redondeo puede errar por un cuadro; da igual, porque una costura corrida un cuadro
    sigue cayendo dentro del corte y lo que importa es que este declarada.

    Si ffmpeg falla se devuelve una lista vacia en vez de romper el preproceso: quedarse sin
    deteccion de cortes es peor que no preprocesar, pero no mucho peor, y el meta registra
    que no se detectaron.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-i", str(video),
        "-filter:v", f"select='gt(scene,{umbral})',showinfo",
        "-f", "null", "-",
    ]
    try:
        salida = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        ).stderr
    except (subprocess.TimeoutExpired, OSError):
        return []

    frames = sorted({round(float(m) * fps) for m in _PTS.findall(salida)})
    minimo = max(1, round(MIN_SEPARACION * fps))
    salida_f: list[int] = []
    previo: int | None = None
    for f in frames:
        # El cuadro 0 no es un corte, es el principio.
        if f <= 0:
            continue
        if previo is None or f - previo >= minimo:
            salida_f.append(f)
        previo = f
    return salida_f
