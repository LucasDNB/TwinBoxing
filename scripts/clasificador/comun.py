"""
BoxTwin - Piezas comunes de la campana de clasificador por tipo y altura (14-10-2026).

POR QUE EXISTE
  Seis scripts de la campana necesitan la misma lista de fuentes, los mismos nombres de
  clase y las mismas metricas. Si cada uno las define por su cuenta, el dia que difieren el
  numero de la tesis sale de una version que nadie reviso. Este modulo es la unica copia.
  Las metricas viven aca y no en numpy suelto porque son las que van al Cap 7 y tienen test.

QUE HACE
  Fuentes pre-registradas, rutas del directorio de trabajo, nombres de clase y su colapso a
  familia y altura, y las metricas: exactitud, exactitud balanceada, recall por clase,
  kappa de Cohen e intervalo de Wilson al 95%.

USO
  from comun import FUENTES, wilson, balanceada
  No corre solo. Se importa desde los otros scripts de scripts/clasificador/.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path

# Raiz del repo: scripts/clasificador/comun.py -> ../../
REPO = Path(__file__).resolve().parents[2]

# Directorio de trabajo fuera del repo, como las mediciones de C8. Nada pesado entra a git.
TRABAJO = Path(os.environ.get("BOXTWIN_CLASIFICADOR", "~/boxtwin/clasificador-2026-10")).expanduser()

# Las siete fuentes anotadas, en el orden en que se reportan. Clave: nombre de la fuente
# (el stem del video, que es lo que usa el exportador en frame_dir). Valor: directorio del
# proyecto de anotacion. Pre-registrado: no se agregan ni se sacan fuentes despues de medir.
FUENTES: dict[str, str] = {
    "Sparring": "anotacion",
    "01-sparring": "anotacion-spar-01",
    "02-sparring": "anotacion-spar-02",
    "03-sparring": "anotacion-spar-03",
    "04-sparring": "anotacion-spar-04",
    "sparring-3-rounds": "anotacion-sparring-3",
    "pacquiao_margarito": "anotacion-pacquiao",
}

# Las cuatro fuentes donde se midio C8 (13-10). La altura del clasificador se compara con la
# regla sobre estas, y se dice que el denominador no es el mismo (ver spec, seccion 6).
FUENTES_C8 = ("03-sparring", "04-sparring", "sparring-3-rounds", "pacquiao_margarito")

# Las tres configuraciones pre-registradas (spec, seccion 4). Una sola cosa cambia de una a la
# siguiente: E1 -> E2 cambia el espacio de clases, E2 -> E3 agrega el rival.
EXPERIMENTOS: dict[str, dict] = {
    "E1": {"variante": "v6lr_at", "remap": None,
           "que": "replica de produccion: 6 clases tipo x rol (lead-rear), solo atacante"},
    "E2": {"variante": "v12s_at", "remap": "tipo_altura",
           "que": "6 clases tipo x altura, sin lado, solo atacante"},
    "E3": {"variante": "v12s_amb", "remap": "tipo_altura",
           "que": "6 clases tipo x altura, sin lado, atacante y rival"},
}
SEMILLAS = (0, 1, 2)

TIPOS = ("straight", "hook", "uppercut")
ALTURAS = ("head", "body")

# Brazos en COCO-17: hombro, codo, muneca.
BRAZO_COCO = {"left": (5, 7, 9), "right": (6, 8, 10)}
MIN_SCORE = 0.3


def ruta_annot(fuente: str) -> Path:
    return REPO / FUENTES[fuente] / "annotations" / f"{fuente}.annot.json"


# ---------------------------------------------------------------------------
# Nombres de clase
# ---------------------------------------------------------------------------


def partes_de_clase(nombre: str) -> dict[str, str | None]:
    """
    Descompone un nombre de clase del exportador ('hook-left-body', 'straight-lead',
    'uppercut-head') en tipo, eje (lado o rol) y altura. Lo que no esta queda en None.
    """
    trozos = nombre.split("-")
    if trozos[0] not in TIPOS:
        raise ValueError(f"clase sin tipo reconocible: {nombre}")
    tipo = trozos[0]
    altura = trozos[-1] if trozos[-1] in ALTURAS else None
    medio = [t for t in trozos[1:] if t not in ALTURAS]
    eje = medio[0] if medio else None
    return {"tipo": tipo, "eje": eje, "altura": altura}


def clase_tipo_altura(nombre: str) -> str:
    """'hook-left-body' -> 'hook-body'. Es el espacio sin lado de E2 y E3."""
    p = partes_de_clase(nombre)
    if p["altura"] is None:
        raise ValueError(f"la clase {nombre} no tiene altura")
    return f"{p['tipo']}-{p['altura']}"


CLASES_TIPO_ALTURA = [f"{t}-{a}" for t in TIPOS for a in ALTURAS]

# Lectura de 8 clases de E2 y E3 (spec, seccion 6, metricas, punto 4): el recto partido en
# jab y cross.
CLASES_OCHO = [f"{t}-{a}" for t in ("jab", "cross", "hook", "uppercut") for a in ALTURAS]


def clase_ocho(nombre: str, rol: str) -> str:
    """
    'straight-head' con rol 'lead' -> 'jab-head'. El rol no sale del modelo de E2 o E3, que no
    tiene lado: sale de la etiqueta de E1 del mismo golpe. Hook y uppercut no lo usan.
    """
    p = partes_de_clase(nombre)
    if p["altura"] is None or p["eje"] is not None:
        raise ValueError(f"la clase {nombre} no es tipo x altura sin lado")
    tipo = p["tipo"]
    if tipo == "straight":
        if rol not in ("lead", "rear"):
            raise ValueError(f"rol invalido para partir el recto: {rol}")
        tipo = "jab" if rol == "lead" else "cross"
    return f"{tipo}-{p['altura']}"


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------


def wilson(aciertos: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    """Intervalo de Wilson al 95%. None si n es cero."""
    if n == 0:
        return None
    p = aciertos / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    medio = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(centro - medio, 4), round(centro + medio, 4))


def exactitud(real: list, pred: list) -> float | None:
    if not real:
        return None
    return sum(1 for r, p in zip(real, pred) if r == p) / len(real)


def recall_por_clase(real: list, pred: list, clases) -> dict:
    salida = {}
    for c in clases:
        n = sum(1 for r in real if r == c)
        a = sum(1 for r, p in zip(real, pred) if r == c and p == c)
        salida[c] = {"n": n, "aciertos": a, "recall": (a / n if n else None), "wilson95": wilson(a, n)}
    return salida


def balanceada(real: list, pred: list, clases) -> float | None:
    """Promedio de recall sobre las clases presentes. Una prediccion fuera de clases es error."""
    rec = [v["recall"] for v in recall_por_clase(real, pred, clases).values() if v["n"]]
    return sum(rec) / len(rec) if rec else None


def kappa(real: list, pred: list) -> float | None:
    """Kappa de Cohen entre dos etiquetadores. Las abstenciones cuentan como una categoria mas."""
    n = len(real)
    if n == 0:
        return None
    po = exactitud(real, pred)
    cr, cp = Counter(real), Counter(pred)
    pe = sum(cr[k] * cp.get(k, 0) for k in cr) / (n * n)
    if pe == 1:
        return None
    return (po - pe) / (1 - pe)


def confusion(real: list, pred: list) -> dict:
    tabla: dict = {}
    for r, p in zip(real, pred):
        tabla.setdefault(str(r), Counter())[str(p)] += 1
    return {k: dict(v) for k, v in sorted(tabla.items())}


def redondear(x, nd: int = 4):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: redondear(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(redondear(v, nd) for v in x)
    return x
