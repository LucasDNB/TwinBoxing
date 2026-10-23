"""
BoxTwin - Piezas comunes de la prueba academica con el modelo de BoxMind (BM, 22-10-2026).

POR QUE EXISTE
  La prueba usa codigo y datos de terceros que no traen licencia (BoxingWeb, commit e06add2,
  sin archivo LICENSE). Las condiciones de uso fijadas en el pre-registro son tres: nada de
  ellos entra a git (este repositorio es publico), nada entrenado con ellos llega al producto,
  y su codigo se usa desde un clon externo, sin copiarlo aca. Este modulo concentra las rutas
  para que esas condiciones se puedan chequear en un solo lugar.

QUE HACE
  Rutas del trabajo, del clon de BoxMind y de los datos; la guarda `fuera_del_repo`; la
  verificacion del commit del clon; las constantes pre-registradas (hiperparametros del paper,
  filtro de largo de BoxingWeb, umbral de identidad, fracciones de la curva).

USO
  from bm import TRABAJO_BM, BOXMIND_REPO, fuera_del_repo
  No corre solo.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[1]
# geo trae FUENTES, TIPOS, ALTURAS, PAD y la ruta del trabajo de E4 (para comparar contra E4)
sys.path.append(str(REPO / "scripts" / "clasificador_geo"))
import geo  # noqa: E402

# Trabajo de esta prueba. Fuera del repo: datos, poses, corridas y pesos de BM viven aca.
TRABAJO_BM = Path(os.environ.get("BOXTWIN_BM", "~/boxtwin/boxmind-prueba-2026-10")).expanduser()
# Clon de github.com/gouba2333/BoxingWeb. Se usa desde ahi; su codigo no se copia al repo.
BOXMIND_REPO = Path(os.environ.get("BOXTWIN_BOXMIND_REPO", "~/externos/BoxingWeb")).expanduser()
BOXMIND_COMMIT = "e06add2"
# Datos de Tsinghua Cloud descomprimidos: <BOXINGWEB>/data_train/<ronda>/... y data_test/...
BOXINGWEB = Path(os.environ.get("BOXTWIN_BOXINGWEB", "~/externos/boxingweb-datos/boxingweb")).expanduser()

# Pre-registro, seccion 4.2. Hiperparametros del paper (material suplementario B.4.1) para el
# modelo de atributos: SGD lr 0,05, 12 epocas, lr / 10 en la 5 y en la 10, momentum 0,9,
# weight decay 1e-4, batch 64. Alphas de la focal loss tal como estan en train.py@e06add2.
HIPER = {
    "modelo": "PoseOnly+TCN+MoE", "lr": 0.05, "epocas": 12, "hitos": (5, 10), "gamma_lr": 0.1,
    "momentum": 0.9, "weight_decay": 1e-4, "batch": 64, "dropout": 0.5, "expertos": 4,
    "focal_gamma": 2.0,
    "alpha_tecnica": (0.3, 0.3, 0.4), "alpha_objetivo": (0.3, 0.4, 0.3),
    "alpha_distancia": (0.3, 0.3, 0.4), "alpha_efecto": (0.3, 0.7), "peso_efecto": 2.0,
    "espejo_p": 0.5,
}
SEMILLAS = (0, 1, 2, 3, 4)

# Filtro de BoxingWeb que aplica dataset_youtube.py: golpes de 4 a 30 cuadros.
LARGO_MIN, LARGO_MAX = 4, 30
# Emparejado de identidad (seccion 4.1, B3): distancia media de hombros, codos y caderas
# menor a TAU_ID anchos de hombro de la referencia; golpe de BoxingWeb usable si el atacante
# queda emparejado en al menos COBERTURA_MIN de sus cuadros.
TAU_ID = 0.5
COBERTURA_MIN = 0.5
# Curva de aprendizaje (seccion 4.3): fraccion de rondas de BoxingWeb, sorteadas con
# semilla 0 una sola vez.
FRACCIONES = (0.25, 0.5, 1.0)

EXPERIMENTOS = {
    "BM-0b": "BoxingWeb train -> BoxingWeb test, poses re-estimadas con el pipeline de BoxTwin",
    "BM-1": "BoxingWeb completo -> los 1047 golpes de BoxTwin, sin ver BoxTwin",
    "BM-1b": "solo BoxTwin, una fuente afuera (arquitectura de BoxMind sobre los datos de E4)",
    "BM-2": "BoxTwin (seis fuentes) + BoxingWeb completo, una fuente afuera",
    "BM-2-25": "como BM-2 con el 25% de las rondas de BoxingWeb",
    "BM-2-50": "como BM-2 con el 50% de las rondas de BoxingWeb",
}


def fuera_del_repo(ruta: Path) -> Path:
    """Frena si una salida de la prueba cae dentro del repositorio (que es publico)."""
    r = Path(ruta).expanduser().resolve()
    try:
        r.relative_to(REPO.resolve())
    except ValueError:
        return r
    raise SystemExit(f"{r} esta dentro del repositorio: los datos de BM no entran a git")


def verificar_clon(ruta: Path = None) -> str:
    """Devuelve el commit del clon de BoxMind y frena si no es el pre-registrado."""
    ruta = Path(ruta or BOXMIND_REPO)
    if not (ruta / "model" / "mfusion.py").is_file():
        raise SystemExit(f"no encuentro el clon de BoxMind en {ruta} (git clone https://github.com/gouba2333/BoxingWeb)")
    r = subprocess.run(["git", "-C", str(ruta), "rev-parse", "HEAD"], capture_output=True, text=True)
    head = r.stdout.strip()
    if not head.startswith(BOXMIND_COMMIT):
        raise SystemExit(f"el clon esta en {head[:7]}, el pre-registro fija {BOXMIND_COMMIT}")
    return head
