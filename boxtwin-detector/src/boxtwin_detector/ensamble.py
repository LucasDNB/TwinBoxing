"""
BoxTwin - Ensamble de semillas.

POR QUE EXISTE
  El desvio entre semillas de una misma configuracion es ~0,10 de F1 por evento, y es IGUAL
  con 62 golpes de validacion que con 380. Si viniera del muestreo de la evaluacion tendria
  que caer como la raiz del tamano, y no cae: la varianza esta en el entrenamiento, no en la
  medicion. Cinco semillas dan cinco modelos genuinamente distintos.

  Medido sobre los cuatro folds cruzados, promediar sus probabilidades gana entre 0,05 y 0,11
  de F1 contra el promedio de las corridas individuales, y en las cuatro:

    fold                     una corrida        ensamble
    sin 02-sparring          0,454 +- 0,107      0,505
    sin Sparring             0,356 +- 0,100      0,421
    sin Pacquiao             0,469 +- 0,098      0,580
    sin sparring-3           0,429 +- 0,104      0,527

  Y gana donde dolia. La precision era el punto debil del detector -0,29 a 0,43 contra 0,903
  del humano- y con el voto unanime queda en 0,85. El desacuerdo entre semillas es
  exactamente donde viven los falsos positivos: los golpes reales sobreviven al voto, las
  marcas espurias son idiosincrasia de cada corrida.

  NO se compara contra la mejor de las cinco semillas, que da 0,638. Ese numero no es
  alcanzable: elegir la mejor semilla mirando la validacion es seleccionar sobre el test.
  La comparacion honesta es contra lo que sale de entrenar una vez con una semilla
  cualquiera, que es lo que uno hace en la practica.

  SE PROMEDIAN PROBABILIDADES, NO LOGITS. Los modelos individuales estan bastante saturados,
  asi que promediar logits lo decidirian los extremos.

  SOBRE EL UMBRAL, Y UN ERROR QUE COSTO UNA MEDICION. Parecia que promediar n modelos
  saturados tenia que dar una salida cuantizada en pasos de 1/n, o sea que los unicos puntos
  de operacion serian "al menos k de n coinciden". Sobre esa teoria se barrieron solo n
  umbrales, y el resultado dio que el ensamble era PEOR que una corrida sola.

  La teoria es falsa. Medido sobre el fold que deja sparring-3 afuera, el reparto de la
  probabilidad promedio es un continuo:

    0,0: 64%   0,1: 8%   0,2: 4%   0,3: 3%   0,4: 2%   0,5: 2%
    0,6:  2%   0,7: 7%   0,8: 6%   0,9: 1%   1,0: 1%

  Los modelos no estan tan saturados como para cuantizar el promedio, y barrer cinco puntos
  se salteaba el optimo. Con grilla fina aparece, y hay un acantilado entre 0,75 y 0,80:

    umbral 0,75   530 marcas   recall 0,429   precision 0,307   F1 0,358
    umbral 0,80   170 marcas   recall 0,382   precision 0,853   F1 0,527

  Trescientas sesenta marcas se caen en un escalon de 0,05, y son casi todas falsos
  positivos. Ahi esta la estructura de voto que la teoria intuia -alrededor de "coinciden
  cuatro de cinco"- pero NO es exacta, asi que el umbral se barre con grilla fina como para
  cualquier otro modelo.

QUE HACE
  Entrena n modelos con n semillas, promedia sus probabilidades y persiste el conjunto.

USO
  ens = entrenar_ensamble(train, val, cfg, [42, 1, 2, 3, 4])
  p = probabilidad(ens, fuente, device)      # una senal por carril
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from boxtwin_detector.dataset import Fuente
from boxtwin_detector.decodificacion import probabilidad_de_golpe
from boxtwin_detector.entrenamiento import (
    Config, Estandarizador, entrenar, predecir_secuencia, sembrar,
)
from boxtwin_detector.modelo import TCN

__all__ = ["Ensamble", "entrenar_ensamble", "probabilidad", "guardar", "cargar"]


@dataclass
class Ensamble:
    """n modelos entrenados con n semillas, y la estandarizacion que comparten."""

    modelos: list[nn.Module]
    estandarizador: Estandarizador
    config: Config
    semillas: list[int]
    historiales: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.modelos:
            raise ValueError("un ensamble sin modelos no es un ensamble")
        if len(self.modelos) != len(self.semillas):
            raise ValueError("hacen falta tantas semillas como modelos")

    @property
    def n(self) -> int:
        return len(self.modelos)

    def a(self, device: torch.device) -> "Ensamble":
        self.modelos = [m.to(device).eval() for m in self.modelos]
        return self


def entrenar_ensamble(
    train: Sequence[Fuente],
    val: Sequence[Fuente],
    cfg: Config,
    semillas: Sequence[int],
    device: torch.device | None = None,
    verbose: bool = True,
) -> Ensamble:
    """
    Entrena un modelo por semilla. La estandarizacion se ajusta UNA vez, sobre el
    entrenamiento, y la comparten los n: son el mismo modelo con distinta inicializacion.
    """
    if not semillas:
        raise ValueError("hace falta al menos una semilla")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    est = Estandarizador.ajustar(train)
    n_features = train[0].features.shape[2]

    modelos, hists = [], []
    for i, s in enumerate(semillas):
        c = Config(**{**asdict(cfg), "semilla": s})
        sembrar(s)                      # antes de construir: fija la inicializacion
        m = TCN(n_features=n_features, canales=c.canales, dropout=c.dropout)
        h = entrenar(m, train, val, est, c, device, verbose=False)
        modelos.append(m.eval())
        hists.append(h)
        if verbose:
            v = h["mejor"]["val"]
            print(f"  semilla {s:3d}: F1 macro por cuadro {v['f1_macro']:.4f} "
                  f"(epoca {h['mejor']['epoca']})")
    return Ensamble(modelos, est, cfg, list(semillas), hists).a(device)


@torch.no_grad()
def probabilidad(ens: Ensamble, f: Fuente, device: torch.device) -> list[np.ndarray]:
    """P(golpe) por carril, promediada sobre los modelos. Es el voto."""
    ens.a(device)
    salida = []
    for c in range(f.features.shape[0]):
        x = ens.estandarizador.aplicar(f.features[c])
        acum = np.zeros(f.T, np.float64)
        for m in ens.modelos:
            acum += probabilidad_de_golpe(predecir_secuencia(m, x, ens.config, device))
        salida.append((acum / ens.n).astype(np.float32))
    return salida


def guardar(ens: Ensamble, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": "boxtwin.detector.ensamble",
        "state_dicts": [{k: v.cpu() for k, v in m.state_dict().items()} for m in ens.modelos],
        "config": asdict(ens.config),
        "estandarizador": ens.estandarizador.a_dict(),
        "semillas": ens.semillas,
        "n_features": ens.modelos[0].n_features,
    }, path)
    return path


def cargar(path: Path, device: torch.device | None = None) -> Ensamble:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(path, map_location="cpu", weights_only=False)
    if d.get("kind") != "boxtwin.detector.ensamble":
        raise ValueError(f"{Path(path).name} no es un ensamble")
    cfg = Config(**d["config"])
    modelos = []
    for sd in d["state_dicts"]:
        m = TCN(n_features=d["n_features"], canales=cfg.canales, dropout=cfg.dropout)
        m.load_state_dict(sd)
        modelos.append(m.eval())
    return Ensamble(modelos, Estandarizador.de_dict(d["estandarizador"]), cfg,
                    list(d["semillas"])).a(device)
