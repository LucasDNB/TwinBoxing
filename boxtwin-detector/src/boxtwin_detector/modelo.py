"""
BoxTwin - Red temporal del detector.

POR QUE EXISTE
  La decision sobre un cuadro depende de cuadros que todavia no pasaron. Se ve en los datos:
  en el golpe del cuadro 161 de sparring-3, los primeros cuatro cuadros ya etiquetados son
  indistinguibles de la guardia de los tres anteriores. La `B` esta donde el anotador dijo
  que arranca el movimiento, no donde el movimiento se hace evidente, y eso recien pasa en
  el cuadro 166. Ningun clasificador cuadro a cuadro puede acertar ahi.

  De ahi salen las dos decisiones de arquitectura:

  - CONVOLUCIONES DILATADAS, para llegar a un campo receptivo de 63 cuadros (2,1 s) con
    cinco capas. Es ~9 veces la duracion mediana de un golpe, que son 7 cuadros.
  - NO CAUSAL. El detector corre sobre video grabado, no en vivo, asi que cada cuadro se
    decide viendo su pasado Y su futuro. Hacerlo causal costaria exactitud a cambio de una
    propiedad que este problema no necesita.

  Es una TCN de una sola etapa, y es a proposito. El refinamiento multi-etapa es lo que se
  prueba DESPUES de saber que la de una etapa no alcanza; empezar por el grande deja sin
  saber cual de las dos cosas aporto.

  Sobre los heatmaps de PoseConv3D: no van. Ese modelo clasifica un clip recortado y trabaja
  en pixeles, que es por donde se cuela la camara. Aca la entrada son 20 features ya
  normalizadas y el modelo tiene que emitir una salida POR CUADRO sobre 17.000.

QUE HACE
  (B, F, T) -> (B, 3, T). Tres clases: O, B, I.

USO
  modelo = TCN(n_features=20)
  logits = modelo(x)          # x: (batch, features, tiempo)
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["TCN", "DILATACIONES", "campo_receptivo"]

# 1 + 2*(1+2+4+8+16) = 63 cuadros = 2,1 s a 30 fps
DILATACIONES = (1, 2, 4, 8, 16)


def campo_receptivo(dilataciones=DILATACIONES, kernel: int = 3) -> int:
    """Cuantos cuadros de entrada ve cada cuadro de salida."""
    return 1 + sum((kernel - 1) * d for d in dilataciones)


class BloqueDilatado(nn.Module):
    """Convolucion dilatada con residual. El residual es lo que deja apilar sin que muera."""

    def __init__(self, canales: int, dilatacion: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        # padding = dilatacion * (kernel-1) / 2 conserva el largo, mirando a los dos lados
        pad = dilatacion * (kernel - 1) // 2
        self.conv = nn.Conv1d(canales, canales, kernel, padding=pad, dilation=dilatacion)
        self.norm = nn.BatchNorm1d(canales)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.act(self.norm(self.conv(x))))


class TCN(nn.Module):
    """Red temporal de convoluciones dilatadas, no causal, salida por cuadro."""

    def __init__(
        self,
        n_features: int,
        canales: int = 64,
        dilataciones=DILATACIONES,
        n_clases: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.dilataciones = tuple(dilataciones)
        self.entrada = nn.Conv1d(n_features, canales, 1)
        self.bloques = nn.ModuleList(
            BloqueDilatado(canales, d, dropout=dropout) for d in self.dilataciones
        )
        self.salida = nn.Conv1d(canales, n_clases, 1)

    @property
    def campo_receptivo(self) -> int:
        return campo_receptivo(self.dilataciones)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"se esperaba (batch, features, tiempo), llego {tuple(x.shape)}")
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"el modelo espera {self.n_features} features, llegaron {x.shape[1]}"
            )
        h = self.entrada(x)
        for b in self.bloques:
            h = b(h)
        return self.salida(h)
