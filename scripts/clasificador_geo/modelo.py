"""
BoxTwin - ClasificadorGolpe: red temporal chica con dos cabezas (tipo y altura) para E4 y E5.

POR QUE EXISTE
  El detector de golpes, una TCN chica sobre features geometricas, transfiere a fuentes nuevas;
  PoseConv3D, con millones de parametros sobre mapas de calor, no. E4 pone un modelo del mismo
  orden de tamano que el detector sobre esas features. La arquitectura esta fijada en el diseno
  (02-diseno.md, seccion 3) antes de medir y no se ajusta mirando resultados (D11).

QUE HACE
  Conv1d(F, 64, 1), tres bloques residuales Conv1d(64, 64, k=3, dilation d) + BatchNorm +
  ReLU + Dropout(0.2) con d = 1, 2, 4, pooling enmascarado (promedio y maximo sobre los pasos
  con M = 1, concatenados: 128) y dos cabezas lineales, tipo (3) y altura (2). Los pasos con
  M = 0 entran en cero, asi su contenido no puede llegar a la salida por las convoluciones.

USO
  from modelo import ClasificadorGolpe
  logits_tipo, logits_altura = ClasificadorGolpe(26)(x, m)   # x (B, 32, F), m (B, 32)
"""

from __future__ import annotations

import torch
from torch import nn

ANCHO = 64
DILATACIONES = (1, 2, 4)
DROPOUT = 0.2


class Bloque(nn.Module):
    def __init__(self, ancho: int, dilatacion: int):
        super().__init__()
        self.conv = nn.Conv1d(ancho, ancho, kernel_size=3, dilation=dilatacion, padding=dilatacion)
        self.bn = nn.BatchNorm1d(ancho)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(torch.relu(self.bn(self.conv(x))))


class ClasificadorGolpe(nn.Module):
    def __init__(self, n_features: int, n_tipos: int = 3, n_alturas: int = 2):
        super().__init__()
        self.entrada = nn.Conv1d(n_features, ANCHO, 1)
        self.bloques = nn.Sequential(*[Bloque(ANCHO, d) for d in DILATACIONES])
        self.tipo = nn.Linear(2 * ANCHO, n_tipos)
        self.altura = nn.Linear(2 * ANCHO, n_alturas)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mf = m.to(x.dtype)                                   # (B, T)
        h = (x * mf[..., None]).transpose(1, 2)              # (B, F, T)
        h = self.bloques(self.entrada(h))                    # (B, C, T)
        w = mf[:, None, :]
        n = w.sum(-1).clamp(min=1.0)
        prom = (h * w).sum(-1) / n
        maxi = h.masked_fill(w == 0, float("-inf")).amax(-1)
        maxi = torch.where(torch.isfinite(maxi), maxi, torch.zeros_like(maxi))   # golpe sin pasos usables
        z = torch.cat([prom, maxi], -1)
        return self.tipo(z), self.altura(z)
