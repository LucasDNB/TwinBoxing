"""
BoxTwin - Entrenamiento del detector.

POR QUE EXISTE
  Cuatro cosas que este problema exige y que un loop de entrenamiento generico no hace:

  1. LA PERDIDA SE ENMASCARA. Los cuadros sin pose confiable, los amagues y todo lo que cae
     fuera del tramo anotado no son fondo: son ausencia de dato. Meterlos como O le ensena
     al modelo que la oclusion y el amague son "no hay golpe", que es exactamente donde mas
     golpes hay.

  2. PESOS POR CLASE. O se lleva el 95,31% de los cuadros usables y B el 0,64%. Sin pesos, el
     minimo de la perdida es decir O siempre. Se usa (1/frecuencia)^alpha con alpha 0,5 y no
     el inverso puro: el inverso puro le da a B un peso de 149 contra 1 de O y el gradiente
     queda dominado por un punado de cuadros.

  3. LAS VENTANAS SE SORTEAN DENTRO DEL TRAMO ANOTADO. El tensor de Pacquiao tiene 68.250
     cuadros y solo 5,8% usables, porque de la pelea esta anotado un round de doce. Sorteando
     sobre el largo total, 19 de cada 20 ventanas caerian donde no hay nada que aprender.

  4. LA INFERENCIA VA POR TROZOS CON SOLAPE. Una secuencia de 68.250 cuadros no entra comoda
     en memoria, y cortarla sin solape mete un artefacto cada N cuadros: los bordes de cada
     trozo se deciden con medio campo receptivo vacio. Se descarta el borde y se conserva el
     centro.

  SOBRE LA METRICA. La exactitud por cuadro no sirve: decir siempre O acierta 95,31%. Aca se
  elige el checkpoint por F1 macro sobre O/B/I, que es lo mejor que se puede hacer sin
  decodificar a segmentos. La metrica de verdad -precision y recall POR EVENTO, contra el
  techo humano de 0,911 y 0,903- llega en el bloque 3. Esta es una escalera, no el piso.

QUE HACE
  Estandariza con estadisticas del entrenamiento, sortea ventanas, entrena y evalua.

USO
  est = Estandarizador.ajustar(fuentes_train)
  hist = entrenar(modelo, fuentes_train, fuentes_val, est, cfg)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import nn

from boxtwin_detector.dataset import Fuente

__all__ = [
    "Config", "Estandarizador", "pesos_de_clase", "sortear_ventanas",
    "predecir_secuencia", "metricas", "entrenar", "sembrar",
]

N_CLASES = 3


def sembrar(semilla: int) -> None:
    """
    Fija todo lo que decide un resultado, y HAY QUE LLAMARLA ANTES DE CONSTRUIR EL MODELO.

    Sembrar dentro de `entrenar` no alcanza: para cuando corre, los pesos ya se inicializaron
    con el estado global que hubiera. Medido, dos corridas con la misma semilla daban curvas
    de perdida distintas por eso, y la diferencia en F1 por evento llegaba a 0,1, o sea mas
    que cualquier efecto que se quiera reportar.

    cudnn.deterministic cierra la otra via: por defecto cuDNN elige kernels no deterministas.
    Cuesta algo de velocidad y compra poder repetir un numero, que en un proyecto donde cada
    export lleva su sha256 no es negociable.
    """
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    ventana: int = 256
    batch: int = 32
    ventanas_por_epoca: int = 512
    epocas: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    alpha_pesos: float = 0.5
    canales: int = 64
    dropout: float = 0.1
    semilla: int = 42
    min_usable_en_ventana: float = 0.25
    trozo_inferencia: int = 8192
    # El sobreajuste llega temprano: con sparring-3 en distribucion, la perdida sigue
    # bajando de 0,124 a 0,012 mientras el recall de B cae de 0,57 a 0,16. Se corta.
    paciencia: int = 8


@dataclass
class Estandarizador:
    """
    Media y desvio por feature, calculados SOLO sobre el entrenamiento.

    Calcularlos sobre todo el corpus filtra la validacion a traves de la escala. Es sutil y
    da una mejora chica y falsa, del tipo que despues no aparece en produccion.
    """

    media: np.ndarray
    desvio: np.ndarray

    @classmethod
    def ajustar(cls, fuentes: Sequence[Fuente]) -> "Estandarizador":
        acum = [f.features[f.usable] for f in fuentes]
        x = np.concatenate(acum, axis=0) if acum else np.zeros((0, 1), np.float32)
        if len(x) == 0:
            raise ValueError("no hay cuadros usables para ajustar la estandarizacion")
        media = x.mean(0)
        desvio = x.std(0)
        desvio[desvio < 1e-6] = 1.0     # una feature constante no aporta, pero no puede dividir por cero
        return cls(media.astype(np.float32), desvio.astype(np.float32))

    def aplicar(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.media) / self.desvio).astype(np.float32)

    def a_dict(self) -> dict:
        return {"media": self.media.tolist(), "desvio": self.desvio.tolist()}

    @classmethod
    def de_dict(cls, d: dict) -> "Estandarizador":
        return cls(np.array(d["media"], np.float32), np.array(d["desvio"], np.float32))


def pesos_de_clase(fuentes: Sequence[Fuente], alpha: float = 0.5) -> np.ndarray:
    """
    (1/frecuencia)^alpha, normalizado a media 1.

    Con alpha=1 (inverso puro) B pesa ~149 veces O y el gradiente lo dominan unos pocos
    cuadros; con alpha=0 no hay correccion y el modelo dice O siempre. 0,5 es el compromiso.
    """
    cuenta = np.zeros(N_CLASES, np.float64)
    for f in fuentes:
        for c in range(N_CLASES):
            cuenta[c] += int(((f.labels == c) & f.usable).sum())
    if cuenta.sum() == 0:
        raise ValueError("no hay cuadros usables para calcular los pesos")
    frec = np.maximum(cuenta / cuenta.sum(), 1e-9)
    w = frec ** (-alpha)
    return (w / w.mean()).astype(np.float32)


def sortear_ventanas(
    fuentes: Sequence[Fuente], cfg: Config, rng: np.random.Generator
) -> list[tuple[int, int, int]]:
    """
    (indice de fuente, carril, inicio) de cada ventana, dentro del tramo anotado.

    Se descartan las ventanas con muy pocos cuadros usables: no aportan gradiente y ocupan
    lugar en el batch.
    """
    candidatas: list[tuple[int, int, int, int]] = []   # (fuente, carril, lo, hi)
    for i, f in enumerate(fuentes):
        lo, hi = f.conteos.get("cobertura", [0, f.T - 1])
        hi = min(hi, f.T - 1)
        if hi - lo + 1 < cfg.ventana:
            lo = max(0, min(lo, f.T - cfg.ventana))
            hi = min(f.T - 1, lo + cfg.ventana - 1)
        if hi - lo + 1 < cfg.ventana:
            continue
        for c in range(f.features.shape[0]):
            candidatas.append((i, c, lo, hi - cfg.ventana + 1))
    if not candidatas:
        raise ValueError("ninguna fuente tiene un tramo anotado mas largo que la ventana")

    minimo = int(cfg.min_usable_en_ventana * cfg.ventana)
    salida: list[tuple[int, int, int]] = []
    intentos = 0
    while len(salida) < cfg.ventanas_por_epoca and intentos < cfg.ventanas_por_epoca * 50:
        intentos += 1
        i, c, lo, hi = candidatas[rng.integers(len(candidatas))]
        s = int(rng.integers(lo, hi + 1))
        if fuentes[i].usable[c, s : s + cfg.ventana].sum() >= minimo:
            salida.append((i, c, s))
    if not salida:
        raise ValueError("ninguna ventana sorteada tiene cuadros usables suficientes")
    return salida


@torch.no_grad()
def predecir_secuencia(
    modelo: nn.Module, x: np.ndarray, cfg: Config, device: torch.device
) -> np.ndarray:
    """
    Logits (T, clases) de una secuencia entera, por trozos con solape.

    El solape es medio campo receptivo a cada lado y se descarta: son los cuadros que el
    modelo decidio viendo ceros en vez de video.
    """
    modelo.eval()
    T = x.shape[0]
    margen = getattr(modelo, "campo_receptivo", 63) // 2
    paso = max(1, cfg.trozo_inferencia - 2 * margen)
    out = np.zeros((T, N_CLASES), np.float32)
    inicio = 0
    while inicio < T:
        a = max(0, inicio - margen)
        b = min(T, inicio + paso + margen)
        t = torch.from_numpy(x[a:b].T[None]).to(device)
        logits = modelo(t)[0].T.cpu().numpy()
        util_a, util_b = inicio, min(inicio + paso, T)
        out[util_a:util_b] = logits[util_a - a : util_b - a]
        inicio += paso
    return out


def metricas(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    """F1 macro y recall por clase sobre los cuadros usables. Sin exactitud, a proposito."""
    p, t = pred[mask], y[mask]
    if len(t) == 0:
        return {"f1_macro": 0.0, "n": 0}
    out: dict = {"n": int(len(t))}
    f1s = []
    for c, nombre in enumerate(["O", "B", "I"]):
        tp = int(((p == c) & (t == c)).sum())
        fp = int(((p == c) & (t != c)).sum())
        fn = int(((p != c) & (t == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        out[f"{nombre}_recall"] = round(rec, 4)
        out[f"{nombre}_precision"] = round(prec, 4)
        out[f"{nombre}_n"] = int((t == c).sum())
    out["f1_macro"] = round(float(np.mean(f1s)), 4)
    # el numero que hay que tener a la vista para no festejar de mas
    out["siempre_O"] = round(float((t == 0).mean()), 4)
    return out


def _evaluar(modelo, fuentes, est, cfg, device) -> dict:
    preds, ys, ms = [], [], []
    for f in fuentes:
        for c in range(f.features.shape[0]):
            logits = predecir_secuencia(modelo, est.aplicar(f.features[c]), cfg, device)
            preds.append(logits.argmax(1))
            ys.append(f.labels[c])
            ms.append(f.usable[c])
    return metricas(np.concatenate(preds), np.concatenate(ys), np.concatenate(ms))


def entrenar(
    modelo: nn.Module,
    train: Sequence[Fuente],
    val: Sequence[Fuente],
    est: Estandarizador,
    cfg: Config,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Entrena y devuelve el historial, con el mejor estado por F1 macro de validacion."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Se vuelve a sembrar por si el modelo se construyo aparte; no reemplaza a llamar
    # sembrar() ANTES de crearlo, que es lo que fija la inicializacion de los pesos.
    sembrar(cfg.semilla)
    rng = np.random.default_rng(cfg.semilla)
    modelo = modelo.to(device)

    w = torch.from_numpy(pesos_de_clase(train, cfg.alpha_pesos)).to(device)
    # reduction="none" porque la mascara se aplica cuadro a cuadro, no al batch
    criterio = nn.CrossEntropyLoss(weight=w, reduction="none")
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epocas, 1))

    hist: dict = {"pesos": w.cpu().numpy().tolist(), "epocas": [], "mejor": None}
    mejor_f1, mejor_estado, mejor_epoca = -1.0, None, -1

    for epoca in range(cfg.epocas):
        modelo.train()
        ventanas = sortear_ventanas(train, cfg, rng)
        perdidas = []
        for i in range(0, len(ventanas), cfg.batch):
            lote = ventanas[i : i + cfg.batch]
            xs, ys, ms = [], [], []
            for fi, c, s in lote:
                f = train[fi]
                xs.append(est.aplicar(f.features[c, s : s + cfg.ventana]).T)
                ys.append(f.labels[c, s : s + cfg.ventana])
                ms.append(f.usable[c, s : s + cfg.ventana])
            x = torch.from_numpy(np.stack(xs)).to(device)
            y = torch.from_numpy(np.stack(ys).astype(np.int64)).to(device)
            m = torch.from_numpy(np.stack(ms)).to(device)

            logits = modelo(x)
            perdida_cuadro = criterio(logits, y)
            n = m.sum()
            if n == 0:
                continue
            perdida = (perdida_cuadro * m).sum() / n
            opt.zero_grad(set_to_none=True)
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            opt.step()
            perdidas.append(float(perdida.detach()))
        sched.step()

        m_val = _evaluar(modelo, val, est, cfg, device)
        fila = {"epoca": epoca, "perdida": round(float(np.mean(perdidas)), 4), "val": m_val}
        hist["epocas"].append(fila)
        if m_val["f1_macro"] > mejor_f1:
            mejor_f1, mejor_epoca = m_val["f1_macro"], epoca
            mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
            hist["mejor"] = fila
        if verbose:
            print(f"  epoca {epoca:3d}  perdida {fila['perdida']:.4f}  "
                  f"F1 macro {m_val['f1_macro']:.4f}  "
                  f"recall O/B/I {m_val['O_recall']:.2f}/{m_val['B_recall']:.2f}/"
                  f"{m_val['I_recall']:.2f}")
        if cfg.paciencia and epoca - mejor_epoca >= cfg.paciencia:
            if verbose:
                print(f"  parada temprana: {cfg.paciencia} epocas sin mejorar")
            hist["parada_temprana"] = epoca
            break

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    hist["config"] = asdict(cfg)
    return hist
