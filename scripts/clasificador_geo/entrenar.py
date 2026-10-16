"""
BoxTwin - Entrenamiento de E4, E4a y E5 dejando una fuente afuera, cinco semillas.

POR QUE EXISTE
  El pre-registro (seccion 2) fija todo lo que hace falta para que E4 se compare con E1 golpe
  contra golpe: siete folds de una fuente afuera, 60 epocas sin early stopping ni validacion
  sobre la fuente de prueba, pesos de clase del fold de entrenamiento, estandarizacion
  ajustada solo en el fold de entrenamiento y semillas 0 a 4. Este script es esa receta y
  ninguna otra; lo que no esta aca no se toca despues de ver numeros (D11).

QUE HACE
  Para cada experimento, fuente y semilla: estandariza con media y desvio por feature sobre
  los pasos usables del fold de entrenamiento, entrena ClasificadorGolpe con Adam (lr 1e-3,
  weight decay 1e-4, batch 64, 60 epocas, CE de tipo + CE de altura con pesos de clase
  inversos a la frecuencia, normalizados a media 1) y aumentacion CropResize (recorte al
  80-100% del largo y remuestreo a 32), y escribe
  <trabajo>/corridas/<exp>/<fuente>/s<semilla>/{pred.npz, config.json, tiempo.json}.
  Reanudable: si pred.npz existe, se saltea.

USO
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/entrenar.py --exp E4 E4a
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/entrenar.py --exp E4 --fuentes Sparring --semillas 0
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/entrenar.py --exp E5 --dispositivo cpu
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import FUENTES, PASOS, REPO, SEMILLAS, TRABAJO  # noqa: E402
from modelo import ClasificadorGolpe  # noqa: E402

HIPER = {"lr": 1e-3, "weight_decay": 1e-4, "batch": 64, "epocas": 60, "crop_min": 0.8, "crop_max": 1.0,
         "optimizador": "Adam", "scheduler": None, "early_stopping": False}

# Columnas de golpes.npz que entran en cada configuracion (pre-registro, seccion 2).
EXPERIMENTOS = {
    "E4": {"columnas": list(range(26)), "con_3d": False, "que": "20 del brazo + 6 del rival"},
    "E4a": {"columnas": list(range(20)), "con_3d": False, "que": "solo las 20 del brazo (ablacion)"},
    "E5": {"columnas": list(range(26)), "con_3d": True, "que": "26 de E4 + 16 features 3D (el diseno dice 18; ver TAREAS.md, decision 9)"},
}


def fijar_semilla(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def pesos_de_clase(y: np.ndarray, n: int) -> np.ndarray:
    """Inversa de la frecuencia en el fold de entrenamiento, normalizada a media 1."""
    cuenta = np.bincount(y, minlength=n).astype(np.float64)
    w = np.where(cuenta > 0, 1.0 / np.maximum(cuenta, 1), 0.0)
    return w / w[cuenta > 0].mean()


def estandarizacion(X: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Media y desvio por feature sobre los pasos con M = 1. Solo se le pasa el fold de entrenamiento."""
    v = X[M.astype(bool)]
    media = v.mean(0)
    desvio = v.std(0)
    desvio = np.where(desvio > 1e-6, desvio, 1.0)
    return media.astype(np.float32), desvio.astype(np.float32)


def crop_resize(x: torch.Tensor, m: torch.Tensor, gen: torch.Generator,
                lo: float = HIPER["crop_min"], hi: float = HIPER["crop_max"]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Recorte aleatorio de la ventana (largo entre lo y hi del total) remuestreado a los mismos
    pasos. Las muestras ya vienen remuestreadas a una grilla uniforme de la ventana, asi que
    recortar sobre esa grilla e interpolar es el mismo recorte sobre la ventana original.
    """
    B, T, _ = x.shape
    largo = (lo + (hi - lo) * torch.rand(B, generator=gen)) * (T - 1)
    inicio = torch.rand(B, generator=gen) * (T - 1 - largo)
    pos = inicio[:, None] + largo[:, None] * torch.linspace(0, 1, T)[None, :]       # (B, T)
    i0 = pos.floor().long().clamp(0, T - 1)
    i1 = (i0 + 1).clamp(max=T - 1)
    a = (pos - i0.to(pos.dtype))[..., None]
    g = lambda t, i: torch.gather(t, 1, i[..., None].expand(-1, -1, t.shape[-1]))  # noqa: E731
    xn = g(x, i0) * (1 - a) + g(x, i1) * a
    mn = torch.gather(m, 1, pos.round().long().clamp(0, T - 1))
    return xn, mn


def entrenar_modelo(X: np.ndarray, M: np.ndarray, yt: np.ndarray, ya: np.ndarray, idx_train: np.ndarray,
                    semilla: int, dispositivo: str = "cpu", epocas: int = HIPER["epocas"]):
    """
    Entrena con las muestras idx_train y nada mas. Devuelve el modelo en modo eval y la
    estandarizacion. Las muestras fuera de idx_train no se leen.
    """
    fijar_semilla(semilla)
    Xtr, Mtr = X[idx_train], M[idx_train].astype(bool)
    media, desvio = estandarizacion(Xtr, Mtr)
    Xtr = (Xtr - media) / desvio
    ytt, yat = yt[idx_train], ya[idx_train]
    dev = torch.device(dispositivo)
    modelo = ClasificadorGolpe(X.shape[-1]).to(dev)
    opt = torch.optim.Adam(modelo.parameters(), lr=HIPER["lr"], weight_decay=HIPER["weight_decay"])
    ce_t = nn.CrossEntropyLoss(weight=torch.tensor(pesos_de_clase(ytt, 3), dtype=torch.float32, device=dev))
    ce_a = nn.CrossEntropyLoss(weight=torch.tensor(pesos_de_clase(yat, 2), dtype=torch.float32, device=dev))
    xs = torch.tensor(Xtr, dtype=torch.float32)
    ms = torch.tensor(Mtr, dtype=torch.float32)
    tt, ta = torch.tensor(ytt), torch.tensor(yat)
    gen = torch.Generator().manual_seed(semilla)
    n = len(xs)
    modelo.train()
    for _ in range(epocas):
        orden = torch.randperm(n, generator=gen)
        for i in range(0, n, HIPER["batch"]):
            b = orden[i:i + HIPER["batch"]]
            if len(b) < 2:                       # BatchNorm no entrena con una muestra
                continue
            xb, mb = crop_resize(xs[b], ms[b], gen)
            lt, la = modelo(xb.to(dev), mb.to(dev))
            perdida = ce_t(lt, tt[b].to(dev)) + ce_a(la, ta[b].to(dev))
            opt.zero_grad()
            perdida.backward()
            opt.step()
    modelo.eval()
    return modelo, media, desvio


@torch.no_grad()
def predecir(modelo, X: np.ndarray, M: np.ndarray, media, desvio, dispositivo: str = "cpu"):
    dev = torch.device(dispositivo)
    x = torch.tensor((X - media) / desvio, dtype=torch.float32, device=dev)
    m = torch.tensor(M.astype(bool), dtype=torch.float32, device=dev)
    lt, la = modelo(x, m)
    return torch.softmax(lt, -1).cpu().numpy(), torch.softmax(la, -1).cpu().numpy()


def cargar_entrada(exp: str, trabajo: Path = None) -> dict:
    """X con las columnas del experimento, M, etiquetas, frame_dir y fuente, en el orden de golpes.npz."""
    trabajo = trabajo or TRABAJO
    with np.load(trabajo / "datos" / "golpes.npz") as z:
        d = {k: z[k] for k in z.files}
    nombres = json.loads((trabajo / "datos" / "golpes2d.meta.json").read_text())["features"]
    cfg = EXPERIMENTOS[exp]
    X = d["X"][..., cfg["columnas"]]
    nombres = [nombres[i] for i in cfg["columnas"]]
    M = d["M"].astype(bool)
    sel = np.ones(len(X), bool)
    if cfg["con_3d"]:
        with np.load(trabajo / "datos" / "features3d.npz") as z:
            f3 = {k: z[k] for k in z.files}
        pos = {n: i for i, n in enumerate(f3["frame_dir"])}
        sel = np.array([n in pos for n in d["frame_dir"]])
        idx = np.array([pos.get(n, 0) for n in d["frame_dir"]])
        X3 = np.where(sel[:, None, None], f3["X3"][idx], 0.0).astype(np.float32)
        X = np.concatenate([X, X3], -1)
        nombres = nombres + json.loads((trabajo / "datos" / "features3d.meta.json").read_text())["features"]
    return {"X": X[sel], "M": M[sel], "y_tipo": d["y_tipo"][sel], "y_altura": d["y_altura"][sel],
            "frame_dir": d["frame_dir"][sel], "fuente": d["fuente"][sel], "nombres": nombres}


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def gpu_libre() -> bool:
    """D5: la GPU se usa solo si no hay otro proceso de computo en ella."""
    if not torch.cuda.is_available():
        return False
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and not r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return False


def correr(exp: str, fuente: str, semilla: int, d: dict, dispositivo: str, trabajo: Path = None) -> Path:
    trabajo = trabajo or TRABAJO
    salida = trabajo / "corridas" / exp / fuente / f"s{semilla}"
    if (salida / "pred.npz").exists():
        return salida
    salida.mkdir(parents=True, exist_ok=True)
    test = d["fuente"] == fuente
    idx_train, idx_test = np.flatnonzero(~test), np.flatnonzero(test)
    t0 = time.time()
    modelo, media, desvio = entrenar_modelo(d["X"], d["M"], d["y_tipo"], d["y_altura"], idx_train,
                                            semilla, dispositivo)
    t1 = time.time()
    pt, pa = predecir(modelo, d["X"][idx_test], d["M"][idx_test], media, desvio, dispositivo)
    np.savez(salida / "pred.npz", frame_dir=d["frame_dir"][idx_test], p_tipo=pt, p_altura=pa)
    (salida / "config.json").write_text(json.dumps({
        "exp": exp, "que": EXPERIMENTOS[exp]["que"], "fuente_afuera": fuente, "semilla": semilla,
        "hiper": HIPER, "pasos": PASOS, "features": d["nombres"], "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)), "dispositivo": dispositivo, "commit": _commit(),
        "pesos_tipo": pesos_de_clase(d["y_tipo"][idx_train], 3).round(4).tolist(),
        "pesos_altura": pesos_de_clase(d["y_altura"][idx_train], 2).round(4).tolist(),
    }, indent=2) + "\n")
    (salida / "tiempo.json").write_text(json.dumps(
        {"entrenamiento_s": round(t1 - t0, 2), "prueba_s": round(time.time() - t1, 2)}) + "\n")
    return salida


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exp", nargs="+", default=["E4", "E4a"], choices=list(EXPERIMENTOS))
    ap.add_argument("--fuentes", nargs="*", default=FUENTES)
    ap.add_argument("--semillas", nargs="*", type=int, default=list(SEMILLAS))
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args(argv)
    dispositivo = args.dispositivo
    if dispositivo == "auto":
        dispositivo = "cuda" if gpu_libre() else "cpu"
    print(f"dispositivo: {dispositivo}", flush=True)
    for exp in args.exp:
        d = cargar_entrada(exp)
        for fuente in args.fuentes:
            for s in args.semillas:
                t0 = time.time()
                out = correr(exp, fuente, s, d, dispositivo)
                print(f"{exp}/{fuente}/s{s}: {time.time() - t0:.1f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
