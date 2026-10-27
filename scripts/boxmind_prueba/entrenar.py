"""
BoxTwin - Entrena el modelo de atributos de BoxMind (PoseOnly+TCN+MoE) para la prueba BM.

POR QUE EXISTE
  El repositorio de BoxMind publica la arquitectura (model/mfusion.py) y un train.py atado a
  sus datos, sin pesos entrenados. Para usar "su modelo" sobre BoxTwin hay que entrenarlo. La
  arquitectura se importa del clon externo sin tocarla; el bucle se escribe aca porque el de
  train.py evalua contra el test en cada epoca y solo sabe leer BoxingWeb. Lo que fija el
  paper (B.4.1) y lo que fija train.py@e06add2 se respeta tal cual (bm.HIPER); las diferencias
  estan en el pre-registro, seccion 4.

QUE HACE
  Por experimento, fold y semilla: arma entrenamiento y prueba (ver bm.EXPERIMENTOS), entrena
  12 epocas con SGD, toma la ultima epoca (sin validacion ni seleccion de epoca) y escribe
  <trabajo_bm>/corridas/<exp>/<fuente>/s<semilla>/{pred.npz, config.json, tiempo.json}, con
  el mismo formato que E4 para que la evaluacion reutilice scripts/clasificador_geo/evaluar.py (evaluar_bm.py).
  Las perdidas de distancia y efecto solo cuentan en los golpes de BoxingWeb (BoxTwin no las
  tiene anotadas). Reanudable: si pred.npz existe, se saltea.

USO (en twinboxing_env)
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/boxmind_prueba/entrenar.py --exp BM-1b
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/boxmind_prueba/entrenar.py --exp BM-2 --semillas 0 1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bm import (BOXMIND_REPO, EXPERIMENTOS, FRACCIONES, HIPER, SEMILLAS, TRABAJO_BM,  # noqa: E402
                fuera_del_repo, geo, verificar_clon)
from representacion import espejar  # noqa: E402


# ---------------------------------------------------------------------------
# Particiones (sin torch: se prueban solas)
# ---------------------------------------------------------------------------


TODAS = "__todas__"


def rondas_de_fraccion(rondas: list[str], fraccion: float, semilla: int = 0) -> list[str]:
    """Sorteo unico de rondas de BoxingWeb para la curva (pre-registro, seccion 4.3)."""
    orden = sorted(set(rondas))
    if fraccion >= 1.0:
        return orden
    rng = np.random.default_rng(semilla)
    elegidas = rng.permutation(len(orden))[:math.ceil(fraccion * len(orden))]
    return sorted(orden[i] for i in elegidas)


def particiones(exp: str, bt: dict, bw: dict | None) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    [(nombre_de_prueba, idx_entrenamiento_bt, idx_entrenamiento_bw, idx_prueba)]. La prueba
    indexa BoxTwin salvo en BM-0b, que indexa BoxingWeb.
    """
    n_bt = len(bt["frame_dir"])
    vacio = np.array([], int)
    if exp == "BM-0b":
        tr = np.flatnonzero(bw["particion"] == "train")
        te = np.flatnonzero(bw["particion"] == "test")
        return [("boxingweb", vacio, tr, te)]
    if exp == "BM-1":       # un solo modelo por semilla; las predicciones se guardan por fuente
        return [(TODAS, vacio, np.arange(len(bw["frame_dir"])), np.arange(n_bt))]
    if exp == "BM-1b":
        idx_bw = vacio
    elif exp.startswith("BM-2"):
        frac = 1.0 if exp == "BM-2" else int(exp.split("-")[-1]) / 100
        if frac not in FRACCIONES:
            raise SystemExit(f"{exp}: la fraccion {frac} no esta pre-registrada")
        rondas = rondas_de_fraccion(list(bw["fuente"]), frac)
        idx_bw = np.flatnonzero(np.isin(bw["fuente"], rondas))
    else:
        raise SystemExit(f"experimento desconocido: {exp}")
    salida = []
    for f in geo.FUENTES:
        te = np.flatnonzero(bt["fuente"] == f)
        tr = np.setdiff1d(np.arange(n_bt), te)
        salida.append((f, tr, idx_bw, te))
    return salida


def p_altura_de(p_obj: np.ndarray) -> np.ndarray:
    """Cabeza de objetivo de 3 salidas (cabeza, pecho, abdomen) con etiquetas 0/1 -> (cabeza, cuerpo)."""
    pa = np.stack([p_obj[:, 0], p_obj[:, 1:].sum(1)], 1)
    return pa / pa.sum(1, keepdims=True)


# ---------------------------------------------------------------------------
# Modelo y perdidas (torch)
# ---------------------------------------------------------------------------


def construir_modelo():
    sys.path.insert(0, str(BOXMIND_REPO))
    from model.mfusion import MFusion
    args = SimpleNamespace(model=HIPER["modelo"], num_experts=HIPER["expertos"],
                           dropout=HIPER["dropout"], router_hidden_dim=None)
    return MFusion(args)


def focal_multiclase(probs, y, alpha, gamma):
    """MulticlassFocalLoss de train.py: probabilidades, -1 se ignora."""
    import torch
    m = y != -1
    if not bool(m.any()):
        return probs.sum() * 0.0
    p, y = probs[m], y[m]
    pt = p.gather(1, y[:, None]).squeeze(1).clamp_min(1e-12)
    a = torch.as_tensor(alpha, device=p.device, dtype=p.dtype)[y]
    return (-a * (1 - pt) ** gamma * pt.log()).mean()


def focal_binaria(prob, y, alpha, gamma):
    """BinaryFocalLoss de train.py, con -1 ignorado (BoxTwin no tiene efecto anotado)."""
    import torch
    m = y != -1
    if not bool(m.any()):
        return prob.sum() * 0.0
    p, y = prob.reshape(-1)[m].clamp(1e-7, 1 - 1e-7), y[m].float()
    bce = torch.nn.functional.binary_cross_entropy(p, y, reduction="none")
    pt = torch.exp(-bce)
    a = torch.as_tensor(alpha, device=p.device, dtype=p.dtype)[y.long()]
    return (a * (1 - pt) ** gamma * bce).mean()


def perdida(salidas, yt, yo, yd, ye):
    tech, tgt, dis, eff, _ = salidas
    g = HIPER["focal_gamma"]
    return (focal_multiclase(tech, yt, HIPER["alpha_tecnica"], g)
            + focal_multiclase(tgt, yo, HIPER["alpha_objetivo"], g)
            + focal_multiclase(dis, yd, HIPER["alpha_distancia"], g)
            + HIPER["peso_efecto"] * focal_binaria(eff, ye, HIPER["alpha_efecto"], g))


def lote(X, L, idx, rng=None):
    """Relleno con ceros hasta el largo maximo del lote (video_collate_fn) y espejo opcional."""
    n = int(L[idx].max())
    x = X[idx, :n].copy()
    if rng is not None:
        for i, j in enumerate(idx):
            if rng.random() < HIPER["espejo_p"]:
                x[i, :L[j]] = espejar(x[i, :L[j]])
    return x


def entrenar_y_predecir(Xtr, Ltr, Y, Xte, Lte, semilla: int, dispositivo: str):
    import torch
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(semilla)
    modelo = construir_modelo().to(dispositivo)
    opt = torch.optim.SGD(modelo.parameters(), lr=HIPER["lr"], momentum=HIPER["momentum"],
                          weight_decay=HIPER["weight_decay"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, list(HIPER["hitos"]), gamma=HIPER["gamma_lr"])
    t = lambda a: torch.as_tensor(a, device=dispositivo)  # noqa: E731
    sin_mano = None
    for _ in range(HIPER["epocas"]):
        modelo.train()
        orden = rng.permutation(len(Xtr))
        for k in range(0, len(orden), HIPER["batch"]):
            idx = orden[k:k + HIPER["batch"]]
            if len(idx) < 2:        # BatchNorm no entrena con un solo ejemplo
                continue
            x = t(lote(Xtr, Ltr, idx, rng)).float()
            sin_mano = torch.zeros((len(idx), 1), device=dispositivo)
            out = modelo(None, x, sin_mano)
            loss = perdida(out, *(t(Y[c][idx]).long() for c in ("y_tipo", "y_altura", "y_distancia", "y_efecto")))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    modelo.eval()
    pt, po = [], []
    with torch.no_grad():
        for k in range(0, len(Xte), 256):
            idx = np.arange(k, min(k + 256, len(Xte)))
            x = t(lote(Xte, Lte, idx)).float()
            tech, tgt, *_ = modelo(None, x, torch.zeros((len(idx), 1), device=dispositivo))
            pt.append(tech.cpu().numpy())
            po.append(tgt.cpu().numpy())
    return np.vstack(pt), np.vstack(po)


def cargar(ruta: Path) -> dict:
    if not ruta.exists():
        raise SystemExit(f"falta {ruta}: correr armar.py")
    with np.load(ruta) as z:
        return {k: z[k] for k in z.files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exp", choices=sorted(EXPERIMENTOS), required=True)
    ap.add_argument("--semillas", type=int, nargs="*", default=list(SEMILLAS))
    ap.add_argument("--dispositivo", default=None)
    args = ap.parse_args(argv)
    import torch
    commit = verificar_clon()
    dispositivo = args.dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    datos = fuera_del_repo(TRABAJO_BM / "datos")
    bt = cargar(datos / "conjunto_boxtwin.npz")
    bw = cargar(datos / "conjunto_boxingweb.npz") if args.exp != "BM-1b" else None
    etiquetas = ("y_tipo", "y_altura", "y_distancia", "y_efecto")
    for nombre, tr_bt, tr_bw, te in particiones(args.exp, bt, bw):
        prueba = bw if args.exp == "BM-0b" else bt
        partes = [(bt, tr_bt)] + ([(bw, tr_bw)] if bw is not None and len(tr_bw) else [])
        L = max(int(d["largo"][i].max()) for d, i in partes if len(i))
        Xtr = np.concatenate([np.pad(d["X"][i], ((0, 0), (0, L - d["X"].shape[1]), (0, 0), (0, 0)))
                              if d["X"].shape[1] < L else d["X"][i][:, :L] for d, i in partes if len(i)])
        Ltr = np.concatenate([d["largo"][i] for d, i in partes if len(i)])
        Y = {c: np.concatenate([d[c][i] for d, i in partes if len(i)]) for c in etiquetas}
        destinos = ([(f, np.flatnonzero(prueba["fuente"][te] == f)) for f in geo.FUENTES]
                    if nombre == TODAS else [(nombre, np.arange(len(te)))])
        for s in args.semillas:
            rutas = [(f, sel, fuera_del_repo(TRABAJO_BM / "corridas" / args.exp / f / f"s{s}")) for f, sel in destinos]
            if all((r / "pred.npz").exists() for _, _, r in rutas):
                continue
            t0 = time.time()
            pt, po = entrenar_y_predecir(Xtr, Ltr, Y, prueba["X"][te], prueba["largo"][te], s, dispositivo)
            for f, sel, salida in rutas:
                salida.mkdir(parents=True, exist_ok=True)
                ti = te[sel]
                np.savez(salida / "pred.npz", frame_dir=prueba["frame_dir"][ti], p_tipo=pt[sel],
                         p_altura=p_altura_de(po[sel]), p_objetivo3=po[sel],
                         y_tipo=prueba["y_tipo"][ti], y_altura=prueba["y_altura"][ti])
                (salida / "config.json").write_text(json.dumps({
                    "exp": args.exp, "que": EXPERIMENTOS[args.exp], "prueba": f, "semilla": s,
                    "boxmind_commit": commit, "hiper": HIPER, "entrenamiento_boxtwin": int(len(tr_bt)),
                    "entrenamiento_boxingweb": int(len(tr_bw)), "prueba_golpes": int(len(ti)),
                    "dispositivo": dispositivo}, indent=2) + "\n")
                (salida / "tiempo.json").write_text(json.dumps({"segundos": round(time.time() - t0, 1)}) + "\n")
            print(f"{args.exp} {nombre} s{s}: {len(Ltr)} de entrenamiento, {len(te)} de prueba, "
                  f"{time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
