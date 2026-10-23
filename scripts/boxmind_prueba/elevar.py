"""
BoxTwin - 3D de MotionBERT del atacante y del rival, por golpe, para la prueba BM.

POR QUE EXISTE
  La representacion de BoxMind lleva el 3D de los dos peleadores, cada uno en su marco. E5
  elevo solo al atacante y sobre la ventana de E4 (inicio - 8 a fin + 8, remuestreada a 32).
  BM usa la ventana anotada tal cual, como BoxMind, y necesita tambien al rival, en BoxTwin y
  en BoxingWeb re-estimado. Se reutiliza el elevador de E5 (mismo config, mismos pesos, mismo
  contexto de 243 cuadros, mismo relleno de huecos), asi el 3D sale del mismo lugar en los dos
  dominios.

QUE HACE
  Para cada golpe y cada persona (0 atacante, 1 rival): contexto de 243 cuadros centrado en la
  ventana anotada [start_frame, end_frame], huecos de hasta 10 cuadros interpolados, elevacion,
  y salida en los cuadros de la ventana con paso 2 si el video supera 45 fps (la misma regla de
  E5; deja todo cerca de los 30 fps de BoxingWeb). Escribe
  <trabajo_bm>/datos/pose3d_<dominio>.npz con frame_dir, persona, largo, P3 (N, L, 17, 3) y
  M3 (N, L), con L el largo maximo y relleno en cero.

USO (en boxtwin_mmaction)
  ~/miniforge3/envs/boxtwin_mmaction/bin/python scripts/boxmind_prueba/elevar.py --dominio boxtwin
  ~/miniforge3/envs/boxtwin_mmaction/bin/python scripts/boxmind_prueba/elevar.py --dominio boxingweb
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bm import TRABAJO_BM, fuera_del_repo, geo  # noqa: E402

sys.path.insert(0, str(Path(geo.__file__).resolve().parent))
from elevar3d import (LOTE, PESOS_URL, Elevador, elegir_dispositivo, indices_contexto,  # noqa: E402
                      paso_de, rellenar_huecos)


def cuadros_de(inicio: int, fin: int, fps: float) -> np.ndarray:
    """Cuadros de la ventana anotada, con paso 2 por encima de 45 fps."""
    return np.arange(int(inicio), int(fin) + 1, paso_de(fps))


def en_cuadros(X: np.ndarray, crudo: np.ndarray, cuadros: np.ndarray) -> np.ndarray:
    """(largo, K, 3) en los cuadros `crudo` -> (len(cuadros), K, 3) interpolado."""
    flat = X.reshape(len(X), -1)
    out = np.stack([np.interp(cuadros, crudo, flat[:, j]) for j in range(flat.shape[1])], -1)
    return out.reshape(len(cuadros), *X.shape[1:]).astype(np.float32)


def golpes_y_series(dominio: str) -> tuple[list[dict], callable]:
    """Golpes del dominio y una funcion fuente -> (kp, sc, valid, fps, ancho, alto)."""
    if dominio == "boxtwin":
        from datos import leer_csv as leer
        base = geo.TRABAJO / "datos"
        golpes = leer(base / "golpes.csv")
    else:
        from boxingweb import leer_csv as leer
        base = TRABAJO_BM / "boxingweb"
        golpes = leer(base / "golpes.csv")

    def serie(fuente):
        with np.load(base / "series" / f"{fuente}.npz") as z:
            return z["kp"], z["sc"], z["valid"], float(z["fps"]), int(z["ancho"]), int(z["alto"])
    return golpes, serie


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dominio", choices=("boxtwin", "boxingweb"), required=True)
    ap.add_argument("--dispositivo", default="auto")
    ap.add_argument("--pesos", type=Path, default=geo.TRABAJO / "pesos" / Path(PESOS_URL).name)
    args = ap.parse_args(argv)
    destino = fuera_del_repo(TRABAJO_BM / "datos")
    destino.mkdir(parents=True, exist_ok=True)
    golpes, serie = golpes_y_series(args.dominio)
    dispositivo = elegir_dispositivo(args.dispositivo)
    print(f"{args.dominio}: {len(golpes)} golpes, dispositivo {dispositivo}", flush=True)
    elev = Elevador(args.pesos, dispositivo)
    filas, t0 = [], time.time()
    for fuente in sorted({g["fuente"] for g in golpes}):
        gs = [g for g in golpes if g["fuente"] == fuente]
        kp, _, valid, fps, ancho, alto = serie(fuente)
        T, paso = kp.shape[1], paso_de(fps)
        rellenos = {p: rellenar_huecos(kp[p], valid[p]) for p in (0, 1)}
        trabajos = [(g, persona) for g in gs for persona in (0, 1)]
        for i in range(0, len(trabajos), LOTE):
            lote = trabajos[i:i + LOTE]
            entradas, ctx = [], []
            for g, persona in lote:
                p = g["p"] if persona == 0 else 1 - g["p"]
                crudo, idx = indices_contexto(g["start_frame"], g["end_frame"], T, paso)
                entradas.append(elev.preparar(rellenos[p][0][idx], ancho, alto))
                ctx.append((crudo, p))
            for (g, persona), (crudo, p), X in zip(lote, ctx, elev.elevar(entradas)):
                cs = cuadros_de(g["start_frame"], g["end_frame"], fps)
                filas.append((g["frame_dir"], persona, en_cuadros(X, crudo, cs),
                              rellenos[p][1][np.clip(cs, 0, T - 1)]))
        print(f"{fuente}: {len(gs)} golpes, paso {paso}, {time.time() - t0:.0f} s acumulados", flush=True)
    L = max(len(f[2]) for f in filas)
    P3 = np.zeros((len(filas), L, 17, 3), np.float32)
    M3 = np.zeros((len(filas), L), bool)
    for i, (_, _, X, m) in enumerate(filas):
        P3[i, :len(X)], M3[i, :len(X)] = X, m
    np.savez_compressed(destino / f"pose3d_{args.dominio}.npz",
                        frame_dir=np.array([f[0] for f in filas]), persona=np.array([f[1] for f in filas]),
                        largo=np.array([len(f[2]) for f in filas]), P3=P3, M3=M3)
    print(f"listo: {len(filas)} elevaciones en {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
