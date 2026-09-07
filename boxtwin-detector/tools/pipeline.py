#!/usr/bin/env python3
"""
BoxTwin - El sistema completo sobre video continuo, etapa 1: detectar.

POR QUE EXISTE
  Las tres piezas del sistema estan medidas por separado -pose, deteccion y familia del
  golpe- y el unico numero de punta a punta que existe es el del demo del 01-09, que usaba
  el disparador por extension de muneca: 14 de 21 golpes encontrados, 66 disparos, 21% de
  precision. Despues se midio que ese disparador NO SUPERA AL AZAR, asi que ese 21% no
  describe al sistema sino al ruido que lo alimentaba.

  Aca se lo reemplaza por el detector entrenado y se vuelve a medir contra la anotacion, con
  el mismo emparejador y el mismo criterio que el resto del proyecto.

  EL DETECTOR SE ELIGE POR FOLD, no por conveniencia: para medir sobre una fuente se usa el
  ensamble entrenado SIN esa fuente. Usar uno que la vio seria evaluar sobre el
  entrenamiento, que es el defecto por el que este proyecto descarto el baseline publico.

QUE HACE
  Corre el ensamble sobre una fuente entera, decodifica a segmentos, los mide contra la
  anotacion y los escribe a JSON para que la etapa 2 -la familia del golpe, que vive en el
  entorno boxtwin_mmaction- los pueda clasificar sin volver a correr nada.

USO
  python tools/pipeline.py sparring-3-rounds --datos data/ --out pipeline/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boxtwin.core.agreement import IOU_MINIMO, emparejar      # noqa: E402
from boxtwin_detector.dataset import leer                     # noqa: E402
from boxtwin_detector.decodificacion import decodificar_score  # noqa: E402
from boxtwin_detector.ensamble import entrenar_ensamble, probabilidad  # noqa: E402
from boxtwin_detector.entrenamiento import Config             # noqa: E402
from boxtwin_detector.evaluacion import segmentos_anotados, sobre_pose_medida  # noqa: E402

SEMILLAS = [42, 1, 2, 3, 4]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("USO")[0])
    p.add_argument("fuente", help="nombre de la fuente sobre la que medir")
    p.add_argument("--datos", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("pipeline"))
    p.add_argument("--umbral", type=float, default=0.80)
    p.add_argument("--semillas", type=int, nargs="+", default=SEMILLAS)
    p.add_argument("--epocas", type=int, default=40)
    args = p.parse_args()

    fuentes = {f.nombre: f for f in (leer(x) for x in sorted(args.datos.glob("*.det.npz")))}
    if args.fuente not in fuentes:
        print(f"error: no encuentro {args.fuente}; tengo {sorted(fuentes)}", file=sys.stderr)
        return 1
    val = fuentes[args.fuente]
    train = [f for n, f in fuentes.items() if n != args.fuente]
    if not train:
        print("error: hace falta al menos otra fuente para entrenar", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(epocas=args.epocas)
    print(f"fuente {args.fuente}: {val.T} cuadros a {val.fps:.0f} fps")
    print(f"detector entrenado SIN esa fuente, sobre {', '.join(f.nombre for f in train)}")
    ens = entrenar_ensamble(train, [val], cfg, args.semillas, device, verbose=False)
    print(f"ensamble de {ens.n} semillas listo")

    probs = probabilidad(ens, val, device)
    a, b = val.conteos.get("cobertura", [0, val.T - 1])
    valido = np.zeros(val.T, bool)
    valido[a : b + 1] = True

    gt = segmentos_anotados(val.labels, (a, b))
    medidos = sobre_pose_medida(gt, val.interpolado)

    anotados, salida = [], []
    n_par = 0
    d_ini, d_fin, ious = [], [], []
    base = 0
    for c, nombre_carril in enumerate(val.carriles):
        for k, s in enumerate(gt[c]):
            anotados.append({"indice": base + k, "carril": nombre_carril,
                             "inicio": s.inicio, "fin": s.fin,
                             "pose_medida": medidos[base + k]})
        pred = decodificar_score(probs[c], args.umbral, valido=valido)
        parejas, _, _ = emparejar(list(gt[c]), list(pred))
        casados = {j: i for i, j, _ in parejas}
        n_par += len(parejas)
        for i, j, v in parejas:
            d_ini.append(pred[j].inicio - gt[c][i].inicio)
            d_fin.append(pred[j].fin - gt[c][i].fin)
            ious.append(v)
        peleador, brazo = nombre_carril.split("-")
        for j, s in enumerate(pred):
            i = casados.get(j)
            salida.append({
                "carril": nombre_carril, "fighter": f"fighter_{peleador}", "side": brazo,
                "inicio": s.inicio, "fin": s.fin,
                "prob": round(float(probs[c][s.inicio : s.fin + 1].max()), 4),
                "empareja_con": None if i is None else base + i,
            })
        base += len(gt[c])

    n_gt, n_pred = len(anotados), len(salida)
    rec = n_par / n_gt if n_gt else 0.0
    pre = n_par / n_pred if n_pred else 0.0
    f1 = 2 * rec * pre / (rec + pre) if rec + pre else 0.0

    print(f"\ndeteccion sobre el video entero, IoU >= {IOU_MINIMO}")
    print(f"  golpes anotados      {n_gt}  ({sum(medidos)} con pose medida)")
    print(f"  marcas del detector  {n_pred}")
    print(f"  emparejados          {n_par}")
    print(f"  recall               {rec:.3f}")
    print(f"  precision            {pre:.3f}")
    print(f"  F1                   {f1:.3f}")
    if ious:
        print(f"  IoU medio            {np.mean(ious):.3f}")
        print(f"  error de fronteras   {np.mean(np.abs(d_ini)):.2f} al inicio, "
              f"{np.mean(np.abs(d_fin)):.2f} al final  (humano: 1,12 y 1,55)")

    args.out.mkdir(parents=True, exist_ok=True)
    dst = args.out / f"{args.fuente}.segmentos.json"
    dst.write_text(json.dumps({
        "fuente": args.fuente, "umbral": args.umbral, "semillas": args.semillas,
        "entrenado_sin_la_fuente": True,
        "train": [f.nombre for f in train],
        "procedencia": val.procedencia,
        "cobertura": [int(a), int(b)],
        "deteccion": {"golpes": n_gt, "marcas": n_pred, "emparejados": n_par,
                      "recall": round(rec, 4), "precision": round(pre, 4),
                      "f1": round(f1, 4)},
        "anotados": anotados,
        "segmentos": salida,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"\n  -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
