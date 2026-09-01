"""
BoxTwin - CLI del detector.

POR QUE EXISTE
  Armar el dataset y auditarlo se hace mucho mas seguido que entrenar, y no necesita GPU.
  Tenerlo como comando aparte deja que el paso barato se repita sin arrastrar torch.

QUE HACE
  `build` toma uno o mas exports `sequence` y escribe los tensores del detector, imprimiendo
  el reparto de clases. Ese reparto es lo primero que hay que mirar: la clase O se lleva mas
  del 95% de los cuadros, asi que la exactitud por cuadro no significa nada y conviene verlo
  antes de entrenar y no despues.

USO
  boxtwin-detector build exports/*.sequence.npz --out data/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from boxtwin_detector.dataset import FPS_DESTINO, construir, escribir, leer
from boxtwin_detector.version import __version__


def _build(args: argparse.Namespace) -> int:
    total = {"golpes": 0, "cuadros": 0, "usables": 0, "O": 0, "B": 0, "I": 0}
    for npz in args.exports:
        if not Path(npz).is_file():
            print(f"error: no existe {npz}", file=sys.stderr)
            return 1
        f = construir(npz, args.fps)
        salida, man = escribir(f, args.out)
        c = f.conteos
        usables = c["cuadros_usables"]
        print(f"\n{f.nombre}")
        print(f"  {c['cuadros_originales']} cuadros a {f.procedencia['fps_origen']:.2f} fps "
              f"-> {c['cuadros']} a {args.fps:.0f}")
        print(f"  {c['golpes']} golpes en {c['carriles']} carriles, "
              f"{c['amagues_enmascarados']} amagues enmascarados")
        if c["segmentos_ajustados_por_colision"]:
            print(f"  ATENCION: {c['segmentos_ajustados_por_colision']} segmentos se "
                  f"recortaron al remuestrear, porque cayeron encima del anterior")
        a, b = c["cobertura"]
        print(f"  tramo anotado {a}-{b} de {c['cuadros']} cuadros; "
              f"{c['cuadros_fuera_de_cobertura']} cuadros usables quedan afuera y no "
              f"cuentan como fondo")
        if c["etiquetados_no_usables"]:
            print(f"  {c['etiquetados_no_usables']} cuadros con etiqueta caen fuera de la "
                  f"mascara y no entran en la perdida")
        cf = c["carriles"] * c["cuadros"]
        print(f"  usables {usables}/{cf} ({usables/cf:.1%})")
        for k in ("O", "B", "I"):
            n = c[f"cuadros_{k}"]
            print(f"    {k}: {n:7d}  {n/max(usables,1):6.2%}")
        print(f"  -> {salida.name}, {man.name}")
        total["golpes"] += c["golpes"]
        total["cuadros"] += c["cuadros"] * c["carriles"]
        total["usables"] += usables
        for k in ("O", "B", "I"):
            total[k] += c[f"cuadros_{k}"]

    u = max(total["usables"], 1)
    print(f"\ntotal: {total['golpes']} golpes, {total['usables']} cuadros-carril usables")
    print(f"  O {total['O']/u:.2%} | B {total['B']/u:.3%} | I {total['I']/u:.2%}")
    print(f"  decir siempre O acierta {total['O']/u:.2%} por cuadro: la exactitud por "
          f"cuadro no es una metrica, se mide por evento")
    return 0


def _train(args: argparse.Namespace) -> int:
    # torch es un extra: importarlo aca deja que `build` corra sin el
    import json
    from dataclasses import asdict

    import torch

    from boxtwin_detector.entrenamiento import Config, Estandarizador, entrenar
    from boxtwin_detector.modelo import TCN
    from boxtwin_detector.splits import leave_one_source_out, partir_en_distribucion

    fuentes = [leer(p) for p in args.datos]
    por_nombre = {f.nombre: f for f in fuentes}
    cfg = Config(epocas=args.epocas, lr=args.lr, ventana=args.ventana, batch=args.batch,
                 ventanas_por_epoca=args.ventanas, alpha_pesos=args.alpha,
                 canales=args.canales, semilla=args.semilla)

    if args.fold == "en-distribucion":
        if args.fuente is None:
            print("error: --fuente es obligatorio con --fold en-distribucion", file=sys.stderr)
            return 1
        if args.fuente not in por_nombre:
            print(f"error: no cargue la fuente {args.fuente}; tengo "
                  f"{sorted(por_nombre)}", file=sys.stderr)
            return 1
        tr, va = partir_en_distribucion(por_nombre[args.fuente], args.fraccion)
        train, val, nombre = [tr], [va], f"en-distribucion-{args.fuente}"
    else:
        folds = {f.nombre: f for f in leave_one_source_out(sorted(por_nombre))}
        if args.fold not in folds:
            print(f"error: fold desconocido; tengo {sorted(folds)}", file=sys.stderr)
            return 1
        fold = folds[args.fold]
        train = [por_nombre[n] for n in fold.train]
        val = [por_nombre[n] for n in fold.val]
        nombre = fold.nombre

    est = Estandarizador.ajustar(train)
    modelo = TCN(n_features=train[0].features.shape[2], canales=cfg.canales,
                 dropout=args.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"fold {nombre}")
    print(f"  train: {', '.join(f.nombre for f in train)} "
          f"({sum(int(f.usable.sum()) for f in train)} cuadros usables)")
    print(f"  val:   {', '.join(f.nombre for f in val)} "
          f"({sum(int(f.usable.sum()) for f in val)} cuadros usables)")
    print(f"  campo receptivo {modelo.campo_receptivo} cuadros, device {device}")

    hist = entrenar(modelo, train, val, est, cfg, device)

    args.out.mkdir(parents=True, exist_ok=True)
    ck = args.out / f"detector-{nombre}.pt"
    torch.save({"state_dict": modelo.state_dict(), "config": asdict(cfg),
                "estandarizador": est.a_dict(), "n_features": modelo.n_features,
                "canales": cfg.canales, "fold": nombre}, ck)
    (args.out / f"detector-{nombre}.json").write_text(json.dumps({
        "fold": nombre, "fraccion": args.fraccion,
        "train": [f.nombre for f in train], "val": [f.nombre for f in val],
        "procedencia": {f.nombre: f.procedencia for f in fuentes},
        "historial": hist,
    }, indent=2, ensure_ascii=False) + "\n")

    m = hist["mejor"]["val"]
    print(f"\nmejor epoca {hist['mejor']['epoca']}: F1 macro {m['f1_macro']:.4f}")
    print(f"  recall  O {m['O_recall']:.3f}  B {m['B_recall']:.3f}  I {m['I_recall']:.3f}")
    print(f"  decir siempre O daria exactitud {m['siempre_O']:.2%} y F1 macro ~0,33")
    print(f"  -> {ck.name}")
    print("\nEsto es F1 por cuadro, que es una escalera y no el piso. La medida que "
          "importa\nes precision y recall POR EVENTO, y llega en el bloque 3.")
    return 0


def _eval(args: argparse.Namespace) -> int:
    import json

    import numpy as np
    import torch

    from boxtwin_detector.entrenamiento import Config, Estandarizador, predecir_secuencia
    from boxtwin_detector.evaluacion import barrer
    from boxtwin_detector.modelo import TCN
    from boxtwin_detector.splits import partir_en_distribucion

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = json.loads(Path(str(args.checkpoint).replace(".pt", ".json")).read_text())
    cfg = Config(**ck["config"])
    est = Estandarizador.de_dict(ck["estandarizador"])
    modelo = TCN(n_features=ck["n_features"], canales=ck["canales"])
    modelo.load_state_dict(ck["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(device).eval()

    por_nombre = {f.nombre: f for f in (leer(p) for p in args.datos)}
    fold = ck["fold"]
    if fold.startswith("en-distribucion-"):
        base = fold[len("en-distribucion-"):]
        if base not in por_nombre:
            print(f"error: falta la fuente {base}", file=sys.stderr)
            return 1
        _, va = partir_en_distribucion(por_nombre[base], meta.get("fraccion", 0.25))
        val = [va]
    else:
        val = [por_nombre[n] for n in meta["val"] if n in por_nombre]
        if not val:
            print(f"error: no cargue ninguna fuente de validacion ({meta['val']})",
                  file=sys.stderr)
            return 1

    print(f"fold {fold}, validando sobre {', '.join(f.nombre for f in val)}\n")
    umbrales = args.umbrales or [round(x, 2) for x in np.arange(0.10, 0.95, 0.05)]

    for f in val:
        logits = [predecir_secuencia(modelo, est.aplicar(f.features[c]), cfg, device)
                  for c in range(f.features.shape[0])]
        cob = tuple(f.conteos.get("cobertura", [0, f.T - 1]))
        filas = barrer(logits, f.labels, f.usable, umbrales, cobertura=cob,
                       largo_minimo=args.largo_minimo, hueco_maximo=args.hueco_maximo)
        n_gt = filas[0]["golpes"]
        n_alc = filas[0].get("golpes_alcanzables", n_gt)
        print(f"{f.nombre}: {n_gt} golpes anotados, {n_alc} con pose usable "
              f"({n_gt - n_alc} que el sistema no puede encontrar)")
        print(f"  {'umbral':>7} {'predichos':>10} {'recall':>8} {'precision':>10} "
              f"{'F1':>7} {'IoU':>6} {'err ini':>8} {'err fin':>8}")
        mejor = None
        for x in filas:
            r, pr = x["recall"], x["precision"]
            f1 = 2 * r * pr / (r + pr) if r + pr else 0.0
            if mejor is None or f1 > mejor[0]:
                mejor = (f1, x)
            print(f"  {x['umbral']:7.2f} {x['predichos']:10d} {r:8.3f} {pr:10.3f} "
                  f"{f1:7.3f} {x['iou_medio']:6.3f} "
                  f"{(x['error_inicio'] if x['error_inicio'] is not None else float('nan')):8.2f} "
                  f"{(x['error_fin'] if x['error_fin'] is not None else float('nan')):8.2f}")
        f1, x = mejor
        print(f"\n  mejor F1 {f1:.3f} en umbral {x['umbral']:.2f}: "
              f"recall {x['recall']:.3f}, precision {x['precision']:.3f}")
        print(f"  recall sobre los alcanzables: {x.get('recall_alcanzables', float('nan')):.3f}")
        # -- la heuristica, por el MISMO decodificador y el MISMO emparejador
        from boxtwin_detector.features import NOMBRES
        i_ext = NOMBRES.index("extension")
        ext = [f.features[c, :, i_ext] for c in range(f.features.shape[0])]
        base = barrer(ext, f.labels, f.usable,
                      [round(x, 2) for x in np.arange(0.3, 1.7, 0.1)],
                      cobertura=cob, como_score=True,
                      largo_minimo=args.largo_minimo, hueco_maximo=args.hueco_maximo)
        mejor_b = max(base, key=lambda x: (2 * x["recall"] * x["precision"] /
                                           (x["recall"] + x["precision"]))
                      if x["recall"] + x["precision"] else 0.0)
        f1b = (2 * mejor_b["recall"] * mejor_b["precision"] /
               (mejor_b["recall"] + mejor_b["precision"])
               if mejor_b["recall"] + mejor_b["precision"] else 0.0)

        print("\n  contra que se compara, todo sobre los mismos golpes:")
        print(f"    modelo                             recall {x['recall']:.3f}  "
              f"precision {x['precision']:.3f}  F1 {f1:.3f}")
        print(f"    extension de muneca, mismo decoder recall {mejor_b['recall']:.3f}  "
              f"precision {mejor_b['precision']:.3f}  F1 {f1b:.3f}  "
              f"(umbral {mejor_b['umbral']:.1f})")
        print("    techo humano (reanotacion ciega)   recall 0,911  precision 0,903")
        print("    error de fronteras del humano      1,12 cuadros al inicio, 1,55 al final")
        if f1 <= f1b:
            print("\n    ATENCION: el modelo NO le gana a la heuristica en este fold.")
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="boxtwin-detector", description=__doc__.split("USO")[0])
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="arma los tensores del detector desde exports sequence")
    b.add_argument("exports", type=Path, nargs="+")
    b.add_argument("--out", type=Path, default=Path("data"))
    b.add_argument("--fps", type=float, default=FPS_DESTINO,
                   help="fps comun al que se lleva todo. Pacquiao viene a 59,94 y el resto "
                        "a 30; sin esto el mismo golpe dura el doble de cuadros en una "
                        "fuente que en otra.")
    b.set_defaults(func=_build)

    tr = sub.add_parser("train", help="entrena el detector sobre un fold")
    tr.add_argument("datos", type=Path, nargs="+", help="los .det.npz que escribio build")
    tr.add_argument("--fold", required=True,
                    help="sin-<fuente> para dejar una afuera, o en-distribucion")
    tr.add_argument("--fuente", default=None, help="cual, con --fold en-distribucion")
    tr.add_argument("--fraccion", type=float, default=0.25)
    tr.add_argument("--out", type=Path, default=Path("modelos"))
    tr.add_argument("--epocas", type=int, default=40)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--ventana", type=int, default=256)
    tr.add_argument("--batch", type=int, default=32)
    tr.add_argument("--ventanas", type=int, default=512)
    tr.add_argument("--alpha", type=float, default=0.5)
    tr.add_argument("--canales", type=int, default=64)
    tr.add_argument("--dropout", type=float, default=0.1)
    tr.add_argument("--semilla", type=int, default=42)
    tr.set_defaults(func=_train)

    ev = sub.add_parser("eval", help="decodifica a segmentos y mide por evento")
    ev.add_argument("checkpoint", type=Path)
    ev.add_argument("datos", type=Path, nargs="+")
    ev.add_argument("--umbrales", type=float, nargs="+", default=None)
    ev.add_argument("--largo-minimo", type=int, default=5,
                    help="los golpes duran 7 cuadros de mediana, p10 en 5")
    ev.add_argument("--hueco-maximo", type=int, default=2)
    ev.set_defaults(func=_eval)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
