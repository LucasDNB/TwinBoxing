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
        "fold": nombre,
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

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
