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
        f = construir(npz, args.fps, interpolados=not args.sin_interpolados)
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

    from boxtwin_detector.entrenamiento import Config, Estandarizador, entrenar, sembrar
    from boxtwin_detector.modelo import TCN
    from boxtwin_detector.splits import leave_one_source_out, partir_en_distribucion

    fuentes = [leer(p) for p in args.datos]
    por_nombre = {f.nombre: f for f in fuentes}
    semillas = args.semillas
    cfg = Config(epocas=args.epocas, lr=args.lr, ventana=args.ventana, batch=args.batch,
                 ventanas_por_epoca=args.ventanas, alpha_pesos=args.alpha,
                 canales=args.canales, semilla=semillas[0], dropout=args.dropout)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rf = TCN(n_features=train[0].features.shape[2], canales=cfg.canales).campo_receptivo

    print(f"fold {nombre}")
    print(f"  train: {', '.join(f.nombre for f in train)} "
          f"({sum(int(f.usable.sum()) for f in train)} cuadros usables)")
    print(f"  val:   {', '.join(f.nombre for f in val)} "
          f"({sum(int(f.usable.sum()) for f in val)} cuadros usables)")
    print(f"  campo receptivo {rf} cuadros, device {device}, "
          f"{len(semillas)} semilla(s): {semillas}")

    args.out.mkdir(parents=True, exist_ok=True)
    if len(semillas) > 1:
        from boxtwin_detector.ensamble import entrenar_ensamble, guardar
        ens = entrenar_ensamble(train, val, cfg, semillas, device)
        ck = args.out / f"detector-{nombre}.ens.pt"
        guardar(ens, ck)
        hist = {"mejor": ens.historiales[0]["mejor"],
                "semillas": semillas,
                "por_semilla": [h["mejor"] for h in ens.historiales]}
    else:
        sembrar(cfg.semilla)      # antes de construir: fija la inicializacion de los pesos
        modelo = TCN(n_features=train[0].features.shape[2], canales=cfg.canales,
                     dropout=args.dropout)
        hist = entrenar(modelo, train, val, est, cfg, device)
        ck = args.out / f"detector-{nombre}.pt"
        torch.save({"state_dict": modelo.state_dict(), "config": asdict(cfg),
                    "estandarizador": est.a_dict(), "n_features": modelo.n_features,
                    "canales": cfg.canales, "fold": nombre}, ck)
    (args.out / f"detector-{nombre}.json").write_text(json.dumps({
        "semillas": semillas,
        "fold": nombre, "fraccion": args.fraccion,
        "train": [f.nombre for f in train], "val": [f.nombre for f in val],
        "procedencia": {f.nombre: f.procedencia for f in fuentes},
        "historial": hist,
    }, indent=2, ensure_ascii=False) + "\n")

    m = hist["mejor"]["val"]
    print(f"\nF1 macro por cuadro {m['f1_macro']:.4f} (epoca {hist['mejor']['epoca']})")
    print(f"  recall  O {m['O_recall']:.3f}  B {m['B_recall']:.3f}  I {m['I_recall']:.3f}")
    print(f"  decir siempre O daria exactitud {m['siempre_O']:.2%} y F1 macro ~0,33")
    print(f"  -> {ck.name}")
    print("\nEsto es F1 por cuadro, que es una escalera y no el piso. La medida que "
          "importa\nes precision y recall POR EVENTO: correr `eval` sobre este checkpoint.")
    if len(semillas) == 1:
        print("\nUna sola semilla: el desvio entre semillas es ~0,10 de F1 por evento, asi "
              "que\neste numero es una muestra de esa distribucion. Con --semillas 42 1 2 3 4 "
              "se\nentrena un ensamble, que gana ~0,10 y ademas es determinista.")
    return 0


def _eval(args: argparse.Namespace) -> int:
    import json

    import numpy as np
    import torch

    from boxtwin_detector.entrenamiento import Config, Estandarizador, predecir_secuencia
    from boxtwin_detector.evaluacion import barrer
    from boxtwin_detector.modelo import TCN
    from boxtwin_detector.splits import partir_en_distribucion

    from boxtwin_detector.ensamble import cargar as cargar_ensamble
    from boxtwin_detector.ensamble import probabilidad

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    es_ensamble = ck.get("kind") == "boxtwin.detector.ensamble"
    nombre_meta = str(args.checkpoint).replace(".ens.pt", ".json").replace(".pt", ".json")
    meta = json.loads(Path(nombre_meta).read_text())
    cfg = Config(**ck["config"])
    if es_ensamble:
        ens = cargar_ensamble(args.checkpoint, device)
        est = ens.estandarizador
        modelo = None
    else:
        est = Estandarizador.de_dict(ck["estandarizador"])
        modelo = TCN(n_features=ck["n_features"], canales=ck["canales"])
        modelo.load_state_dict(ck["state_dict"])
        modelo = modelo.to(device).eval()
        ens = None

    por_nombre = {f.nombre: f for f in (leer(p) for p in args.datos)}
    fold = ck.get("fold") or meta["fold"]
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

    if es_ensamble:
        print(f"fold {fold}, ENSAMBLE de {ens.n} semillas {ens.semillas}")
        print("  el umbral se barre con grilla fina: la salida NO queda cuantizada en\n"
              "  pasos de 1/n, y el optimo suele caer en un acantilado angosto\n")
    else:
        print(f"fold {fold}, un solo modelo\n")
    print(f"validando sobre {', '.join(f.nombre for f in val)}\n")
    # grilla fina en los dos casos. El ensamble tiene un acantilado angosto -entre 0,75 y
    # 0,80 se caen 360 marcas y la precision salta de 0,31 a 0,85- que una grilla gruesa se
    # saltea.
    umbrales = args.umbrales or [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]

    for f in val:
        cob = tuple(f.conteos.get("cobertura", [0, f.T - 1]))
        if es_ensamble:
            senal = probabilidad(ens, f, device)
            filas = barrer(senal, f.labels, f.usable, umbrales, cobertura=cob,
                           como_score=True, largo_minimo=args.largo_minimo,
                           hueco_maximo=args.hueco_maximo)
        else:
            logits = [predecir_secuencia(modelo, est.aplicar(f.features[c]), cfg, device)
                      for c in range(f.features.shape[0])]
            filas = barrer(logits, f.labels, f.usable, umbrales, cobertura=cob,
                           largo_minimo=args.largo_minimo, hueco_maximo=args.hueco_maximo)
        n_gt = filas[0]["golpes"]
        n_alc = filas[0].get("golpes_alcanzables", n_gt)
        print(f"{f.nombre}: {n_gt} golpes anotados, {n_alc} con pose usable "
              f"({n_gt - n_alc} que el sistema no puede encontrar)")
        print(f"  {'umbral':>7} {'predichos':>10} {'recall':>8} {'precision':>10} "
              f"{'F1':>7} {'IoU':>6} {'err ini':>8} {'err fin':>8}")
        mejor = None
        for k, x in enumerate(filas, start=1):
            r, pr = x["recall"], x["precision"]
            f1 = 2 * r * pr / (r + pr) if r + pr else 0.0
            if mejor is None or f1 > mejor[0]:
                mejor = (f1, x)
            etiqueta = f"{x['umbral']:.2f}"
            print(f"  {etiqueta:>7} {x['predichos']:10d} {r:8.3f} {pr:10.3f} "
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


def _folds(args: argparse.Namespace) -> int:
    """
    El protocolo del bloque 4: varias semillas por fold, y la heuristica al lado.

    Varias semillas y no una porque esta medido que el desvio entre semillas es del orden de
    0,1 de F1 por evento, o sea mas grande que casi cualquier efecto que se quiera reportar.
    Una corrida sola elige el numero que a uno le guste.
    """
    import json

    import numpy as np
    import torch

    from boxtwin_detector.decodificacion import probabilidad_de_golpe
    from boxtwin_detector.entrenamiento import (
        Config, Estandarizador, entrenar, predecir_secuencia, sembrar,
    )
    from boxtwin_detector.evaluacion import barrer
    from boxtwin_detector.features import NOMBRES
    from boxtwin_detector.modelo import TCN
    from boxtwin_detector.splits import leave_one_source_out, partir_en_distribucion

    fuentes = {f.nombre: f for f in (leer(p) for p in args.datos)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    umbrales = [round(x, 2) for x in np.arange(0.10, 0.95, 0.05)]

    def f1(r, p):
        return 2 * r * p / (r + p) if r + p else 0.0

    def probs_modelo(m, est, cfg, vals):
        """P(golpe) por fuente y carril. Se guarda para poder promediar el ensamble."""
        return [[probabilidad_de_golpe(
                    predecir_secuencia(m, est.aplicar(f.features[c]), cfg, device))
                 for c in range(f.features.shape[0])] for f in vals]

    def curva_de_probs(probs, vals, us):
        out = []
        for f, ps in zip(vals, probs):
            cob = tuple(f.conteos.get("cobertura", [0, f.T - 1]))
            out.append(barrer(ps, f.labels, f.usable, us, cobertura=cob, como_score=True,
                              interpolado=f.interpolado))
        return out

    def mejor_en(us, curvas):
        best = None
        for i, u in enumerate(us):
            g = sum(c[i]["golpes"] for c in curvas)
            pr = sum(c[i]["predichos"] for c in curvas)
            em = sum(c[i]["emparejados"] for c in curvas)
            r, p = (em / g if g else 0.0), (em / pr if pr else 0.0)
            v = f1(r, p)
            if best is None or v > best["f1"]:
                # los recall parciales se recomponen desde los conteos y no se promedian:
                # cada fuente aporta una cantidad distinta de golpes medidos
                gm = sum(c[i].get("golpes_medidos") or 0 for c in curvas)
                gr = sum(c[i].get("golpes_rellenados") or 0 for c in curvas)
                hm = sum(round((c[i].get("recall_medidos") or 0)
                               * (c[i].get("golpes_medidos") or 0)) for c in curvas)
                hr = sum(round((c[i].get("recall_rellenados") or 0)
                               * (c[i].get("golpes_rellenados") or 0)) for c in curvas)
                best = {"umbral": u, "golpes": g, "predichos": pr, "recall": round(r, 4),
                        "precision": round(p, 4), "f1": round(v, 4),
                        "golpes_medidos": gm, "golpes_rellenados": gr,
                        "recall_medidos": round(hm / gm, 4) if gm else None,
                        "recall_rellenados": round(hr / gr, 4) if gr else None}
        return best

    def curva_heuristica(vals):
        i = NOMBRES.index("extension")
        us = [round(x, 2) for x in np.arange(0.3, 1.7, 0.05)]
        out = []
        for f in vals:
            ext = [f.features[c, :, i] for c in range(f.features.shape[0])]
            cob = tuple(f.conteos.get("cobertura", [0, f.T - 1]))
            out.append(barrer(ext, f.labels, f.usable, us, cobertura=cob, como_score=True))
        best = None
        for k, u in enumerate(us):
            g = sum(c[k]["golpes"] for c in out)
            pr = sum(c[k]["predichos"] for c in out)
            em = sum(c[k]["emparejados"] for c in out)
            r, p = (em / g if g else 0.0), (em / pr if pr else 0.0)
            v = f1(r, p)
            if best is None or v > best["f1"]:
                best = {"umbral": u, "golpes": g, "recall": round(r, 4),
                        "precision": round(p, 4), "f1": round(v, 4)}
        return best

    tareas = []
    if args.en_distribucion:
        base = args.en_distribucion
        if base not in fuentes:
            print(f"error: no cargue {base}", file=sys.stderr)
            return 1
        tr, va = partir_en_distribucion(fuentes[base], args.fraccion)
        tareas.append((f"en-distribucion-{base}", [tr], [va]))
    for fold in leave_one_source_out(sorted(fuentes)):
        tareas.append((fold.nombre, [fuentes[n] for n in fold.train],
                       [fuentes[n] for n in fold.val]))

    def version_de(f):
        return f.procedencia.get("boundary_definitions_version")

    resultados = []
    avisos: list[str] = []
    n = len(args.semillas)
    us_voto = umbrales
    versiones = {k: version_de(f) for k, f in fuentes.items()}
    print(f"{n} semillas por fold, alpha {args.alpha}")
    if any(v is not None for v in versiones.values()):
        print("  boundary_definitions_version: "
              + ", ".join(f"{k} v{v}" for k, v in sorted(versiones.items())))
    print()
    print(f"{'fold':>26} {'golpes':>7} | {'una corrida':>12} {'desvio':>7} | "
          f"{'ensamble':>9} {'recall':>7} {'prec':>6} | {'rec medidos':>12} "
          f"{'rellenados':>11} | {'heur':>6}")
    for nombre, train, val in tareas:
        est = Estandarizador.ajustar(train)
        fs, corridas = [], []
        acum = [[np.zeros(f.T) for _ in range(f.features.shape[0])] for f in val]
        for s in args.semillas:
            cfg = Config(epocas=args.epocas, alpha_pesos=args.alpha, semilla=s)
            sembrar(s)
            m = TCN(n_features=train[0].features.shape[2], canales=cfg.canales)
            entrenar(m, train, val, est, cfg, device, verbose=False)
            probs = probs_modelo(m, est, cfg, val)
            for i, ps in enumerate(probs):
                for c, x in enumerate(ps):
                    acum[i][c] += x
            b = mejor_en(umbrales, curva_de_probs(probs, val, umbrales))
            corridas.append({"semilla": s, **b})
            fs.append(b["f1"])
        ens_probs = [[a / n for a in fuente_] for fuente_ in acum]
        e = mejor_en(us_voto, curva_de_probs(ens_probs, val, us_voto))
        h = curva_heuristica(val)
        rm = e.get("recall_medidos"); rr = e.get("recall_rellenados")
        s_rm = f"{rm:.3f} ({e['golpes_medidos']})" if rm is not None else "-"
        s_rr = f"{rr:.3f} ({e['golpes_rellenados']})" if rr is not None else "-"
        print(f"{nombre:>26} {h['golpes']:7d} | {np.mean(fs):12.3f} {np.std(fs):7.3f} | "
              f"{e['f1']:9.3f} {e['recall']:7.3f} {e['precision']:6.3f} | {s_rm:>12} "
              f"{s_rr:>11} | {h['f1']:6.3f}")
        v_val = {version_de(f) for f in val}
        v_train = {version_de(f) for f in train}
        if (v_val - v_train) and None not in (v_val | v_train):
            avisos.append(
                f"{nombre}: la validacion esta anotada bajo "
                f"boundary_definitions_version {sorted(v_val)} y el entrenamiento bajo "
                f"{sorted(v_train)}. Ese fold mide un cambio de convencion de anotacion "
                f"ademas del detector, y NO es comparable con los otros."
            )
        resultados.append({"fold": nombre, "corridas": corridas, "heuristica": h,
                           "boundary_version_val": sorted(x for x in v_val if x is not None),
                           "boundary_version_train": sorted(x for x in v_train if x is not None),
                           "ensamble": e,
                           "f1_medio": round(float(np.mean(fs)), 4),
                           "f1_desvio": round(float(np.std(fs)), 4)})

    for a in avisos:
        print(f"\n  ATENCION: {a}")
    print("\n  techo humano: recall 0,911  precision 0,903  F1 0,907")
    print("  \"una corrida\" es el promedio de las semillas por separado: lo que sale de "
          "entrenar una vez.")
    print("  NO se reporta la mejor de las semillas: elegirla mirando la validacion es "
          "seleccionar sobre el test.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"alpha": args.alpha, "semillas": args.semillas, "epocas": args.epocas,
             "procedencia": {n: f.procedencia for n, f in fuentes.items()},
             "resultados": resultados}, indent=2, ensure_ascii=False) + "\n")
        print(f"  -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="boxtwin-detector", description=__doc__.split("USO")[0])
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="arma los tensores del detector desde exports sequence")
    b.add_argument("exports", type=Path, nargs="+")
    b.add_argument("--out", type=Path, default=Path("data"))
    b.add_argument("--sin-interpolados", action="store_true",
                   help="saca de la mascara los cuadros de pose rellenada. No son un dato "
                        "observado: una interpolacion es una recta entre dos puntos. Pesa "
                        "desparejo, del 23,6% de los golpes en Sparring al 0% en las "
                        "fuentes nuevas.")
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
    tr.add_argument("--semillas", type=int, nargs="+", default=[42],
                    help="una sola entrena un modelo; varias entrenan un ENSAMBLE, que gana "
                         "~0,10 de F1 por evento y no tiene varianza de semilla")
    tr.set_defaults(func=_train)

    ev = sub.add_parser("eval", help="decodifica a segmentos y mide por evento")
    ev.add_argument("checkpoint", type=Path)
    ev.add_argument("datos", type=Path, nargs="+")
    ev.add_argument("--umbrales", type=float, nargs="+", default=None)
    ev.add_argument("--largo-minimo", type=int, default=5,
                    help="los golpes duran 7 cuadros de mediana, p10 en 5")
    ev.add_argument("--hueco-maximo", type=int, default=2)
    ev.set_defaults(func=_eval)

    fo = sub.add_parser("folds", help="el protocolo completo: varias semillas por fold")
    fo.add_argument("datos", type=Path, nargs="+")
    fo.add_argument("--semillas", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    fo.add_argument("--alpha", type=float, default=0.5)
    fo.add_argument("--epocas", type=int, default=40)
    fo.add_argument("--en-distribucion", default=None,
                    help="ademas de los folds cruzados, la particion en distribucion de esta fuente")
    fo.add_argument("--fraccion", type=float, default=0.25)
    fo.add_argument("--out", type=Path, default=None)
    fo.set_defaults(func=_folds)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
