"""
BoxTwin - CLI del detector de guantes.

POR QUE EXISTE
  Bajar el dataset y armar los recortes se hace mucho mas seguido que entrenar, y no
  necesita GPU. Tenerlo en comandos separados deja que el paso barato se repita sin
  arrastrar torch ni ultralytics.

QUE HACE
  `fetch` baja el export de una version, le devuelve la relacion de aspecto que el
  preprocesado de Roboflow le habia sacado, y lo reparte por clip de origen. Lo primero que
  hay que mirar de su salida es `clips_en_mas_de_un_split`: si no esta vacio, hay cuadros
  vecinos del mismo video en train y en validacion y cualquier metrica posterior mide de mas.

USO
  ROBOFLOW_API_KEY=... boxtwin-guantes fetch --version 1 --out data/roboflow
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from boxtwin_guantes.version import __version__


def _fetch(args) -> int:
    from boxtwin_guantes.roboflow import ErrorRoboflow, clave_api, descargar_dataset

    # Con un zip bajado a mano la clave es opcional: solo sirve para cruzar el catalogo y
    # saber el tamano original de cada imagen.
    try:
        key = clave_api()
    except ErrorRoboflow as e:
        if not args.zip:
            print(f"error: {e}", file=sys.stderr)
            return 1
        key = None
        print(
            "aviso: sin ROBOFLOW_API_KEY no se puede cruzar el catalogo; las imagenes "
            "quedan con el estirado del export",
            file=sys.stderr,
        )

    def progreso(hechas: int, total: int) -> None:
        print(f"  {hechas}/{total}", file=sys.stderr)

    try:
        m = descargar_dataset(
            key, Path(args.out), version=args.version, val=args.val, test=args.test,
            zip_local=Path(args.zip) if args.zip else None,
            progreso=None if args.silencioso else progreso,
        )
    except ErrorRoboflow as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"\n{m['imagenes']} imagenes, {m['cajas']} cajas, "
          f"{m['clips_de_origen']} clips de origen distintos")
    for s, c in m["conteos"].items():
        print(f"  {s:<6} {c['imagenes']:>5} imagenes  {c['cajas']:>6} cajas")
    if m["sin_cruce_con_catalogo"]:
        print(f"\n{m['sin_cruce_con_catalogo']} imagenes no cruzaron con el catalogo y "
              "quedaron con el estirado del export:\n  sin el tamano original no se puede "
              "deshacer, y asumir una relacion de aspecto\n  deformaria las que no la "
              "tenian. Con ROBOFLOW_API_KEY en el entorno se cruzan.")
    if m["ilegibles"]:
        print(f"{m['ilegibles']} imagenes no se pudieron leer y quedaron afuera")

    fuga = m["clips_en_mas_de_un_split"]
    if fuga:
        print(f"\nCUIDADO: {len(fuga)} clips caen en mas de un split. Eso filtra cuadros")
        print("vecinos entre train y validacion y las metricas van a medir de mas.")
    else:
        print("\nNingun clip cruza splits: la validacion no ve cuadros vecinos de train.")

    print(f"\nmanifiesto en {Path(args.out) / 'manifiesto.json'}")
    return 0


def _recortes(args) -> int:
    from boxtwin_guantes.recortes import ConfigRecortes, construir_recortes

    cfg = ConfigRecortes(
        margen=args.margen, alto_minimo=args.alto_minimo,
        ratio_min=args.ratio_min, ratio_max=args.ratio_max,
    )

    def progreso(hechas: int, total: int) -> None:
        print(f"  {hechas}/{total}", file=sys.stderr)

    m = construir_recortes(
        Path(args.entrada), Path(args.out), args.model,
        cfg=cfg, progreso=None if args.silencioso else progreso,
    )
    st = m["stats"]
    print(f"\n{st['recortes']} recortes con {st['cajas']} cajas, desde "
          f"{st['imagenes_leidas']} imagenes")
    for s, c in m["conteos"].items():
        print(f"  {s:<6} {c['recortes']:>5} recortes  {c['cajas']:>6} cajas")
    print("\ndescartes:")
    print(f"  imagen sin ninguna persona          {st['sin_persona']:>6}")
    print(f"  persona sin guante adentro          {st['descartados_sin_guante']:>6}")
    print(f"  proporcion guante/cuerpo fuera de banda {st['descartados_por_proporcion']:>6}")
    print(f"  persona mas chica que el minimo     {st['descartados_por_tamano']:>6}")
    return 0


def _train(args) -> int:
    from boxtwin_guantes.entrenamiento import ConfigEntrenamiento, entrenar

    cfg = ConfigEntrenamiento(
        pesos=args.pesos, epocas=args.epocas, imgsz=args.imgsz,
        batch=args.batch, semilla=args.semilla, device=args.device,
    )
    try:
        s = entrenar(Path(args.datos), Path(args.out), cfg)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"\nmodelo en {Path(args.out) / 'guantes.pt'}")
    for k, v in s["metricas_validacion"].items():
        print(f"  {k:<28} {v:.4f}")
    print(f"\n{s['advertencia']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="boxtwin-guantes", description=__doc__.split("USO")[0]
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="bajar el dataset de guantes desde Roboflow")
    f.add_argument("--out", default="data/roboflow", help="directorio de salida")
    f.add_argument(
        "--version", type=int, default=1, dest="version",
        help="version del dataset. La 1 es la unica sin ecualizacion de contraste ni "
             "escala de grises; las otras hornean preprocesado que no queremos",
    )
    f.add_argument("--val", type=float, default=0.2, help="fraccion de validacion")
    f.add_argument("--test", type=float, default=0.1, help="fraccion de test")
    f.add_argument(
        "--zip", default=None,
        help="usar un export ya bajado a mano en vez de pedirselo a la API. Tiene que ser "
             "el formato YOLOv8 de Roboflow",
    )
    f.add_argument("--silencioso", action="store_true", help="sin progreso")
    f.set_defaults(func=_fetch)

    r = sub.add_parser("recortes", help="derivar recortes de persona del dataset")
    r.add_argument("entrada", help="dataset de fetch, p.ej. data/roboflow-v3")
    r.add_argument("--out", default="data/recortes", help="directorio de salida")
    r.add_argument("--model", default="../yolov8l-pose.pt", help="pesos de pose")
    r.add_argument("--margen", type=float, default=0.08,
                   help="cuanto se agranda la caja de persona para no cortar el guante")
    r.add_argument("--alto-minimo", type=int, default=64, dest="alto_minimo",
                   help="personas mas bajas que esto no dan un recorte con detalle")
    r.add_argument("--ratio-min", type=float, default=0.05, dest="ratio_min",
                   help="alto minimo del guante como fraccion del alto de la persona")
    r.add_argument("--ratio-max", type=float, default=0.35, dest="ratio_max",
                   help="alto maximo. La referencia anatomica es 0,167; arriba de la banda "
                        "son retratos de primer plano, que ensenan una proporcion que en un "
                        "ring no existe")
    r.add_argument("--silencioso", action="store_true", help="sin progreso")
    r.set_defaults(func=_recortes)

    t = sub.add_parser("train", help="entrenar el detector sobre los recortes")
    t.add_argument("datos", help="dataset de recortes")
    t.add_argument("--out", default="modelos", help="directorio de salida")
    t.add_argument("--pesos", default="yolov8n.pt", help="pesos de partida")
    t.add_argument("--epocas", type=int, default=80)
    t.add_argument("--imgsz", type=int, default=320,
                   help="las entradas son recortes, no cuadros completos: subirlo no agrega "
                        "detalle que el recorte no tenga")
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--semilla", type=int, default=0)
    t.add_argument("--device", default="0")
    t.set_defaults(func=_train)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
