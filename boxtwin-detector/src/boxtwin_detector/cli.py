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

from boxtwin_detector.dataset import FPS_DESTINO, construir, escribir
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

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
