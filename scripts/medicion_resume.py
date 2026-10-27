#!/usr/bin/env python3
"""
BoxTwin - Deriva S-19, S-17 y S-04 desde las corridas crudas.

POR QUE EXISTE
  Los tres supuestos del Capitulo 5 no se leen de una corrida, se derivan de muchas: la
  mediana sobre las corridas en caliente, el rango, y la dispersion entre videos. Hacer esa
  cuenta a mano invita a elegir el numero que queda lindo. Este script la hace siempre
  igual, separa frio de caliente como manda el criterio M-07, y se niega a informar un
  supuesto para el que no hay filas.

  Criterio que se respeta: ninguna cifra pasa de "estimado" a "medido" sin la fila
  correspondiente en corridas.csv o potencia.csv. Si potencia.csv no existe, S-17 sale
  declarado como NO MEDIDO y no como cero.

QUE HACE
  Lee mediciones/corridas.csv y, si existe, mediciones/potencia.csv. Informa por video y en
  conjunto: mediana y rango del ratio de computo, reparto por etapa, y la derivacion de
  costo contra el precio de lista que se le pase.

USO
  python ../scripts/medicion_resume.py --corridas mediciones/corridas.csv \
      --precio-usd-hora 0.526 --instancia "AWS g4dn.xlarge (T4), us-east-1, 15-09-2026"
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

ETAPAS = ["preparacion", "decodificacion", "pose", "seguimiento",
          "identidad", "deteccion", "clasificacion"]


def mediana_rango(xs: list[float]) -> tuple[float, float, float]:
    return st.median(xs), min(xs), max(xs)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("USO")[0])
    p.add_argument("--corridas", type=Path, default=Path("mediciones/corridas.csv"))
    p.add_argument("--potencia", type=Path, default=Path("mediciones/potencia.csv"))
    p.add_argument("--precio-usd-hora", type=float, default=None)
    p.add_argument("--instancia", default="")
    p.add_argument("--overhead-pct", type=float, default=15.0,
                   help="almacenamiento y transferencia, en % sobre el computo")
    a = p.parse_args()

    if not a.corridas.is_file():
        print(f"error: no existe {a.corridas}")
        return 1

    filas = [r for r in csv.DictReader(a.corridas.open(encoding="utf-8"))]
    ok = [r for r in filas if r["estado"] in ("frio", "caliente") and r["ratio_computo"]]
    frio = [r for r in ok if r["estado"] == "frio"]
    cal = [r for r in ok if r["estado"] == "caliente"]

    # Una corrida cuyo cache no reproduce al de referencia no se usa: si los track_id
    # cambiaron, la identidad apunta a otra persona y las etapas 4 y 5 midieron otra cosa.
    # La excepcion es la divergencia documentada del recorte, que vive fuera del rango
    # anotado y por lo tanto no toca nada de lo que corre aguas abajo.
    def usable(r: dict) -> bool:
        if r.get("cache_identico") != "no":
            return True
        return r.get("notas", "").startswith("divergencia-ultimo-cuadro")

    malas = [r for r in ok if not usable(r)]
    if malas:
        print(f"AVISO: {len(malas)} corridas con cache distinto al de referencia. "
              "No se usan para derivar.")
    exc = [r for r in ok if r.get("cache_identico") == "no" and usable(r)]
    if exc:
        print(f"AVISO: {len(exc)} corridas se aceptan por excepcion documentada "
              "(divergencia confinada al ultimo cuadro del recorte).")
    cal = [r for r in cal if usable(r)]
    frio = [r for r in frio if usable(r)]

    print(f"corridas leidas: {len(filas)}  validas: {len(ok)}  "
          f"frias: {len(frio)}  calientes: {len(cal)}\n")

    if not cal:
        print("no hay corridas en caliente: S-19 no se puede derivar todavia")
        return 1

    # -- por video
    print("por video, corridas en caliente")
    print(f"  {'video':<28} {'dur_s':>8} {'n':>2} {'ratio med':>10} {'rango':>16}")
    por_video: dict[str, list[dict]] = {}
    for r in cal:
        por_video.setdefault(r["video"], []).append(r)
    medianas = []
    for v, rs in sorted(por_video.items()):
        ratios = [float(x["ratio_computo"]) for x in rs]
        m, lo, hi = mediana_rango(ratios)
        medianas.append(m)
        print(f"  {v:<28} {float(rs[0]['duracion_video_s']):>8.1f} {len(rs):>2} "
              f"{m:>10.3f} {lo:>7.3f}-{hi:<8.3f}")

    ratios_todos = [float(r["ratio_computo"]) for r in cal]
    m, lo, hi = mediana_rango(ratios_todos)
    print(f"\nS-19  relacion de computo por hora de video")
    print(f"  mediana sobre {len(cal)} corridas en caliente : {m:.3f}x tiempo real")
    print(f"  rango                                   : {lo:.3f}x a {hi:.3f}x")
    print(f"  dispersion entre videos (mediana por video): "
          f"{min(medianas):.3f}x a {max(medianas):.3f}x")
    if frio:
        rf = [float(r["ratio_computo"]) for r in frio]
        mf, lof, hif = mediana_rango(rf)
        print(f"  corridas en frio (aparte, M-07)         : mediana {mf:.3f}x, "
              f"rango {lof:.3f}x a {hif:.3f}x")
    print(f"  en horas de computo por hora de video   : {m:.3f} h/h")

    # -- reparto por etapa
    print("\nreparto por etapa, corridas en caliente (mediana del % del total)")
    for e in ETAPAS:
        pcts = []
        for r in cal:
            t = float(r["tiempo_total_s"])
            v = float(r.get(f"t_{e}_s") or 0.0)
            if t > 0:
                pcts.append(100.0 * v / t)
        if pcts:
            print(f"  {e:<16} {st.median(pcts):>6.1f}%   "
                  f"({min(pcts):.1f}% a {max(pcts):.1f}%)")
    cubierto = []
    for r in cal:
        t = float(r["tiempo_total_s"])
        s = sum(float(r.get(f"t_{e}_s") or 0.0) for e in ETAPAS)
        if t > 0:
            cubierto.append(100.0 * s / t)
    print(f"  {'SUMA':<16} {st.median(cubierto):>6.1f}%   "
          f"({min(cubierto):.1f}% a {max(cubierto):.1f}%)  <- cuanto del total explican")
    print("  estimacion de resultado: NO MEDIDA, no existe como codigo (desvio D-1)")

    # -- S-17
    print("\nS-17  consumo electrico")
    if a.potencia.is_file():
        ms = [r for r in csv.DictReader(a.potencia.open(encoding="utf-8"))]
        ws = [float(r["potencia_w"]) for r in ms if r.get("potencia_w")]
        if ws:
            print(f"  muestras            : {len(ws)}")
            print(f"  potencia media      : {st.mean(ws):.1f} W")
            print(f"  mediana             : {st.median(ws):.1f} W")
            print(f"  pico                : {max(ws):.1f} W")
            print("  (consumo de la placa, no del equipo ni de una instancia de nube)")
        else:
            print("  potencia.csv existe pero no tiene muestras")
    else:
        print("  NO MEDIDO. nvidia-smi no arranca en esta maquina (desvio D-3).")
        print("  El supuesto de 250 W sigue siendo supuesto.")

    # -- S-04
    print("\nS-04  costo variable por hora de video")
    if a.precio_usd_hora is None:
        print("  falta --precio-usd-hora; no se deriva")
    else:
        base = m * a.precio_usd_hora
        total = base * (1 + a.overhead_pct / 100.0)
        base_lo, base_hi = lo * a.precio_usd_hora, hi * a.precio_usd_hora
        print(f"  instancia           : {a.instancia or '(sin declarar)'}")
        print(f"  precio de lista     : {a.precio_usd_hora:.4f} USD/h")
        print(f"  computo             : {m:.3f} h x {a.precio_usd_hora:.4f} = "
              f"{base:.4f} USD por hora de video")
        print(f"  + {a.overhead_pct:.0f}% almacenamiento y transferencia = "
              f"{total:.4f} USD por hora de video")
        print(f"  rango por el rango de S-19: {base_lo*(1+a.overhead_pct/100):.4f} a "
              f"{base_hi*(1+a.overhead_pct/100):.4f} USD")
        print("  SUPUESTO NO CERRADO: que esa instancia rinda como una RTX 2080 Super.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
