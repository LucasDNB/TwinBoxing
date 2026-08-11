#!/usr/bin/env python3
"""
BoxingVI - Puntaje de la verificacion ciega de V8/V9/V10.

POR QUE EXISTE
  El criterio de decision sobre los tres videos sin verificar quedo fijado antes
  de mirar ningun resultado: 16 o mas aciertos sobre 18 y el video se usa tal
  cual, 11 a 15 y se reanota completo, 10 o menos y se descarta. Un criterio
  pre-registrado no sirve de nada si despues la cuenta se hace a mano en una
  terminal, porque ahi es donde se cuela el ajuste post-hoc: redondear un
  "dudoso" para el lado conveniente, o cambiar el denominador cuando el numero
  no da. Este script se escribio con el CSV de salida todavia vacio, asi que la
  regla de conteo tampoco se eligio viendo los datos.

  Ademas fuerza a declarar el denominador. La reanotacion de V2 dejo dos
  numeros correctos y distintos (7.3% sobre los 232 clips del video, 12.7%
  sobre los 134 que si contenian un golpe) y los dos andaban dando vueltas sin
  aclarar cual era cual. Aca salen siempre los dos, etiquetados.

QUE HACE
  - Cuenta aciertos por video contra la etiqueta original y aplica la tabla.
  - Separa los clips sin golpe: un clip vacio es fallo de segmentacion, no de
    clase, y las dos cosas se arreglan distinto (resegmentar contra reetiquetar).
  - Trata los "dudoso" como no resueltos: calcula la decision en el mejor y en
    el peor caso y, si caen en filas distintas de la tabla, se niega a decidir.
  - Reporta el limite inferior Wilson al 95%: 18 de 18 no son 100% de calidad,
    son 82% de piso, y en el capitulo se escribe asi.
  - Con --detalle, en que se convirtio cada clase original.

USO
  cd ~/Proyectos/TwinBoxing/test-BoxingVI
  python ../scripts/boxingvi_verifica.py --anot ./clips/verificacion_v8v9v10.csv
  python ../scripts/boxingvi_verifica.py --anot ./clips/reanotado.csv --videos V2 --n 232
"""

import argparse
import math
from collections import Counter, defaultdict

import pandas as pd

CLASSES = ["Jab", "Cross", "Lead Hook", "Rear Hook", "Lead Uppercut", "Rear Uppercut"]
SIN_GOLPE = "sin golpe"
DUDOSO = "dudoso"

# Tabla pre-registrada, sobre 18 clips por video. Ver CLAUDE.md seccion 5.
# Son cuentas absolutas calibradas para ese n: aplicarlas a otro tamano de
# muestra no significa nada, asi que el script se niega a hacerlo.
TABLA_N = 18
UMBRAL_USAR = 16
UMBRAL_REANOTAR = 11


def wilson_low(k, n, z=1.96):
    """Limite inferior del intervalo de Wilson al 95%.

    Wald no sirve acá: con k = n da un intervalo de ancho cero, que es
    justamente el caso que mas nos importa reportar con honestidad.
    """
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    centro = p + z * z / (2 * n)
    margen = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centro - margen) / d)


def _detalle(sub):
    """En que se convirtio cada clase original. Asi salio el hallazgo de V2 de
    que cero de 116 uppercuts eran uppercuts."""
    conf = defaultdict(Counter)
    for o, nu in zip(sub["cls_original"], sub["nueva_cls"]):
        conf[o][nu] += 1
    print("  detalle:")
    for o in CLASSES:
        if o not in conf:
            continue
        partes = ", ".join(f"{k} {v}" for k, v in conf[o].most_common())
        print(f"    {o:<15} -> {partes}")


def decision(aciertos):
    if aciertos >= UMBRAL_USAR:
        return "USAR TAL CUAL"
    if aciertos >= UMBRAL_REANOTAR:
        return "REANOTAR COMPLETO"
    return "DESCARTAR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anot", default="./clips/verificacion_v8v9v10.csv",
                    help="salida de boxingvi_annot.py")
    ap.add_argument("--videos", nargs="+", default=None,
                    help="video_keys a puntuar (default: todos los que aparezcan)")
    ap.add_argument("--n", type=int, default=18,
                    help="tamano de muestra esperado por video, para detectar anotacion parcial")
    ap.add_argument("--detalle", action="store_true",
                    help="matriz de en que se convirtio cada clase original")
    args = ap.parse_args()

    df = pd.read_csv(args.anot)
    if args.videos:
        df = df[df["video_key"].isin(args.videos)]
    if df.empty:
        print("no hay filas que puntuar")
        return

    for vk in sorted(df["video_key"].unique()):
        sub = df[df["video_key"] == vk]
        n = len(sub)
        aciertos = int((sub["nueva_cls"] == sub["cls_original"]).sum())
        vacios = int((sub["nueva_cls"] == SIN_GOLPE).sum())
        dudosos = int((sub["nueva_cls"] == DUDOSO).sum())
        con_golpe = n - vacios - dudosos

        print(f"\n=== {vk} ===")
        if n < args.n:
            print(f"  PARCIAL: {n} de {args.n} anotados, la decision no se aplica todavia")
        elif n > args.n:
            print(f"  OJO: {n} filas, mas que los {args.n} esperados (duplicados?)")

        print(f"  aciertos            {aciertos}/{n}  ({100 * aciertos / n:.1f}% sobre la muestra entera)")
        if con_golpe:
            # Segundo denominador: solo los clips que contienen un golpe. Mide
            # calidad de clase aislando los fallos de segmentacion.
            print(f"  aciertos utiles     {aciertos}/{con_golpe}  "
                  f"({100 * aciertos / con_golpe:.1f}% sobre los que contienen un golpe)")
        print(f"  sin golpe           {vacios}  ({100 * vacios / n:.1f}%, fallo de segmentacion)")
        if dudosos:
            print(f"  dudosos             {dudosos}  (sin resolver)")
        print(f"  piso Wilson 95%     {100 * wilson_low(aciertos, n):.1f}%  "
              f"(sobre la muestra entera)")

        if n < args.n:
            continue
        if n != TABLA_N:
            print(f"  DECISION            no aplica: la tabla pre-registrada es sobre "
                  f"{TABLA_N} clips y aca hay {n}")
            if args.detalle:
                _detalle(sub)
            continue

        # Los dudosos son la unica ambiguedad honesta: se acota por los dos lados.
        peor, mejor = decision(aciertos), decision(aciertos + dudosos)
        if peor == mejor:
            print(f"  DECISION            {peor}")
        else:
            print(f"  DECISION            INDETERMINADA: entre {peor} (dudosos fallan) "
                  f"y {mejor} (dudosos aciertan)")
            print(f"                      resolver los {dudosos} dudosos antes de decidir")

        if args.detalle:
            _detalle(sub)

    # Agregado: sirve para comparar contra el 16.7% de azar de 6 clases.
    n = len(df)
    aciertos = int((df["nueva_cls"] == df["cls_original"]).sum())
    print(f"\n=== agregado ===")
    print(f"  {aciertos}/{n} = {100 * aciertos / n:.1f}%, piso Wilson 95% "
          f"{100 * wilson_low(aciertos, n):.1f}%  (azar con 6 clases: 16.7%)")


if __name__ == "__main__":
    main()
