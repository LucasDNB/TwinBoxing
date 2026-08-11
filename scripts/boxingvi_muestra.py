#!/usr/bin/env python3
"""
BoxingVI - Muestreo estratificado para verificacion ciega.

POR QUE EXISTE
  La verificacion manual mostro que la calidad de anotacion de BoxingVI varia por
  video y no correlaciona con el tamano: V2 tiene 232 clips y esta tan roto como
  V1 con 1863. No hay forma de inferir la calidad de V8, V9 y V10 sin mirarlos,
  y mirarlos enteros son 498 clips.
  La muestra tiene que quedar FIJADA antes de anotar. Si se resortea despues de
  ver resultados parciales, el criterio de decision pre-registrado (16+ usar /
  11-15 reanotar / 10- descartar) deja de valer y la verificacion no se puede
  defender en el Capitulo 4. Por eso el script es determinista por semilla y se
  niega a pisar una muestra ya escrita salvo --force explicito.

QUE HACE
  Toma el manifest, filtra los videos pedidos y sortea N clips por video con
  reparto estratificado: un piso de --min-por-clase para cada clase presente y el
  resto asignado por deficit proporcional, de modo que la muestra respete la
  distribucion real del video sin dejar clases minoritarias afuera. Los topes por
  disponibilidad se respetan (si una clase tiene menos clips que el piso, entra
  con los que tiene).
  Escribe un CSV con el MISMO esquema que manifest_filtrado.csv, para que
  boxingvi_annot.py lo consuma sin cambios. Las filas salen mezcladas: si la
  muestra quedara agrupada por clase, el orden filtraria la etiqueta y la
  anotacion dejaria de ser ciega.

USO
  # desde test-BoxingVI/
  python ../scripts/boxingvi_muestra.py \
      --csv ./clips/manifest_filtrado.csv \
      --videos V8 V9 V10 --n 18 --min-por-clase 2 \
      --out ./clips/muestra_v8v9v10.csv

  # y despues, para anotar la muestra:
  python ../scripts/boxingvi_annot.py --csv ./clips/muestra_v8v9v10.csv \
      --videos V8 V9 V10 --out ./clips/verificacion_v8v9v10.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def repartir(disponible: dict, n: int, minimo: int) -> dict:
    """Reparte n cupos entre clases respetando un piso y los topes de stock.

    Primero pone el piso en cada clase presente (o todo su stock si tiene menos).
    Los cupos que sobran se entregan de a uno a la clase con mayor deficit contra
    su cuota proporcional, que es lo que mantiene la muestra parecida al video.
    Empata por nombre de clase para que el resultado sea reproducible.
    """
    total = sum(disponible.values())
    asignado = {c: min(minimo, disponible[c]) for c in disponible}

    # No se puede sortear mas de lo que hay: el tope duro es el stock del video.
    n = min(n, total)

    def deficit(c):
        # Cuanto le falta a la clase para llegar a su cuota proporcional.
        # Desempata por nombre para que el reparto sea siempre el mismo.
        return (disponible[c] / total * n - asignado[c], c)

    while sum(asignado.values()) < n:
        # Solo compiten las clases que todavia tienen clips sin asignar.
        libres = [c for c in disponible if asignado[c] < disponible[c]]
        if not libres:
            break
        asignado[max(libres, key=deficit)] += 1

    return {c: k for c, k in asignado.items() if k > 0}


def main():
    ap = argparse.ArgumentParser(
        description="Muestreo estratificado por video para verificacion ciega.")
    ap.add_argument("--csv", default="./clips/manifest_filtrado.csv",
                    help="manifest de entrada")
    ap.add_argument("--videos", nargs="+", required=True,
                    help="video_keys a muestrear, ej: V8 V9 V10")
    ap.add_argument("--n", type=int, default=18,
                    help="clips por video (default 18)")
    ap.add_argument("--min-por-clase", type=int, default=2,
                    help="piso por clase presente (default 2)")
    ap.add_argument("--seed", type=int, default=42,
                    help="semilla del sorteo, queda registrada en el log")
    ap.add_argument("--out", default="./clips/muestra.csv")
    ap.add_argument("--force", action="store_true",
                    help="pisar una muestra ya existente (invalida el pre-registro)")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        sys.exit(
            f"ERROR: {out} ya existe.\n"
            "  La muestra ya fue sorteada y puede estar anotandose. Resortearla\n"
            "  ahora rompe el criterio de decision fijado de antemano.\n"
            "  Si de verdad queres rehacerla, borrala a mano o pasa --force.")

    df = pd.read_csv(args.csv)

    faltan = [v for v in args.videos if v not in set(df["video_key"])]
    if faltan:
        sys.exit(f"ERROR: estos video_key no estan en {args.csv}: {faltan}")

    partes = []
    print(f"\nmuestreo estratificado  n={args.n}  piso={args.min_por_clase}  "
          f"seed={args.seed}\n")

    for v in args.videos:
        sub = df[df["video_key"] == v]
        disponible = sub["cls"].value_counts().to_dict()
        plan = repartir(disponible, args.n, args.min_por_clase)

        print(f"{v}: {len(sub)} clips, {len(disponible)} clases presentes")
        for c in sorted(plan):
            print(f"    {c:<16} {plan[c]:>2} de {disponible[c]:>3}")
        total_v = sum(plan.values())
        if total_v < args.n:
            print(f"    aviso: solo se pudieron sortear {total_v} de {args.n}")
        print(f"    total {total_v}\n")

        for c, k in plan.items():
            # random_state por clase para que agregar un video no altere lo ya sorteado.
            partes.append(sub[sub["cls"] == c].sample(n=k, random_state=args.seed))

    muestra = pd.concat(partes)
    # Mezclado final: agrupada por clase, el orden delataria la etiqueta original.
    muestra = muestra.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    muestra.to_csv(out, index=False)

    print(f"escrito: {out}  ({len(muestra)} clips)")
    print(f"\nanotar con:\n  python ../scripts/boxingvi_annot.py --csv {out} \\")
    print(f"      --videos {' '.join(args.videos)} "
          f"--out ./clips/verificacion.csv\n")


if __name__ == "__main__":
    main()
