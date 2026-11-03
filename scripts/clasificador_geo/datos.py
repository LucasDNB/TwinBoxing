"""
BoxTwin - Golpes completos por fuente para E4 a E6, con su ventana y sus etiquetas.

POR QUE EXISTE
  E1 a E3 entraron por los pickles de mmaction, que ya traen la ventana cortada en pixeles.
  E4 necesita otra cosa: las features del detector se calculan sobre el video entero del
  peleador (las derivadas necesitan vecinos) y despues se corta la ventana, y el rival tiene
  que estar cuadro a cuadro. Ademas E5 eleva a 3D en el entorno boxtwin_mmaction, que no tiene
  el paquete boxtwin instalado. Por eso las series por peleador se resuelven una sola vez aca,
  con la identidad anotada, y se guardan en un npz por fuente que lee cualquiera de los dos
  entornos. Los golpes tienen que ser exactamente los de E1 (mismo frame_dir), porque la
  comparacion del pre-registro es golpe contra golpe; si no lo son, se frena (D3).

QUE HACE
  Por fuente: `series_por_peleador` (la misma resolucion que el export `sequence` y que
  medir_objetivo.py), y de cada evento con completeness == full: peleador, lado, tipo, altura,
  rol (`Event.arm_role`), ventana [inicio - 8, fin + 8] recortada al video. Escribe
  <trabajo>/datos/series/<fuente>.npz, <trabajo>/datos/golpes.csv y golpes.meta.json (con el
  sha256 de cada anotacion leida). Despues chequea 1047 golpes, conteo por fuente y el mismo
  conjunto de frame_dir que los folds de E1; si algo no coincide, sale con codigo 2.

USO
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/datos.py
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/datos.py --solo-chequear
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import GOLPES_POR_FUENTE, PAD, TOTAL_GOLPES, TRABAJO, TRABAJO_E1E3, comun  # noqa: E402

CAMPOS = ("frame_dir", "fuente", "event_id", "fighter", "p", "side", "tipo", "altura", "rol",
          "guardia", "fps", "start_frame", "end_frame", "ini", "fin")


def _v(x):
    """Valor de un enum o el dato tal cual: las pruebas pasan strings."""
    return getattr(x, "value", x)


def ventana(inicio: int, fin: int, T: int, pad: int = PAD) -> tuple[int, int]:
    """[inicio - pad, fin + pad] recortado a [0, T - 1]. Inclusivo en los dos extremos."""
    return max(0, int(inicio) - pad), min(T - 1, int(fin) + pad)


def golpes_de_eventos(eventos, fuente: str, T: int, fps: float) -> list[dict]:
    """Un registro por golpe completo. Amagues y abortados quedan afuera, como en E1."""
    salida = []
    for ev in eventos:
        if _v(ev.completeness) != "full":
            continue
        ini, fin = ventana(ev.start_frame, ev.end_frame, T)
        fighter = _v(ev.fighter)
        salida.append({
            "frame_dir": f"{fuente}_{ev.id}", "fuente": fuente, "event_id": ev.id,
            "fighter": fighter, "p": 0 if fighter == "fighter_A" else 1,
            "side": _v(ev.side), "tipo": _v(ev.punch_type), "altura": _v(ev.target),
            "rol": _v(ev.arm_role), "guardia": _v(ev.guard), "fps": float(fps),
            "start_frame": int(ev.start_frame), "end_frame": int(ev.end_frame),
            "ini": ini, "fin": fin,
        })
    return salida


def frame_dirs_de_folds(base: Path) -> set[str]:
    """Todos los frame_dir de prueba de los siete folds de E1 (cada golpe esta en uno)."""
    nombres: set[str] = set()
    for fuente in comun.FUENTES:
        with (base / f"fold_{fuente}.pkl").open("rb") as fh:
            nombres.update(pickle.load(fh)["split"]["test"])
    return nombres


def chequear(golpes: list[dict], esperados: set[str] | None,
             por_fuente: dict[str, int] = GOLPES_POR_FUENTE, total: int = TOTAL_GOLPES) -> list[str]:
    """Lista de diferencias contra el pre-registro y los folds de E1. Vacia si todo coincide."""
    errores = []
    nombres = [g["frame_dir"] for g in golpes]
    if len(nombres) != len(set(nombres)):
        errores.append(f"frame_dir repetidos: {len(nombres) - len(set(nombres))}")
    if len(golpes) != total:
        errores.append(f"golpes: {len(golpes)}, el pre-registro dice {total}")
    cuenta: dict[str, int] = {}
    for g in golpes:
        cuenta[g["fuente"]] = cuenta.get(g["fuente"], 0) + 1
    for f, n in por_fuente.items():
        if cuenta.get(f, 0) != n:
            errores.append(f"{f}: {cuenta.get(f, 0)} golpes, el pre-registro dice {n}")
    if esperados is None:
        errores.append("no estan los folds de E1 para comparar el conjunto de golpes")
    elif set(nombres) != esperados:
        sobran, faltan = set(nombres) - esperados, esperados - set(nombres)
        errores.append(f"conjunto distinto al de E1: sobran {len(sobran)}, faltan {len(faltan)}"
                       f" (p. ej. {sorted(sobran)[:2]} / {sorted(faltan)[:2]})")
    return errores


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def series_de_fuente(fuente: str):
    """
    kp (2, T, 17, 2), sc (2, T, 17), valid (2, T) con la identidad anotada, y el documento.
    Copia la resolucion de rutas de boxtwin-annotator/scripts/medir_objetivo.py.
    """
    from boxtwin.core.annotations import load
    from boxtwin.core.posecache import PoseCache
    from boxtwin.mvp.orquesta import series_por_peleador

    ruta = comun.ruta_annot(fuente)
    crudo = json.loads(ruta.read_text())
    doc, _ = load(ruta)
    npz = Path(crudo["pose"]["npz_path"])
    if not npz.is_file():          # el proyecto se pudo haber movido: la cache esta al lado
        npz = ruta.parents[1] / "cache" / npz.name
    cache = PoseCache.open(npz)
    kp, sc, valid, _ = series_por_peleador(doc, cache)
    return kp, sc, valid, doc


def escribir_csv(golpes: list[dict], destino: Path) -> None:
    with destino.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(golpes)


def leer_csv(origen: Path) -> list[dict]:
    enteros = ("p", "start_frame", "end_frame", "ini", "fin")
    with origen.open() as fh:
        filas = list(csv.DictReader(fh))
    for f in filas:
        for k in enteros:
            f[k] = int(f[k])
        f["fps"] = float(f["fps"])
    return filas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--solo-chequear", action="store_true", help="no recalcula, chequea golpes.csv")
    args = ap.parse_args(argv)
    destino = TRABAJO / "datos"
    (destino / "series").mkdir(parents=True, exist_ok=True)
    csv_path = destino / "golpes.csv"

    if args.solo_chequear:
        golpes = leer_csv(csv_path)
    else:
        golpes, shas = [], {}
        for fuente in comun.FUENTES:
            kp, sc, valid, doc = series_de_fuente(fuente)
            v = doc.video
            np.savez_compressed(destino / "series" / f"{fuente}.npz", kp=kp, sc=sc, valid=valid,
                                fps=v.fps, ancho=v.width, alto=v.height)
            gs = golpes_de_eventos(doc.events, fuente, kp.shape[1], v.fps)
            golpes.extend(gs)
            ruta = comun.ruta_annot(fuente)
            shas[str(ruta.relative_to(comun.RAIZ_FUENTES))] = _sha(ruta)
            print(f"{fuente}: {len(gs)} golpes completos, T={kp.shape[1]}, fps={v.fps:.3f}, "
                  f"cobertura A={valid[0].mean():.3f} B={valid[1].mean():.3f}")
        escribir_csv(golpes, csv_path)
        (destino / "golpes.meta.json").write_text(json.dumps(
            {"golpes": len(golpes), "pad": PAD, "anotaciones_sha256": shas}, indent=2) + "\n")

    base = TRABAJO_E1E3 / "folds" / "E1"
    esperados = frame_dirs_de_folds(base) if (base / "manifest.json").exists() else None
    errores = chequear(golpes, esperados)
    if errores:
        print("D3: el conjunto de golpes no coincide con el de E1\n  " + "\n  ".join(errores))
        return 2
    print(f"ok: {len(golpes)} golpes, mismo conjunto de frame_dir que los folds de E1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
