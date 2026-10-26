"""
BoxTwin - BM2: inspeccion de BoxingWeb antes de usarlo.

POR QUE EXISTE
  Este proyecto ya se comio dos errores por creerle a una fuente (los fps del paper de
  BoxingVI y la calidad de sus anotaciones). Antes de re-estimar BoxingWeb hay que verificar
  lo que se supone de el: cuantas rondas hay por particion, que trae cada ronda, si el
  paquete tiene terminos de uso, y que es pose_gt.pkl. La historia del repositorio de BoxMind
  sugiere que pose_gt es 4D-Humans con la identidad verificada y uve.pkl la salida del
  seguidor automatico (pre-registro, seccion 7): si es asi, en una ronda los dos coinciden
  cuadro a cuadro salvo donde el seguidor cruzo rojo y azul.

QUE HACE
  Recorre data_train y data_test; por ronda: archivos, claves y formas de cada pkl, cantidad
  de eventos y cuantos pasan el filtro de largo. Busca archivos de licencia o terminos en la
  carpeta de datos y su padre. En las rondas con los dos pkl, compara el 2D rojo de pose_gt
  contra el rojo y el azul de uve cuadro por cuadro (distancia media en pixeles normalizados
  sobre las articulaciones comunes) y cuenta en que fraccion de cuadros el rojo de pose_gt esta
  mas cerca del azul de uve (identidad cruzada). Escribe <trabajo_bm>/inspeccion.json.
  No decide nada: la lectura va a Decisiones de TAREAS.md.

USO (en twinboxing_env)
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/boxmind_prueba/inspeccionar.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bm import BOXINGWEB, TRABAJO_BM, fuera_del_repo  # noqa: E402
from boxingweb import eventos_validos, rondas  # noqa: E402

PATRONES_TERMINOS = ("*licen*", "*LICEN*", "*term*", "*TERM*", "*readme*", "*README*")


def describir(obj) -> dict:
    """Claves y formas de un pkl de pose."""
    if isinstance(obj, dict):
        return {str(k): list(np.shape(v)) for k, v in obj.items()}
    return {"tipo": type(obj).__name__, "forma": list(np.shape(obj))}


def comparar_identidad(gt: dict, uve: dict) -> dict | None:
    """
    Distancia media por cuadro del 2D rojo de gt contra rojo y azul de uve, sobre las primeras
    articulaciones comunes y los primeros dos canales. `cruzado` es la fraccion de cuadros en
    que el rojo de gt queda mas cerca del azul de uve.
    """
    try:
        g = np.asarray(gt["pose_red_2d"], np.float32)
        ur = np.asarray(uve["pose_red_2d"], np.float32)
        ub = np.asarray(uve["pose_blue_2d"], np.float32)
    except (KeyError, TypeError, ValueError):
        return None
    T = min(len(g), len(ur), len(ub))
    J = min(g.shape[1], ur.shape[1], ub.shape[1])
    if T == 0 or J == 0:
        return None
    g, ur, ub = g[:T, :J, :2], ur[:T, :J, :2], ub[:T, :J, :2]
    d_r = np.nanmean(np.linalg.norm(g - ur, axis=-1), axis=1)
    d_b = np.nanmean(np.linalg.norm(g - ub, axis=-1), axis=1)
    ok = np.isfinite(d_r) & np.isfinite(d_b)
    if not ok.any():
        return None
    return {"cuadros": int(ok.sum()), "articulaciones_comparadas": int(J),
            "dist_rojo_rojo_mediana": float(np.median(d_r[ok])),
            "dist_rojo_azul_mediana": float(np.median(d_b[ok])),
            "cruzado": float((d_b[ok] < d_r[ok]).mean()),
            "identicos": float((d_r[ok] < 1e-6).mean())}


def terminos(base: Path) -> list[str]:
    encontrados = set()
    for carpeta in (base, base.parent):
        for patron in PATRONES_TERMINOS:
            for p in carpeta.glob(patron):
                if p.is_file():
                    encontrados.add(str(p))
    return sorted(encontrados)


def main() -> int:
    salida = fuera_del_repo(TRABAJO_BM)
    salida.mkdir(parents=True, exist_ok=True)
    informe = {"raiz": str(BOXINGWEB), "terminos_o_licencia": terminos(Path(BOXINGWEB)),
               "rondas": {}, "por_particion": {}}
    for ronda, particion, carpeta in rondas():
        r = {"particion": particion, "archivos": sorted(p.name for p in carpeta.iterdir() if p.is_file())}
        ev = carpeta / "video_event.json"
        if ev.is_file():
            eventos = json.loads(ev.read_text())
            r["eventos"] = len(eventos)
            r["eventos_filtro_largo"] = len(eventos_validos(eventos))
            r["claves_evento"] = sorted(eventos[0]) if eventos else []
        pkls = {}
        for p in sorted(carpeta.glob("*.pkl")):
            with p.open("rb") as fh:
                pkls[p.name] = pickle.load(fh)
            r.setdefault("pkl", {})[p.name] = describir(pkls[p.name])
        gt = next((v for k, v in pkls.items() if k.endswith("pose_gt.pkl")), None)
        uv = next((v for k, v in pkls.items() if k.endswith("uve.pkl")), None)
        if gt is not None and uv is not None:
            r["gt_contra_uve"] = comparar_identidad(gt, uv)
        informe["rondas"][ronda] = r
        c = informe["por_particion"].setdefault(particion, {"rondas": 0, "eventos_filtro_largo": 0})
        c["rondas"] += 1
        c["eventos_filtro_largo"] += r.get("eventos_filtro_largo", 0)
    comps = [r["gt_contra_uve"] for r in informe["rondas"].values() if r.get("gt_contra_uve")]
    informe["resumen_gt_contra_uve"] = (
        {"rondas_comparadas": len(comps),
         "cruzado_mediana": float(np.median([c["cruzado"] for c in comps])),
         "identicos_mediana": float(np.median([c["identicos"] for c in comps]))} if comps else None)
    (salida / "inspeccion.json").write_text(json.dumps(informe, indent=1) + "\n")
    print(json.dumps({k: informe[k] for k in ("terminos_o_licencia", "por_particion", "resumen_gt_contra_uve")},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
