"""
BoxTwin - Lectura de E8b, E10, E11 y E12 contra su pre-registro (T11, 30-10-2026).

POR QUE EXISTE
  Las reglas de lectura estan escritas antes de medir (docs/specs/2026-10-30-mejoras-post-e8/,
  §4 o §5 de cada pre-registro). Si se aplican a mano, terminan aplicandose a ojo despues de
  ver el numero. Aca se aplican como estan escritas, con pruebas en los bordes, y el auditor las
  recalcula por su lado.

QUE HACE
  - Detector (P13a, P14a, P16a): F1 del ensamble en 0,80, media de los seis folds comparables,
    con la misma lectura de reportes que E8 (lectura_e8._resumen_detector). Mejora con +0,03 y
    5 de 6 folds ganados.
  - E4 (P13b, P14b, P16e): familia del ensamble, la regla de E7 y E8 (evaluar.veredicto_mejora,
    5 de 7, desvio entre semillas <= 0,03 o no concluyente).
  - E8b suma dos categorias (01 §4): iguala si la diferencia queda a menos de 0,03, peor si queda
    0,03 o mas abajo. Lo que supera 0,03 sin los folds queda "no demostrada".
  - E11 (03 §5): mejora si la precision por cuadro sube 0,03 o mas y la cobertura no baja mas de
    0,02; empeora si la precision baja 0,03 o mas o la cobertura baja mas de 0,05; si no, sin
    diferencia demostrada.
  - Las diferencias sin umbral (P13c, P14c) van con su direccion declarada.
  Lo que no esta medido queda "no realizado". Escribe <mejoras>/<exp>/veredictos.json.

USO
  python scripts/mejoras/lectura.py e8b
  python scripts/mejoras/lectura.py e11 --mejoras ~/boxtwin-mejoras
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import rutas  # noqa: E402

sys.path.insert(0, str(rutas.REPO / "scripts" / "clasificador_geo"))
import lectura_e8  # noqa: E402

CURVA = lectura_e8.CURVA
DIF_MIN = 0.03
EPS = 1e-9
FOLDS_DETECTOR = 5     # de 6
FOLDS_E4 = 5           # de 7
E11_PRECISION = 0.03
E11_COBERTURA_MEJORA = 0.02
E11_COBERTURA_EMPEORA = 0.05
NO_REALIZADO = {"veredicto": "no realizado"}


def regla(total_nuevo: float, total_base: float, folds_nuevo: dict, folds_base: dict,
          folds_min: int, desvios: list[float] | None = None, tres: bool = False) -> dict:
    """La regla de E7 y E8; con `tres`, las categorias iguala y peor de E8b."""
    v = lectura_e8._evaluar().veredicto_mejora(total_nuevo, total_base, folds_nuevo, folds_base,
                                               folds_min, desvios)
    if tres and v["veredicto"] == "no demostrada":
        d = v["diferencia"]
        if abs(d) < DIF_MIN - EPS:
            v["veredicto"] = "iguala"
        elif d <= -DIF_MIN + EPS:
            v["veredicto"] = "peor"
    return v


def detector(base: dict, nuevo: dict, tres: bool = False) -> dict:
    b, n = lectura_e8._resumen_detector(base), lectura_e8._resumen_detector(nuevo)
    v = regla(n["f1_080"], b["f1_080"], n["f1_080_por_fold"], b["f1_080_por_fold"],
              FOLDS_DETECTOR, None, tres)
    return {**v, "base": b, "nuevo": n}


def e4(base: dict, nuevo: dict, tres: bool = False) -> dict:
    desv = [d for d in (lectura_e8._desvio(base), lectura_e8._desvio(nuevo)) if d is not None]
    v = regla(nuevo["ensamble_total"]["familia"]["exactitud"], base["ensamble_total"]["familia"]["exactitud"],
              lectura_e8._fam_por_fold(nuevo), lectura_e8._fam_por_fold(base), FOLDS_E4, desv or None, tres)
    return {**v, "base": lectura_e8._extras(base), "nuevo": lectura_e8._extras(nuevo)}


def regla_e11(con: dict, sin: dict) -> dict:
    dp = con["precision"] - sin["precision"]
    dc = con["cobertura"] - sin["cobertura"]
    if dp <= -E11_PRECISION + EPS or dc < -E11_COBERTURA_EMPEORA - EPS:
        v = "empeora"
    elif dp >= E11_PRECISION - EPS and dc >= -E11_COBERTURA_MEJORA - EPS:
        v = "mejora"
    else:
        v = "sin diferencia demostrada"
    return {"veredicto": v, "diferencia_precision": dp, "diferencia_cobertura": dc,
            "con_corte": con, "sin_corte": sin}


def diferencia(det_a, det_b, e4_a, e4_b) -> dict:
    """a menos b, en F1 a 0,80 y en familia, sin umbral. Direccion declarada: positiva."""
    r: dict = {"direccion_declarada": "positiva"}
    if det_a and det_b:
        r["f1_080"] = (lectura_e8._resumen_detector(det_a)["f1_080"]
                       - lectura_e8._resumen_detector(det_b)["f1_080"])
    if e4_a and e4_b:
        r["familia"] = (e4_a["ensamble_total"]["familia"]["exactitud"]
                        - e4_b["ensamble_total"]["familia"]["exactitud"])
    return r if len(r) > 1 else dict(NO_REALIZADO)


def _leer(p: Path):
    return json.loads(Path(p).read_text()) if Path(p).is_file() else None


def _cond(raiz: Path) -> tuple[dict | None, dict | None]:
    """folds.json del detector y E4.json de una condicion."""
    return _leer(raiz / "folds.json"), _leer(raiz / "clasificador-geo" / "resultados" / "E4.json")


def _si(f, *args, **kw):
    return f(*args, **kw) if all(a is not None for a in args) else dict(NO_REALIZADO)


def leer(exp: str, mejoras: Path, e8: Path) -> dict:
    det_v8, e4_v8 = _cond(e8 / "base")
    det_y26, e4_y26 = _cond(e8 / "y26")
    ver: dict = {"exp": exp}
    if exp == "e8b":
        det, e4_ = _cond(mejoras / "e8b" / "y26cal")
        ver["P13a"] = _si(detector, det_v8, det, tres=True)
        ver["P13b"] = _si(e4, e4_v8, e4_, tres=True)
        ver["P13c"] = diferencia(det, det_y26, e4_, e4_y26)
    elif exp == "e10":
        det, e4_ = _cond(mejoras / "e10" / "v8r")
        ver["P14a"] = _si(detector, det_v8, det)
        ver["P14b"] = _si(e4, e4_v8, e4_)
        det_c, e4_c = _cond(mejoras / "e8b" / "y26cal")
        det_cr, e4_cr = _cond(mejoras / "e10" / "y26calr")
        ver["P14c"] = diferencia(det_cr, det_c, e4_cr, e4_c)
        if "f1_080" in ver["P14c"] and "f1_080" in (d := diferencia(det, det_v8, None, None)):
            ver["P14c"]["mayor_que_v8"] = ver["P14c"]["f1_080"] > d["f1_080"]
    elif exp == "e12":
        det, e4_ = _cond(mejoras / "e12" / "v8g")
        ver["P16a"] = _si(detector, det_v8, det)
        ver["P16e"] = _si(e4, e4_v8, e4_)
    elif exp == "e11":
        conf = _leer(mejoras / "e11" / "confirmacion.json")
        ver["P15ab"] = (regla_e11(conf["con_corte"]["total"], conf["sin_corte"]["total"])
                        if conf else dict(NO_REALIZADO))
        ver["desarrollo"] = _leer(mejoras / "e11" / "desarrollo.json") or dict(NO_REALIZADO)
        if not conf:
            ver["nota"] = "sin material de confirmacion: E11 se informa como exploratorio (03 §4)"
    else:
        raise ValueError(f"experimento desconocido: {exp}")
    return ver


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("exp", choices=["e8b", "e10", "e11", "e12"])
    ap.add_argument("--mejoras", type=Path, default=rutas.MEJORAS)
    ap.add_argument("--e8", type=Path, default=rutas.E8)
    args = ap.parse_args(argv)
    ver = leer(args.exp, args.mejoras, args.e8)
    out = args.mejoras / args.exp / "veredictos.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ver, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: (v.get("veredicto") if isinstance(v, dict) else v) for k, v in ver.items()},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
