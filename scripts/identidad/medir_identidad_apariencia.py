"""
BoxTwin - IE7: verificador de apariencia (guante, remera, pantalon) con usuario simulado v2.

POR QUE EXISTE
  IE6 (3-11) dio "no estable": 1,54% de cuadros con rol equivocado que nunca llegan a la cola,
  casi todos en 03-sparring, donde el color del guante no separa. IE7 suma a la cola los
  tramos que guante, remera y pantalon contradicen (boxtwin/core/apariencia.py). Pre-registro:
  docs/specs/2026-11-02-identidad-estable/02-ie7-apariencia.md.

QUE HACE
  Por fuente (las seis de C1, la siembra de C1 y E11) y condicion (base, apariencia):
  - la evidencia de C1 mas la ropa medida con las funciones del producto (las muestras de
    diagnostico_senales_apariencia.py, mismos cuadros y mismo criterio de aislamiento);
  - usuario simulado v2: parte un item si la verdad tiene dos o mas roles entre A, B y
    ninguno con al menos 5 cuadros cada uno (2 decisiones); si no, contesta la mayoria (1);
  - rondas: la cola se recalcula con lo corregido, como el producto, hasta que no queda nada
    pendiente o 3 rondas;
  - error silencioso: cuadro con A o B equivocado en lo automatico, fuera de todo item de
    todas las rondas; precision y cobertura despues; decisiones por minuto; items por motivo.
  Informa tambien la medicion con el simulado de IE6 (una ronda) y audita base contra IE6.
  Aplica la regla del pre-registro y escribe un JSON.

USO
  python scripts/identidad/medir_identidad_apariencia.py --raiz ~/Proyectos/TwinBoxing \\
      --evidencia-dir ~/boxtwin/mediciones/c1 \\
      --muestras-dir ~/boxtwin/mediciones/identidad-ropa/muestras \\
      --out ~/boxtwin/mediciones/identidad-ropa/ie7.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import diagnostico_identidad_estable as diag  # noqa: E402
import medir_identidad_estable as m  # noqa: E402  (arma sys.path del repo)

A, B = m.A, m.B
MIN_LADO = m.MIN_LADO
MAX_RONDAS = 3
CONDICIONES = {"base": False, "apariencia": True}
E1_TOTAL, E1_FUENTE, E2_PRECISION, EMPATE = m.E1_TOTAL, m.E1_FUENTE, m.E2_PRECISION, 0.1
# La auditoria: base con el simulado de IE6 tiene que dar lo de IE6.
IE6_BASE = {"silenciosos": 1275, "precision_oraculo": 0.9698}


# ---------------------------------------------------------------------------
# Partes puras
# ---------------------------------------------------------------------------


def oraculo_v2(cola: list[dict], verdad: dict, frames_de) -> tuple[list, int]:
    """
    El usuario simulado del pre-registro de IE7: parte el item si la verdad tiene al menos dos
    roles entre A, B y ninguno con MIN_LADO cuadros cada uno (2 decisiones, se parta en dos o
    en mas); si no, contesta la mayoria (1 decision).
    """
    from boxtwin.mvp.revision import Correccion

    correcciones, decisiones = [], 0
    for it in cola:
        if not it.get("revision_pendiente"):
            continue
        t = int(it["track"])
        fs = frames_de(t)
        fs = np.asarray(fs if fs is not None else [])
        fs = fs[(fs >= it["desde"]) & (fs < it["hasta"])]
        if not len(fs):
            continue
        v = m._serie(verdad.get(t, []), fs)
        na, nb = int((v == A).sum()), int((v == B).sum())
        nn = len(fs) - na - nb
        if sum(n >= MIN_LADO for n in (na, nb, nn)) >= 2:
            decisiones += 2
            for d, h, rol in m._corridas(fs, v, it["hasta"]):
                correcciones.append(Correccion(t, d, h, rol))
            continue
        decisiones += 1
        rol = max((("A", na), ("B", nb), ("ninguno", nn)), key=lambda x: x[1])[0]
        correcciones.append(Correccion(t, int(it["desde"]), int(it["hasta"]), rol))
    return correcciones, decisiones


def revisar(auto, pendientes_de, oraculo, verdad, frames_de, max_rondas: int):
    """
    Rondas del usuario simulado. `pendientes_de(corregidos)` da los items pendientes de la cola
    recalculada. Devuelve los corregidos finales, todos los items vistos, decisiones, rondas.
    """
    from boxtwin.mvp.revision import aplicar_correcciones

    correcciones, vistos, decisiones, rondas = [], [], 0, 0
    corregidos = auto
    for _ in range(max_rondas):
        cola = pendientes_de(corregidos)
        if not cola:
            break
        rondas += 1
        nuevas, d = oraculo(cola, verdad, frames_de)
        vistos += cola
        decisiones += d
        correcciones += nuevas
        corregidos = aplicar_correcciones(auto, correcciones)
    return corregidos, vistos, decisiones, rondas


def regla(totales: dict[str, dict]) -> dict:
    """La regla del pre-registro de IE7."""
    ver = {}
    for c, t in totales.items():
        e = t["errores"]
        ver[c] = {"E1": e["fraccion_silenciosa"] <= E1_TOTAL and e["peor_fuente"] <= E1_FUENTE,
                  "E2": (t["oraculo"]["precision"] or 0.0) >= E2_PRECISION}
    banderas = {c: int(CONDICIONES[c]) for c in totales}
    pasan = [c for c in totales if ver[c]["E1"] and ver[c]["E2"]]
    if pasan:
        mejor = min(totales[c]["decisiones_por_minuto"] for c in pasan)
        emp = [c for c in pasan if totales[c]["decisiones_por_minuto"] - mejor <= EMPATE]
        elegida = min(emp, key=lambda c: (banderas[c], totales[c]["decisiones_por_minuto"]))
        estado = "estable"
    else:
        elegida = min(totales, key=lambda c: (totales[c]["errores"]["silenciosos"], banderas[c]))
        estado = "E1 si, E2 no" if any(v["E1"] for v in ver.values()) else "no estable"
    return {"veredictos": ver, "elegida": elegida, "estado": estado,
            "banderas": {"BOXTWIN_IDENTIDAD_APARIENCIA": banderas[elegida]}}


# ---------------------------------------------------------------------------
# Medicion
# ---------------------------------------------------------------------------


def sumar_ropa(ev: dict, muestras: list[dict]) -> int:
    """La ropa de las muestras de diagnostico, en la evidencia. Devuelve cuantas entraron."""
    n = 0
    for s in muestras:
        e = ev.get(int(s["t"]))
        if e is None or (s["remera"] is None and s["pantalon"] is None):
            continue
        e.ropa.append((int(s["f"]), tuple(s["remera"]) if s["remera"] else None,
                       tuple(s["pantalon"]) if s["pantalon"] else None, bool(s["aislado"])))
        n += 1
    for e in ev.values():
        e.ropa.sort(key=lambda r: r[0])
    return n


def medir_condicion(p: dict, con_apariencia: bool) -> dict:
    from boxtwin.core.apariencia import ajustar
    from boxtwin.core.identidad_auto import proponer
    from boxtwin.mvp.revision import cola_de_revision, tramos_de_propuesta

    ev, cfg, fps, verdad = p["ev"], p["cfg"], p["fps"], p["verdad"]
    fd = p["cache"].frames_of_track
    prop = proponer(ev, cfg, p["T"], fps, semillas=p["semillas"], cache=p["cache"],
                    cuadros_semilla=p["cuadro"])
    auto = tramos_de_propuesta(prop, ev, cfg)
    modelo = ajustar(ev, tuple(p["semillas"])) if con_apariencia else None

    def pendientes_de(corregidos):
        return [i for i in cola_de_revision(auto, corregidos, ev, prop.perfiles, cfg, fps,
                                            todos=False, apariencia=modelo)
                if i["revision_pendiente"]]

    salida = {"modelo": modelo.a_dict() if modelo else None}
    for nombre, oraculo, rondas_max in (("v2", oraculo_v2, MAX_RONDAS),
                                        ("ie6", m.oraculo, 1)):
        final, vistos, dec, rondas = revisar(auto, pendientes_de, oraculo, verdad, fd, rondas_max)
        por_motivo: dict[str, int] = {}
        for i in vistos:
            por_motivo[i["motivo"]] = por_motivo.get(i["motivo"], 0) + 1
        salida[nombre] = {
            "auto": m.por_cuadro(verdad, m.en_rangos(auto), fd),
            "errores": m.errores(verdad, m.en_rangos(auto), vistos, fd),
            "cola": {"items": len(vistos), "por_motivo": por_motivo,
                     "segundos": round(sum((i["hasta"] - i["desde"]) / fps for i in vistos), 1)},
            "rondas": rondas,
            "decisiones": dec,
            "oraculo": m.por_cuadro(verdad, m.en_rangos(final), fd),
        }
    return salida


def sumar(fuentes: list[dict], cond: str, simulado: str) -> dict:
    ms = [f[cond][simulado] for f in fuentes if cond in f]
    minutos = sum(f["minutos"] for f in fuentes if cond in f)
    s = lambda campo, k: sum(x[campo][k] for x in ms)  # noqa: E731
    sil, pel = s("errores", "silenciosos"), s("errores", "peleador")
    ob, oa, op = s("oraculo", "bien"), s("oraculo", "asignados"), s("oraculo", "peleador")
    ab, aa = s("auto", "bien"), s("auto", "asignados")
    dec = sum(x["decisiones"] for x in ms)
    motivos: dict[str, int] = {}
    for x in ms:
        for k, v in x["cola"]["por_motivo"].items():
            motivos[k] = motivos.get(k, 0) + v
    return {
        "fuentes": len(ms),
        "auto": {"precision": ab / aa if aa else None, "cobertura": ab / pel if pel else None},
        "errores": {"silenciosos": sil, "visibles": s("errores", "visibles"), "peleador": pel,
                    "fraccion_silenciosa": sil / pel if pel else None,
                    "peor_fuente": max(x["errores"]["fraccion_silenciosa"] or 0.0 for x in ms)},
        "cola": {"items": sum(x["cola"]["items"] for x in ms), "por_motivo": motivos,
                 "segundos": round(sum(x["cola"]["segundos"] for x in ms), 1)},
        "decisiones": dec,
        "decisiones_por_minuto": dec / minutos if minutos else None,
        "oraculo": {"precision": ob / oa if oa else None, "cobertura": ob / op if op else None},
    }


def main(argv=None) -> int:
    from medir_c1 import FUENTES

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--raiz", type=Path, default=Path.home() / "Proyectos" / "TwinBoxing")
    ap.add_argument("--evidencia-dir", type=Path,
                    default=Path.home() / "boxtwin" / "mediciones" / "c1")
    ap.add_argument("--muestras-dir", type=Path, default=Path.home() / "boxtwin" / "mediciones"
                    / "identidad-ropa" / "muestras")
    ap.add_argument("--out", type=Path, default=Path.home() / "boxtwin" / "mediciones"
                    / "identidad-ropa" / "ie7.json")
    args = ap.parse_args(argv)
    raiz, evd = args.raiz.expanduser(), args.evidencia_dir.expanduser()
    mdir = args.muestras_dir.expanduser()

    fuentes = []
    for f in FUENTES:
        video = raiz / f
        arch = mdir / f"{video.stem}.muestras.json"
        if not arch.is_file():
            raise FileNotFoundError(f"faltan las muestras de ropa {arch}: corre "
                                    "diagnostico_senales_apariencia.py")
        p = diag.preparar(video, evd)
        r = {"fuente": video.stem, "minutos": p["T"] / p["fps"] / 60.0,
             "ropa": sumar_ropa(p["ev"], json.loads(arch.read_text()))}
        if p["semillas"]:
            for cond, con in CONDICIONES.items():
                r[cond] = medir_condicion(p, con)
                x = r[cond]["v2"]
                print(f"{r['fuente']:>18} {cond:>10}: silenciosos {x['errores']['silenciosos']:5d} "
                      f"({x['errores']['fraccion_silenciosa']:.3%}), items {x['cola']['items']:4d}, "
                      f"rondas {x['rondas']}, decisiones {x['decisiones']:4d}, "
                      f"precision tras revisar {x['oraculo']['precision']:.3f}")
        fuentes.append(r)

    totales = {s: {c: sumar(fuentes, c, s) for c in CONDICIONES} for s in ("v2", "ie6")}
    b = totales["ie6"]["base"]
    auditoria = {"silenciosos": b["errores"]["silenciosos"],
                 "precision_oraculo": round(b["oraculo"]["precision"], 4),
                 "esperado": IE6_BASE}
    auditoria["coincide"] = (auditoria["silenciosos"] == IE6_BASE["silenciosos"]
                             and auditoria["precision_oraculo"] == IE6_BASE["precision_oraculo"])
    salida = {"etiqueta": "desarrollo: material donde se diseno IE7, no confirma",
              "spec": "docs/specs/2026-11-02-identidad-estable/02-ie7-apariencia.md",
              "auditoria": auditoria, "fuentes": fuentes, "totales": totales}
    if auditoria["coincide"]:
        salida["lectura"] = regla(totales["v2"])
        print("auditoria ok;", salida["lectura"])
    else:
        print("AUDITORIA FALLA, no se lee:", auditoria)
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(salida, indent=1, ensure_ascii=False))
    print(f"-> {out}")
    return 0 if auditoria["coincide"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
