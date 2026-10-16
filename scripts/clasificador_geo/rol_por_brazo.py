"""
BoxTwin - E6: el rol (mano adelantada o atrasada) sacado del brazo del detector y la guardia.

POR QUE EXISTE
  En produccion jab y cross salen del argmax del clasificador sobre seis clases, y fuera de
  distribucion ese clasificador acierta el rol en 0,656, por debajo de decir siempre adelantada
  (E1, 15-10). Pero el detector ya trabaja por brazo: con el brazo y la guardia el rol sale de
  una regla, sin modelo. E6 mide si esa regla acierta mas (pre-registro, P7).

  `pares_nivel2` de medir_objetivo.py empareja por carril (peleador y brazo), y asi no puede ver
  un brazo cambiado: un golpe del detector en el brazo equivocado no se empareja con nada. Aca se
  empareja por peleador, sin separar por brazo, y despues se compara el brazo.

QUE HACE
  Para las cuatro sesiones de C8 nivel 2: orientacion A/B con medir_sesion (como pares_nivel2),
  emparejamiento por peleador a IoU 0,3 de todos los golpes de la Fight-Card contra los anotados
  completos, acuerdo de brazo sobre todos los emparejados, y sobre los rectos anotados: rol por
  la regla (arm_role con la guardia del evento anotado y el brazo del detector), por la regla con
  todos ortodoxos, y por el clasificador, contra el rol anotado.

  El clasificador: las Fight-Cards de C8 nivel 2 se completaron sin clasificador (tipo en None
  en los 1285 golpes), asi que se usa E1, que es la configuracion de produccion medida con la
  fuente afuera (pred.pkl del 15-10, semilla 0): el argmax de sus seis clases sobre el evento
  anotado emparejado; un hook o un uppercut sobre un recto anotado es error de rol. Ver
  TAREAS.md, decisiones.

  Escribe <trabajo>/resultados/E6.json con exactitud y Wilson, total y por fuente, y el veredicto.

USO
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/rol_por_brazo.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import REPO, TRABAJO, TRABAJO_E1E3, comun  # noqa: E402

SESIONES = Path("~/boxtwin/mediciones/c8-nivel2").expanduser()
# sesion de C8 nivel 2 -> fuente
SESION_FUENTE = {"03-sparring": "03-sparring", "04-sparring": "04-sparring",
                 "sparring-3": "sparring-3-rounds", "pacquiao_margarito": "pacquiao_margarito"}
IOU = 0.3
DIF_MIN = 0.05
DE_LADO = {"izq": "left", "der": "right"}


def rol_regla(guardia: str, lado: str) -> str:
    """Adelantado si (ortodoxa y brazo izquierdo) o (zurda y brazo derecho). Via core.types.arm_role."""
    from boxtwin.core.types import Guard, Side, arm_role
    return arm_role(Guard(guardia), Side(lado)).value


def emparejar_por_peleador(golpes_fc: dict[str, list[dict]], eventos: list[dict], invertido: bool,
                           emparejar=None) -> list[tuple[dict, dict, str]]:
    """
    (golpe de la Fight-Card, evento anotado, peleador) emparejados por solapamiento, por
    peleador y sin separar por brazo. Solo eventos completos.
    """
    if emparejar is None:
        from boxtwin.core.agreement import emparejar
    por_p: dict[str, list[dict]] = {}
    for e in eventos:
        if e.get("completeness") != "full":
            continue
        f = e["fighter"][-1]
        if invertido:
            f = "B" if f == "A" else "A"
        por_p.setdefault(f, []).append(e)
    pares = []
    for p in sorted(set(por_p) | set(golpes_fc)):
        a = sorted(por_p.get(p, []), key=lambda e: (e["start_frame"], e["id"]))
        m = sorted(golpes_fc.get(p, []), key=lambda g: (g["cuadro_inicio"], g["id"]))
        parejas, _, _ = emparejar([NS(start_frame=e["start_frame"], end_frame=e["end_frame"]) for e in a],
                                  [NS(start_frame=g["cuadro_inicio"], end_frame=g["cuadro_fin"]) for g in m],
                                  min_iou=IOU)
        pares.extend((m[j], a[i], p) for i, j, _ in parejas)
    return pares


def evaluar_pares(pares: list[tuple[dict, dict, str]], rol_clasificador: dict[str, str | None],
                  fuente: str, rol_regla_fn=rol_regla) -> dict:
    """
    Acuerdo de brazo sobre todos los pares y exactitud de rol sobre los rectos anotados.
    rol_clasificador: frame_dir -> 'lead', 'rear' o None (el clasificador no dijo recto).
    """
    from boxtwin.core.types import Guard, Side, arm_role
    brazo = [(DE_LADO[g["brazo"]], e["side"]) for g, e, _ in pares]
    filas = []
    for g, e, _ in pares:
        if e["punch_type"] != "straight":
            continue
        lado_det = DE_LADO[g["brazo"]]
        verdad = arm_role(Guard(e["guard"]), Side(e["side"])).value
        filas.append({"frame_dir": f"{fuente}_{e['id']}", "verdad": verdad,
                      "regla": rol_regla_fn(e["guard"], lado_det),
                      "ortodoxos": rol_regla_fn("orthodox", lado_det),
                      "clasificador": rol_clasificador.get(f"{fuente}_{e['id']}"),
                      "brazo_ok": lado_det == e["side"]})
    return {"pares": len(pares), "brazo": brazo, "rectos": filas}


def exactitud_wilson(filas: list[dict], clave: str) -> dict:
    n = len(filas)
    a = sum(1 for f in filas if f[clave] == f["verdad"])
    return {"n": n, "aciertos": a, "exactitud": (a / n if n else None), "wilson95": comun.wilson(a, n)}


def resumen(filas: list[dict], brazo: list[tuple[str, str]]) -> dict:
    ac = sum(1 for d, a in brazo if d == a)
    return {"acuerdo_brazo": {"n": len(brazo), "aciertos": ac, "exactitud": ac / len(brazo) if brazo else None,
                              "wilson95": comun.wilson(ac, len(brazo))},
            "rectos": len(filas),
            "rol_regla": exactitud_wilson(filas, "regla"),
            "rol_todos_ortodoxos": exactitud_wilson(filas, "ortodoxos"),
            "rol_clasificador": exactitud_wilson(filas, "clasificador"),
            "siempre_adelantada": exactitud_wilson([{**f, "c": "lead"} for f in filas], "c")}


def veredicto(regla: float | None, clasificador: float | None) -> dict:
    if regla is None or clasificador is None:
        return {"veredicto": "no realizado"}
    dif = regla - clasificador
    return {"veredicto": "mejora" if dif >= DIF_MIN - 1e-9 else "no demostrada", "diferencia": dif, "minimo": DIF_MIN}


def roles_e1() -> dict[str, str | None]:
    """frame_dir -> rol del argmax de E1 si dijo recto, None si dijo hook o uppercut."""
    spec = importlib.util.spec_from_file_location("evaluar_e1e3", REPO / "scripts" / "clasificador" / "evaluar.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    base = TRABAJO_E1E3
    clases = json.loads((base / "folds" / "E1" / "manifest.json").read_text())["clases"]
    salida = {}
    for f in SESION_FUENTE.values():
        nombres, y = ev.esperado(base / "folds" / "E1" / f"fold_{f}.pkl")
        P, gt = ev.leer_pred(base / "corridas" / "E1" / f / "s0" / "pred.pkl")
        if gt is None or not np.array_equal(gt, y):
            raise SystemExit(f"E1/{f}: el orden del pred.pkl no sigue el fold")
        for n, i in zip(nombres, P.argmax(1)):
            p = comun.partes_de_clase(clases[i])
            salida[n] = p["eje"] if p["tipo"] == "straight" else None
    return salida


def main(argv=None) -> int:
    from boxtwin.mvp.medicion import medir_sesion
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    faltan = [s for s in SESION_FUENTE if not (SESIONES / s / "fightcard.json").exists()]
    dest = TRABAJO / "resultados"
    dest.mkdir(parents=True, exist_ok=True)
    if faltan:   # D4
        (dest / "E6.json").write_text(json.dumps({"exp": "E6", "veredicto": "no realizado",
                                                  "motivo": f"faltan sesiones de C8 nivel 2: {faltan}"}, indent=2) + "\n")
        print(f"E6 no realizado: faltan {faltan}")
        return 0
    rol_cl = roles_e1()
    por_fuente, todas, brazos, sin_tipo = {}, [], [], 0
    for sesion, fuente in SESION_FUENTE.items():
        fc = json.loads((SESIONES / sesion / "fightcard.json").read_text())
        annot = json.loads(comun.ruta_annot(fuente).read_text())
        inv = medir_sesion(fc, annot).invertido
        golpes = {p: d["golpes"] for p, d in fc["peleadores"].items()}
        sin_tipo += sum(1 for gs in golpes.values() for g in gs if g.get("tipo") is None)
        r = evaluar_pares(emparejar_por_peleador(golpes, annot["events"], inv), rol_cl, fuente)
        por_fuente[fuente] = {"roles_invertidos": inv, "golpes_fightcard": sum(len(v) for v in golpes.values()),
                              "pares": r["pares"], **resumen(r["rectos"], r["brazo"])}
        todas.extend(r["rectos"])
        brazos.extend(r["brazo"])
    total = resumen(todas, brazos)
    out = {"exp": "E6", "que": "rol = arm_role(guardia anotada, brazo del detector) contra el clasificador",
           "clasificador": "E1 (configuracion de produccion con la fuente afuera, s0, 15-10): las Fight-Cards "
                           f"de C8 nivel 2 no traen tipo ({sin_tipo} golpes con tipo None)",
           "advertencia": "el detector de produccion se entreno con estas fuentes: el acuerdo de brazo es optimista",
           "iou": IOU, "total": total, "por_fuente": por_fuente,
           "P7": veredicto(total["rol_regla"]["exactitud"], total["rol_clasificador"]["exactitud"]),
           "filas": todas}
    out = comun.redondear(out)
    (dest / "E6.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"E6: {len(todas)} rectos emparejados, {len(brazos)} pares -> resultados/E6.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
