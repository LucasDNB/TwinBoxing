"""
BoxTwin - C4 y C5 por recorte de sesiones (cierre del 4-11-2026).

POR QUE EXISTE
  El pre-registro de C5 habla de "las sesiones de prueba" y ningun documento fijo cuales son. C5 se
  corrio sobre las siete sesiones desde el 27-10 antes de escribir esto, asi que no se elige un
  recorte: se informan todos (docs/specs/2026-11-04-cierre/00-preregistro.md §3).

QUE HACE
  Lee el JSON de medir_c5.py y la tabla trabajos en solo lectura. Por recorte da la mediana del
  factor y si queda en 2,0 o menos; para C4 cuenta sesiones que llegaron a Fight-Card y trabajos con
  error. Los ids salen cortados a 8 caracteres y no escribe nombres.

USO
  python scripts/mejoras/recortes_c5.py --c5 c5.json --db ~/boxtwin/datos/boxtwin.db -o salida.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from pathlib import Path

UMBRAL = 2.0
CUENTA_FUENTES = "487f4ecfcea240cd972db04fc76d0334"     # la que cargo las fuentes de desarrollo
TRANSMISION = "3b82b7f8"


def recorte(sesiones: list[dict]) -> dict:
    f = [s["factor"] for s in sesiones]
    m = statistics.median(f) if f else None
    return {"n": len(f), "mediana": m, "maximo": max(f, default=None), "pasa": m is not None and m <= UMBRAL}


def recortes(sesiones: list[dict]) -> dict:
    llegan = [s for s in sesiones if s["llego_a_fightcard"]]
    largas = [s for s in llegan if s["duracion_s"] >= 170]
    return {"todas": recorte(llegan),
            "otras_cuentas": recorte([s for s in llegan if s["cuenta"] != CUENTA_FUENTES]),
            "sin_transmision": recorte([s for s in llegan if s["id"] != TRANSMISION]),
            "tres_minutos_o_mas": recorte(largas),
            "tres_minutos_o_mas_sin_transmision": recorte([s for s in largas if s["id"] != TRANSMISION])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c5", type=Path, required=True)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("-o", "--salida", type=Path, required=True)
    a = ap.parse_args(argv)
    con = sqlite3.connect(f"file:{a.db.expanduser()}?mode=ro", uri=True)
    sesiones = []
    errores = 0
    for s in json.loads(a.c5.expanduser().read_text())["sesiones"]:
        cuenta = con.execute("select usuario_id from sesiones where id=?", (s["sesion"],)).fetchone()[0]
        malos = con.execute("select count(*) from trabajos where sesion_id=? and (estado<>'listo' or error is not null)",
                            (s["sesion"],)).fetchone()[0]
        errores += malos
        sesiones.append({**s, "id": s["sesion"][:8], "cuenta": cuenta, "trabajos_con_error": malos})
    r = {"c5": recortes(sesiones), "c4": {"sesiones": len(sesiones), "llegaron": sum(s["llego_a_fightcard"] for s in sesiones),
                                          "trabajos_con_error": errores},
         "sesiones": [{k: s[k] for k in ("id", "duracion_s", "segundos", "factor", "solapada", "llego_a_fightcard",
                                         "trabajos_con_error")} for s in sesiones]}
    a.salida.write_text(json.dumps(r, indent=1))
    print(json.dumps({k: {x: v[x] for x in ("n", "mediana", "maximo", "pasa")} for k, v in r["c5"].items()}, indent=1), r["c4"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
