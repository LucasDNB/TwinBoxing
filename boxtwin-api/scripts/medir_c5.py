"""
BoxTwin - Medicion de C5: tiempo de procesamiento hasta la Fight-Card, sobre la estacion.

POR QUE EXISTE
  C5 (docs/specs/MVP.md §7) pide que la mediana del tiempo de procesamiento sobre las
  sesiones de prueba sea a lo sumo 2 veces la duracion del video. El 22-09 se midio 1,098x
  sobre un video de 2 minutos sin clasificar; el ensayo del 20-10 sobre las sesiones de la
  estacion dio 2,3x a 3,0x en seis de ocho. Que reloj cuenta quedo fijado en
  docs/specs/2026-10-20-c5-tiempo-preregistro.md antes de las sesiones de prueba.

QUE HACE
  Lee la tabla `trabajos` (solo SELECT) y, por sesion, suma terminado_en - tomado_en de:
  el primer `procesar` listo, el primer `completar` listo tomado despues, y el primer
  `clasificar` listo tomado despues de ese completar. Divide por video.duracion_s de
  sesion.json, porque la columna de la tabla queda vacia. Marca la sesion como solapada si
  otro trabajo de otra sesion corrio al mismo tiempo que alguna de esas etapas: la GPU es una
  sola y el tiempo no seria el de la sesion sola. Escribe un JSON con la mediana y el detalle.

USO
  python scripts/medir_c5.py --db sqlite:////home/lucasb/boxtwin/datos/boxtwin.db \
      --datos ~/boxtwin/datos --sesiones <id> <id> ... -o c5.json
  (sin --sesiones toma todas; --desde 2026-10-27 filtra por fecha de creacion)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

UMBRAL = 2.0
ETAPAS_C5 = ("procesar", "completar", "clasificar")


# ---------------------------------------------------------------------------
# Partes puras (con pruebas)
# ---------------------------------------------------------------------------


def _fecha(x):
    if x is None or isinstance(x, datetime):
        return x
    return datetime.fromisoformat(str(x))


def elegir_etapas(trabajos: list[dict]) -> dict[str, dict]:
    """
    Los trabajos que cuentan para C5, segun el pre-registro. `trabajos` son dicts con etapa,
    estado, tomado_en y terminado_en. Devuelve etapa -> trabajo; falta la que no hubo.
    """
    listos = sorted((t for t in trabajos if t["estado"] == "listo" and t["tomado_en"]
                     and t["terminado_en"]), key=lambda t: _fecha(t["tomado_en"]))
    elegidos: dict[str, dict] = {}
    desde = None
    for etapa in ETAPAS_C5:
        t = next((t for t in listos if t["etapa"] == etapa
                  and (desde is None or _fecha(t["tomado_en"]) >= desde)), None)
        if t is None:
            break
        elegidos[etapa] = t
        desde = _fecha(t["terminado_en"])
    return elegidos


def segundos(t: dict) -> float:
    return (_fecha(t["terminado_en"]) - _fecha(t["tomado_en"])).total_seconds()


def solapada(elegidos: dict[str, dict], otros: list[dict]) -> bool:
    """Si algun trabajo de otra sesion corrio a la vez que alguna etapa contada."""
    for t in elegidos.values():
        a0, a1 = _fecha(t["tomado_en"]), _fecha(t["terminado_en"])
        for o in otros:
            if not o.get("tomado_en"):
                continue
            b0 = _fecha(o["tomado_en"])
            b1 = _fecha(o["terminado_en"]) or datetime.max.replace(tzinfo=b0.tzinfo)
            if b0 < a1 and a0 < b1:
                return True
    return False


def medir_sesion(trabajos: list[dict], duracion_s: float | None, otros: list[dict]) -> dict:
    el = elegir_etapas(trabajos)
    fila = {"etapas": {e: round(segundos(t), 1) for e, t in el.items()},
            "duracion_s": duracion_s, "solapada": solapada(el, otros)}
    render = [t for t in trabajos if t["etapa"] == "render" and t["estado"] == "listo"
              and t["tomado_en"] and t["terminado_en"]]
    fila["render_s"] = round(segundos(min(render, key=lambda t: _fecha(t["tomado_en"]))), 1) \
        if render else None
    if "completar" not in el:
        fila["llego_a_fightcard"] = False
        return fila
    fila["llego_a_fightcard"] = True
    fila["sin_clasificar"] = "clasificar" not in el
    fila["segundos"] = round(sum(segundos(t) for t in el.values()), 1)
    fila["factor"] = round(fila["segundos"] / duracion_s, 3) if duracion_s else None
    return fila


def resumir(filas: list[dict]) -> dict:
    validas = [f["factor"] for f in filas if f.get("llego_a_fightcard") and f.get("factor")]
    sin_sol = [f["factor"] for f in filas
               if f.get("llego_a_fightcard") and f.get("factor") and not f["solapada"]]
    med = round(statistics.median(validas), 3) if validas else None
    return {
        "sesiones": len(filas),
        "llegaron_a_fightcard": len(validas),
        "mediana_factor": med,
        "maximo_factor": max(validas) if validas else None,
        "mediana_sin_solapadas": round(statistics.median(sin_sol), 3) if sin_sol else None,
        "solapadas": sum(1 for f in filas if f.get("solapada")),
        "umbral": UMBRAL,
        "pasa": med is not None and med <= UMBRAL,
    }


# ---------------------------------------------------------------------------
# Entrada y salida
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", required=True, help="URL de SQLAlchemy, la misma que BOXTWIN_DB")
    ap.add_argument("--datos", default="~/boxtwin/datos", help="BOXTWIN_DATOS")
    ap.add_argument("--sesiones", nargs="*", help="ids de las sesiones de prueba")
    ap.add_argument("--desde", default=None, help="solo sesiones creadas desde esta fecha")
    ap.add_argument("-o", "--salida", default=None)
    args = ap.parse_args()

    from sqlalchemy import create_engine, text

    motor = create_engine(args.db)
    with motor.connect() as c:
        trabajos = [dict(r._mapping) for r in c.execute(text(
            "select sesion_id, etapa, estado, tomado_en, terminado_en from trabajos"))]
        sesiones = [dict(r._mapping) for r in c.execute(text("select id, creada from sesiones"))]
    if args.sesiones:
        ids = list(args.sesiones)
    else:
        ids = [s["id"] for s in sesiones
               if not args.desde or str(_fecha(s["creada"]).date()) >= args.desde]

    datos = Path(args.datos).expanduser()
    filas = []
    for sid in ids:
        propios = [t for t in trabajos if t["sesion_id"] == sid]
        otros = [t for t in trabajos if t["sesion_id"] != sid]
        ruta = datos / "sesiones" / sid / "sesion.json"
        dur = json.loads(ruta.read_text())["video"].get("duracion_s") if ruta.is_file() else None
        fila = {"sesion": sid, **medir_sesion(propios, dur, otros)}
        filas.append(fila)
        print(f"{sid[:8]}  video {dur or 0:7.1f} s  {fila['etapas']}  "
              f"factor {fila.get('factor')}  solapada {fila['solapada']}")
    res = resumir(filas)
    print(f"\nC5: mediana {res['mediana_factor']} sobre {res['llegaron_a_fightcard']} sesiones"
          f" (umbral {UMBRAL}) -> {'PASA' if res['pasa'] else 'NO PASA'};"
          f" sin solapadas {res['mediana_sin_solapadas']}")
    if args.salida:
        Path(args.salida).expanduser().write_text(
            json.dumps({"resumen": res, "sesiones": filas}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
