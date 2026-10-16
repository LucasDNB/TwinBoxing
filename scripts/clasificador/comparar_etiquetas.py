"""
BoxTwin - Acuerdo entre las etiquetas propuestas a ciegas y la anotacion de Lucas.

POR QUE EXISTE
  Antes de usar a Claude como preetiquetador hay que saber cuanto le acierta a la anotacion
  humana, con un umbral fijado antes de mirar. Este script es la medicion: lee la clave del
  lote y las respuestas del subagente etiquetador y decide contra los umbrales del
  pre-registro. No sabe nada del etiquetador; sirve igual para comparar dos personas.

  Lectura estricta, fijada en el spec antes de correr: un id sin respuesta, una respuesta
  "no_se" o "no_golpe" cuentan como error. La cobertura (cuantas respondio) se reporta al
  lado, sin umbral. Si un id tiene dos respuestas vale la primera y se avisa.

QUE HACE
  Exactitud por familia (straight, hook, uppercut), exactitud balanceada de familia, kappa,
  exactitud balanceada de altura y recall de cuerpo con Wilson, todo en total y por fuente,
  con las matrices de confusion. Escribe un JSON y dice si pasa.

USO
  python scripts/clasificador/comparar_etiquetas.py --lote piloto \
      --familia-min 0.85 --altura-min 0.80 --salida ~/boxtwin/clasificador-2026-10/lotes/piloto/resultado.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import ALTURAS, TIPOS, TRABAJO, balanceada, confusion, exactitud, kappa, recall_por_clase, redondear, wilson  # noqa: E402

TIPOS_VALIDOS = set(TIPOS) | {"no_golpe", "no_se"}
ALTURAS_VALIDAS = set(ALTURAS) | {"no_se"}
SIN_RESPUESTA = "sin_respuesta"


def leer_respuestas(directorio: Path) -> tuple[dict, list[str]]:
    resp, avisos = {}, []
    for archivo in sorted(directorio.glob("*.jsonl")):
        for n, linea in enumerate(archivo.read_text().splitlines(), 1):
            if not linea.strip():
                continue
            try:
                r = json.loads(linea)
            except json.JSONDecodeError:
                avisos.append(f"{archivo.name}:{n} no es JSON")
                continue
            if r.get("id") in resp:
                avisos.append(f"{r['id']} respondido dos veces; vale la primera")
                continue
            tipo, altura = r.get("tipo"), r.get("altura")
            if tipo not in TIPOS_VALIDOS or altura not in ALTURAS_VALIDAS:
                avisos.append(f"{archivo.name}:{n} valor fuera de vocabulario: {tipo!r}, {altura!r}")
                tipo = tipo if tipo in TIPOS_VALIDOS else "no_se"
                altura = altura if altura in ALTURAS_VALIDAS else "no_se"
            resp[r["id"]] = {"tipo": tipo, "altura": altura, "confianza": r.get("confianza")}
    return resp, avisos


def medir(filas: list[dict], resp: dict) -> dict:
    real_t = [f["punch_type"] for f in filas]
    real_a = [f["target"] for f in filas]
    pred_t = [resp.get(f["id"], {}).get("tipo", SIN_RESPUESTA) for f in filas]
    pred_a = [resp.get(f["id"], {}).get("altura", SIN_RESPUESTA) for f in filas]
    n = len(filas)
    ac_t = sum(1 for r, p in zip(real_t, pred_t) if r == p)
    rec_a = recall_por_clase(real_a, pred_a, ALTURAS)
    return {
        "golpes": n,
        "cobertura_tipo": sum(1 for p in pred_t if p in TIPOS) / n if n else None,
        "cobertura_altura": sum(1 for p in pred_a if p in ALTURAS) / n if n else None,
        "familia": {
            "exactitud": exactitud(real_t, pred_t),
            "wilson95": wilson(ac_t, n),
            "balanceada": balanceada(real_t, pred_t, TIPOS),
            "kappa": kappa(real_t, pred_t),
            "recall": recall_por_clase(real_t, pred_t, TIPOS),
            "confusion": confusion(real_t, pred_t),
        },
        "altura": {
            "balanceada": balanceada(real_a, pred_a, ALTURAS),
            "recall_cabeza": rec_a["head"]["recall"],
            "recall_cuerpo": rec_a["body"]["recall"],
            "recall_cuerpo_wilson95": rec_a["body"]["wilson95"],
            "kappa": kappa(real_a, pred_a),
            "confusion": confusion(real_a, pred_a),
        },
    }


def comparar(clave: Path, respuestas: Path, familia_min: float, altura_min: float) -> dict:
    filas = [f for f in csv.DictReader(clave.open())]
    if any(not f["punch_type"] for f in filas):
        raise SystemExit("la clave no tiene etiqueta real: este lote es de material nuevo, no se puede medir")
    resp, avisos = leer_respuestas(respuestas)
    sobrantes = sorted(set(resp) - {f["id"] for f in filas})
    if sobrantes:
        avisos.append(f"{len(sobrantes)} respuestas con id que no esta en el lote: {sobrantes[:5]}")
    total = medir(filas, resp)
    fuentes = sorted({f["fuente"] for f in filas})
    pasa_f = total["familia"]["exactitud"] is not None and total["familia"]["exactitud"] >= familia_min
    pasa_a = total["altura"]["balanceada"] is not None and total["altura"]["balanceada"] >= altura_min
    return redondear({
        "lectura": "estricta: sin respuesta, no_se y no_golpe cuentan como error",
        "umbrales": {"familia_exactitud_min": familia_min, "altura_balanceada_min": altura_min},
        "total": total,
        "por_fuente": {fu: medir([f for f in filas if f["fuente"] == fu], resp) for fu in fuentes},
        "pasa_familia": pasa_f,
        "pasa_altura": pasa_a,
        "pasa": pasa_f and pasa_a,
        "avisos": avisos,
    })


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lote", required=True)
    ap.add_argument("--familia-min", type=float, required=True)
    ap.add_argument("--altura-min", type=float, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    args = ap.parse_args(argv)
    lote = TRABAJO / "lotes" / args.lote
    r = comparar(lote / "clave" / "clave.csv", lote / "respuestas", args.familia_min, args.altura_min)
    args.salida.expanduser().write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n")
    t = r["total"]
    print(f"golpes {t['golpes']}  cobertura tipo {t['cobertura_tipo']}  altura {t['cobertura_altura']}")
    print(f"familia: exactitud {t['familia']['exactitud']} (Wilson {t['familia']['wilson95']}), "
          f"balanceada {t['familia']['balanceada']}, kappa {t['familia']['kappa']}")
    print(f"altura: balanceada {t['altura']['balanceada']}, recall cuerpo {t['altura']['recall_cuerpo']} "
          f"(Wilson {t['altura']['recall_cuerpo_wilson95']})")
    print("PASA" if r["pasa"] else "NO PASA", "->", args.salida)
    for a in r["avisos"]:
        print("  aviso:", a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
