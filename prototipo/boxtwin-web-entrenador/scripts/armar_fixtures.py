"""
POR QUE EXISTE
  Las pruebas de integracion (06 §1) corren contra respuestas copiadas de una sesion real y
  no contra datos inventados: asi la web se prueba contra la forma que el worker escribe de
  verdad, con sus nulos y sus rarezas (las guardias con el brazo en ingles, por ejemplo).

QUE HACE
  Lee una sesion procesada de BOXTWIN_DATOS y escribe en tests/fixtures/sesion-real/:
  la Fight-Card 0.1 tal cual, una 0.2 derivada (con tipos de prueba, descartados y tramos,
  marcados como tales), una 0.3 (la 0.2 con lo que agrega la ampliacion del 13-10, calculado
  con el codigo del anotador de ESTE repo y no copiado) y un GET /jobs/{id} por estado. No
  toca la sesion original.

USO
  python -I scripts/armar_fixtures.py ~/boxtwin/datos/sesiones/4ad5553ab817425e8e80bd23c5c931dd
"""
import argparse
import json
import sys
from pathlib import Path

# El anotador de este repo, no el instalado en el entorno (puede ser otro checkout).
ANOTADOR = Path(__file__).resolve().parents[3] / "boxtwin-annotator" / "src"

TIPOS = ["jab", "cross", "hook", "uppercut"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sesion", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "tests/fixtures/sesion-real")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fc = json.loads((a.sesion / "fightcard.json").read_text())
    ses = json.loads((a.sesion / "sesion.json").read_text())
    sid = a.sesion.name

    (a.out / "fightcard-0.1.json").write_text(json.dumps(fc, indent=1, ensure_ascii=False))

    # 0.2: la misma sesion con lo que agrega el contrato nuevo. Los tipos son de PRUEBA (la
    # sesion real se proceso sin clasificador) y lo dice el checkpoint.
    fc2 = json.loads(json.dumps(fc))
    fc2["version"] = "0.2"
    fc2["clasificador"] = {"checkpoint": "prueba-fixture-no-es-un-modelo",
                           "exactitud_familia_fuente_no_vista": None, "estimado": True}
    for p, datos in fc2["peleadores"].items():
        for i, g in enumerate(datos["golpes"]):
            g["tipo"] = TIPOS[i % 4]
            g["confianza_tipo"] = round(0.3 + (i % 7) / 10, 2)
            g["descartado"] = False
        datos["descartados"] = 0
    g0 = fc2["peleadores"]["A"]["golpes"][1]
    g0["descartado"] = True
    g0["corregido"] = {"tipo": "no_golpe", "tipo_original": g0["tipo"], "checkpoint_original": "prueba-fixture-no-es-un-modelo",
                       "cuando": "2026-10-13T10:00:00-03:00", "por": "lucas@usal.edu.ar"}
    fc2["peleadores"]["A"]["descartados"] = 1
    fc2["identidad"].update({"tramos_total": 20, "tramos_asignados_auto": 13,
                             "tramos_asignados_manual": 0, "separacion_color": 1.375})
    (a.out / "fightcard-0.2.json").write_text(json.dumps(fc2, indent=1, ensure_ascii=False))

    # 0.3 (ampliacion del 13-10): las combinaciones izq-der-izq, con la funcion del anotador.
    sys.path.insert(0, str(ANOTADOR))
    from boxtwin.mvp.fightcard import _combinaciones

    fc3 = json.loads(json.dumps(fc2))
    fc3["version"] = "0.3"
    _combinaciones(fc3)
    (a.out / "fightcard-0.3.json").write_text(json.dumps(fc3, indent=1, ensure_ascii=False))

    base = {
        "id": sid, "nombre": "sparring del martes", "estado": "listo", "error": None,
        "video": ses["video"]["nombre"], "duracion_s": ses["video"]["duracion_s"],
        "fps": ses["video"]["fps"],
        "rounds": {"round_s": ses["rounds"]["round_s"], "descanso_s": ses["rounds"]["descanso_s"], "cantidad": 1},
        "siembra": None, "creada": ses["creada"], "actualizada": ses["actualizada"],
        "contexto": "sparring", "practica": None,
        "boxeadores": {"A": None, "B": None},
        "candidatos": [], "avisos": ses.get("avisos", []), "etapas": ses.get("etapas", []),
        "progreso": None, "trabajos": [],
    }
    cands = [{**c, "recorte_url": f"/jobs/{sid}/candidatos/{c['track']}"} for c in ses["candidatos"]]
    estados = {
        "en_cola": {"estado": "en_cola"},
        "procesando": {"estado": "procesando", "progreso": {"etapa": "pose", "hecho": 3358, "total": 5248, "fraccion": 0.64, "segundos": 160.2}},
        "espera_siembra": {"estado": "espera_siembra", "candidatos": cands,
                           "pareja_sugerida": ses["identidad"]["pareja_sugerida"]},
        "completando": {"estado": "completando", "siembra": ses["siembra"]},
        "listo": {"estado": "listo", "siembra": ses["siembra"], "candidatos": cands,
                  "pareja_sugerida": ses["identidad"]["pareja_sugerida"]},
        "fallo": {"estado": "fallo", "error": "la etapa de pose se quedo sin memoria de GPU"},
    }
    for e, extra in estados.items():
        (a.out / f"job-{e}.json").write_text(json.dumps({**base, **extra}, indent=1, ensure_ascii=False))
    print(f"fixtures escritos en {a.out}")


if __name__ == "__main__":
    main()
