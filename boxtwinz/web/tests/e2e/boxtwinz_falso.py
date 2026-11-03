"""
Motor falso para los e2e de BoxTwinZ: imita `boxtwin-annotator procesar` y `completar`.

POR QUE EXISTE
  Los e2e corren la API y el worker reales, sin GPU. Este script escribe en disco lo mismo
  que dejaria el motor: sesion.json con candidatos y recortes, y una Fight-Card 0.3 con
  golpes deterministas a partir de la duracion del video.

USO
  BOXTWINZ_CMD="python -I tests/e2e/boxtwinz_falso.py" (lo arma levantar.sh)
"""

import json
import subprocess
import sys
from pathlib import Path

JPG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432ffc0000b080001000101011100ffc4001f0000010501010101010100000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffda0008010100003f00fbd3ffd9")


def duracion(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 8)


def procesar(args):
    video, out = args[0], Path(args[args.index("--out") + 1])
    (out / "candidatos").mkdir(parents=True, exist_ok=True)
    for t in (3, 7):
        (out / "candidatos" / f"track_{t}.jpg").write_bytes(JPG)
    d = duracion(video)
    (out / "sesion.json").write_text(json.dumps({
        "estado": "espera_siembra",
        "video": {"duracion_s": d, "fps": 30.0, "ancho": 640, "alto": 360},
        "candidatos": [
            {"track": 3, "sugerido": True, "t_desde": 0, "t_hasta": d, "alto_relativo": 1.0, "color_guante": {"nombre": "rojos"}},
            {"track": 7, "sugerido": True, "t_desde": 0, "t_hasta": d * 0.9, "alto_relativo": 0.9, "color_guante": {"nombre": "azules"}},
        ],
    }))


def completar(args):
    out = Path(args[0])
    s = json.loads((out / "sesion.json").read_text())
    d = s["video"]["duracion_s"]
    dos = "--semilla-b" in args

    def golpes(lado, paso, desfase):
        n = int((d - desfase) / paso)
        return [{"id": f"{lado}-{i}", "t_inicio": round(desfase + i * paso, 3), "t_fin": round(desfase + i * paso + 0.25, 3),
                 "brazo": "izq" if i % 3 else "der", "score": 0.9, "descartado": False,
                 "objetivo": "cabeza" if i % 2 else "cuerpo", "tipo": None} for i in range(max(0, n))]

    fc = {"version": "0.3", "video": {"duracion_s": d, "fps": 30.0},
          "identidad": {"cobertura_A": 0.95, "cobertura_B": 0.9 if dos else 0.0, "sin_asignar": 0.1},
          "peleadores": {"A": {"golpes": golpes("A", 0.4, 0.2)}, "B": {"golpes": golpes("B", 1.1, 0.5) if dos else []}}}
    (out / "fightcard.json").write_text(json.dumps(fc))
    s["estado"] = "listo"
    (out / "sesion.json").write_text(json.dumps(s))


if __name__ == "__main__":
    etapa, resto = sys.argv[1], sys.argv[2:]
    {"procesar": procesar, "completar": completar}[etapa](resto)
