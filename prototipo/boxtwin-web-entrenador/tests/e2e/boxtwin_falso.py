"""
POR QUE EXISTE
  Las pruebas de punta a punta (06 §4) corren la API y el worker de verdad, pero sin GPU.
  El worker llama a `boxtwin-annotator` por subproceso (BOXTWIN_CMD); esto lo reemplaza y
  escribe en cada etapa los archivos de una sesion real, sacados de los fixtures.

QUE HACE
  procesar   sesion.json en espera_siembra, con dos candidatos y sus recortes (imagenes
             lisas generadas aca: no se versionan fotos de personas)
  completar  fightcard.json (la 0.2 de los fixtures, sin tramos sin asignar) y sesion.json
             en listo
  render     copia el video subido como procesado.mp4

USO
  BOXTWIN_CMD="python -I tests/e2e/boxtwin_falso.py" python -m boxtwin_api.worker
"""
import json
import shutil
import sys
import time
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sesion-real"


def _jpg(ruta: Path, bgr: tuple[int, int, int]) -> None:
    import numpy as np
    import cv2

    img = np.zeros((170, 120, 3), np.uint8)
    img[:] = bgr
    ruta.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta), img)


def _sesion(d: Path) -> dict:
    ruta = d / "sesion.json"
    return json.loads(ruta.read_text()) if ruta.is_file() else {}


def _guardar(d: Path, s: dict) -> None:
    (d / "sesion.json").write_text(json.dumps(s, indent=1))


def procesar(args: list[str]) -> None:
    video = Path(args[0])
    d = Path(args[args.index("--out") + 1])
    time.sleep(0.5)
    _jpg(d / "candidatos" / "track_41.jpg", (43, 65, 200))
    _jpg(d / "candidatos" / "track_64.jpg", (200, 116, 47))
    job = json.loads((FIXTURES / "job-espera_siembra.json").read_text())
    _guardar(d, {
        "estado": "espera_siembra",
        "video": {"nombre": video.name, "fps": 30.0, "duracion_s": 174.933},
        "candidatos": [{**c, "recorte": f"candidatos/track_{c['track']}.jpg"} for c in job["candidatos"][:2]],
        "identidad": {"pareja_sugerida": [41, 64]},
        "avisos": [], "etapas": [{"etapa": "preproceso", "segundos": 0.5}],
    })


def completar(args: list[str]) -> None:
    d = Path(args[0])
    time.sleep(0.5)
    fc = json.loads((FIXTURES / "fightcard-0.2.json").read_text())
    fc["identidad"]["sin_asignar"] = 0.0
    (d / "fightcard.json").write_text(json.dumps(fc))
    s = _sesion(d)
    s["estado"] = "listo"
    s.setdefault("identidad", {})["tramos"] = []
    _guardar(d, s)


def render(args: list[str]) -> None:
    d = Path(args[0])
    videos = list((d / "videos").glob("*"))
    if videos:
        shutil.copy(videos[0], d / "procesado.mp4")


if __name__ == "__main__":
    etapa, resto = sys.argv[1], sys.argv[2:]
    {"procesar": procesar, "completar": completar, "render": render}.get(etapa, lambda _: None)(resto)
