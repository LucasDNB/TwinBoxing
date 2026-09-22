"""
BoxTwin - Entrenamiento del detector de guantes.

POR QUE EXISTE
  Una clase, objeto distintivo, pocas instancias por cuadro: no hace falta nada grande.
  YOLOv8n desde pesos COCO alcanza y entra holgado en los 8 GB de la 2080.

  Dos ajustes de aumentacion que no son el default y tienen motivo. `flipud=0` porque un
  guante de boxeo dado vuelta no existe y ensenar esa orientacion gasta capacidad en algo
  que nunca se va a ver; el dataset publico la trae horneada y no se puede sacar de ahi,
  pero al menos no se agrega mas. Y `imgsz=320` porque las entradas son recortes de persona,
  no cuadros completos: subirlo no agrega detalle que el recorte no tenga.

  LIMITACION CONOCIDA, y es grande: la validacion de este dataset no tiene una sola imagen
  de ring. Los 853 cuadros de video son un solo clip y el reparto por procedencia los manda
  enteros a train. El mAP que este comando imprime mide fotografia de producto. Sirve para
  ver si el entrenamiento converge, no para decir si el detector sirve. Eso se mide contra
  las fuentes anotadas, aparte.

QUE HACE
  Entrena y guarda el checkpoint junto a un sidecar JSON con la procedencia, los ajustes y
  las metricas por epoca, con el mismo par .pt/.json que usa boxtwin-detector.

USO
  boxtwin-guantes train data/recortes --out modelos/
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

__all__ = ["ConfigEntrenamiento", "entrenar"]


@dataclass(frozen=True)
class ConfigEntrenamiento:
    pesos: str = "yolov8n.pt"
    epocas: int = 80
    imgsz: int = 320
    batch: int = 64
    paciencia: int = 20
    semilla: int = 0
    device: str = "0"
    # Un guante dado vuelta no existe. El default de ultralytics ya es 0, pero se fija
    # explicito porque es una decision de dominio y no un default que convenga heredar.
    flipud: float = 0.0
    fliplr: float = 0.5


def entrenar(datos: Path, salida: Path, cfg: ConfigEntrenamiento | None = None) -> dict:
    """Entrena sobre el dataset de recortes y deja `<salida>/guantes.pt` y su sidecar."""
    from ultralytics import YOLO

    cfg = cfg or ConfigEntrenamiento()
    datos, salida = Path(datos), Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    yaml = (datos / "data.yaml").resolve()
    if not yaml.exists():
        raise FileNotFoundError(f"no esta {yaml}; corre primero el subcomando recortes")

    modelo = YOLO(cfg.pesos)
    res = modelo.train(
        data=str(yaml), epochs=cfg.epocas, imgsz=cfg.imgsz, batch=cfg.batch,
        patience=cfg.paciencia, seed=cfg.semilla, device=cfg.device,
        flipud=cfg.flipud, fliplr=cfg.fliplr,
        # Absoluto: con una ruta relativa ultralytics la cuelga de su propio
        # `runs/detect/` y la corrida termina en un lugar distinto del que dice.
        project=str((salida / "_corridas").resolve()), name="guantes", exist_ok=True,
        verbose=False, plots=False,
    )

    mejor = Path(res.save_dir) / "weights" / "best.pt"
    destino = salida / "guantes.pt"
    shutil.copyfile(mejor, destino)

    m = getattr(res, "results_dict", {}) or {}
    entrada = json.loads((datos / "manifiesto.json").read_text()) \
        if (datos / "manifiesto.json").exists() else {}
    sidecar = {
        "kind": "boxtwin.guantes.modelo",
        "creado": datetime.now().astimezone().isoformat(),
        "config": asdict(cfg),
        "datos": str(datos),
        "procedencia_de_los_datos": {
            "kind": entrada.get("kind"),
            "conteos": entrada.get("conteos"),
            "config": entrada.get("config"),
            "entrada": entrada.get("entrada"),
        },
        "metricas_validacion": {k: float(v) for k, v in m.items() if isinstance(v, (int, float))},
        "advertencia": (
            "las metricas de validacion se midieron sobre fotografia de producto: la "
            "validacion de este dataset no tiene una sola imagen de ring, porque los "
            "cuadros de video son un solo clip y el reparto por procedencia los manda "
            "enteros a train. No leer este mAP como rendimiento en ringside"
        ),
        "corrida": str(res.save_dir),
    }
    (salida / "guantes.json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n"
    )
    return sidecar
