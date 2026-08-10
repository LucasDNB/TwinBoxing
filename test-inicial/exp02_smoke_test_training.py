"""
Experimento 02: Smoke test del pipeline base sobre video de entrenamiento.

Pipeline: YOLOv8l-pose + BoT-SORT (mismo stack que smoke test de pelea, Fase 0).
Sin lógica de dominio (ROI/Hungarian) — eso es Fase 1.

Objetivo: caracterizar el comportamiento del pipeline en el caso de uso primario
del proyecto (video controlado de entrenamiento), por formato:
- bolsa (1 sujeto, foco frontal/lateral)
- sombra (1 sujeto, sin objetivo físico)
- sparring (2 sujetos en interacción)

Métricas registradas por video:
- FPS promedio sostenido
- IDs únicos generados por el tracker (proxy de estabilidad)
- Detecciones promedio y máximas por frame (proxy de falsos positivos)

Salida: JSON estructurado en docs/experiments/02_artifacts/results.json
"""

from ultralytics import YOLO
import time
import json
from pathlib import Path


# Configuración del pipeline (mismo stack que smoke test de pelea, Fase 0)
PIPELINE_CONFIG = {
    "model": "yolov8l-pose.pt",
    "tracker": str(Path(__file__).parent / "botsort.yaml"),
    "conf": 0.5,
    "device": 0,
}

# Mapeo formato -> archivo de video
VIDEOS_DIR = Path(__file__).parent / "video-samples"
VIDEOS = {
    "bolsa": VIDEOS_DIR / "Bolsa.mp4",
    "sombra": VIDEOS_DIR / "Sombra.mp4",  # TODO: agregar cuando esté disponible
    "sparring": VIDEOS_DIR / "Sparring.mp4",
}


def run_on_video(format_name: str, video_path: Path, model: YOLO) -> dict:
    """Corre el pipeline sobre un video y extrae métricas."""
    print(f"\n{'=' * 60}")
    print(f"Formato: {format_name}")
    print(f"Video:   {video_path.name}")
    print(f"{'=' * 60}")

    start = time.time()
    results = model.track(
        source=str(video_path),
        save=True,
        device=PIPELINE_CONFIG["device"],
        conf=PIPELINE_CONFIG["conf"],
        persist=True,
        tracker=PIPELINE_CONFIG["tracker"],
        verbose=False,
    )
    elapsed = time.time() - start

    # Extraer métricas del tracking
    n_frames = len(results)
    all_ids = []
    detections_per_frame = []

    for r in results:
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().tolist()
            all_ids.extend(ids)
            detections_per_frame.append(len(ids))
        else:
            detections_per_frame.append(0)

    unique_ids = set(all_ids)

    metrics = {
        "formato": format_name,
        "video": video_path.name,
        "frames_procesados": n_frames,
        "tiempo_total_s": round(elapsed, 2),
        "fps_promedio": round(n_frames / elapsed, 1),
        "ms_por_frame": round(elapsed / n_frames * 1000, 1),
        "ids_unicos_generados": len(unique_ids),
        "detecciones_promedio_por_frame": round(sum(detections_per_frame) / n_frames, 2),
        "detecciones_max_en_un_frame": max(detections_per_frame) if detections_per_frame else 0,
    }

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def main():
    model = YOLO(PIPELINE_CONFIG["model"])

    all_results = []
    for format_name, video_path in VIDEOS.items():
        if not video_path.exists():
            print(f"⚠️  Skip {format_name}: no existe {video_path}")
            continue
        metrics = run_on_video(format_name, video_path, model)
        all_results.append(metrics)

    # Guardar resultados estructurados
    output_dir = Path(__file__).parent.parent / "docs" / "experiments" / "02_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"

    output_path.write_text(json.dumps({
        "experimento": "02 - Smoke test pipeline base sobre video de entrenamiento",
        "pipeline": PIPELINE_CONFIG,
        "resultados": all_results,
    }, indent=2, ensure_ascii=False))

    print(f"\n✅ Resultados guardados en: {output_path}")


if __name__ == "__main__":
    main()