from ultralytics import YOLO
import time

model = YOLO("yolov8s-pose.pt")  # se descarga automático la primera vez

start = time.time()
results = model.predict(
    source="smoke_test.mp4",
    save=True,
    device=0,
    conf=0.5,
    stream=False,
)
elapsed = time.time() - start
n_frames = len(results)
print(f"Frames procesados: {n_frames}")
print(f"Tiempo total: {elapsed:.1f}s")
print(f"FPS promedio: {n_frames/elapsed:.1f}")