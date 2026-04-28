from ultralytics import YOLO
import time

model = YOLO("yolov8l-pose.pt") #Descarga automatica la primera vez

start = time.time()
#results = model.predict(
results = model.track(
    source ="video-samples/amateur_estatico.mp4",
    save=True,
    device=0,
    conf=0.5,
    persist = True, #Agregado con botsort
    tracker = "/home/lucasb/Proyectos/TwinBoxing/test-inicial/botsort.yaml"
    #stream=False se desabilito agregando botsort
    )


elapsed = time.time() - start
n_frames = len(results)
print(f"Frames procesados: {n_frames}")
print(f"Tiempo total:      {elapsed:.2f}s")
print(f"Tiempo por frame:  {elapsed / n_frames * 1000:.1f}ms")
print(f"FPS promedio:      {n_frames / elapsed:.1f}")


