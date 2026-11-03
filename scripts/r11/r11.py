"""
BoxTwin - R-11: orquestador del benchmark de perfiles (3-11-2026).

POR QUE EXISTE
  El benchmark comparte la 2080 con A y BoxTwinZ. A no conoce /tmp/boxtwin-gpu.lock, asi que el
  candado solo no alcanza: una celda con un trabajo de A adentro da un numero falso sin avisar
  (docs/specs/2026-11-03-r11-perfiles/00-preregistro.md §4). Este es el unico camino para medir.

QUE HACE
  - cuadros: arma la lista de cuadros por video desde el cache de raiz-v8, con su sha256.
  - correr: etapa 1, etapa 2 (segun regla.celdas_etapa2) y las dos rondas, con el candado tomado.
    Antes de cada celda lee las colas de A y BoxTwinZ: si hay trabajos, suelta el candado, espera y
    lo vuelve a tomar. Antes y despues compara las colas y nvidia-smi; si algo cambio, descarta la
    celda y la repite. Una celda hecha deja su json y no se repite. Despues de --corte no arranca
    ninguna celda.
  - calidad: mAP pose en COCO val2017 de cada celda medida (§5).
  - resumen: estadisticos, concordancia, perfiles y controles en resultado.json.

USO
  PY=~/boxtwin-mejoras/r11/venv/bin/python
  $PY scripts/r11/r11.py cuadros
  $PY scripts/r11/r11.py correr --corte 2026-11-04T04:00
  $PY scripts/r11/r11.py calidad --corte 2026-11-04T04:00
  $PY scripts/r11/r11.py resumen
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import regla  # noqa: E402

BASE = Path(os.environ.get("BOXTWIN_R11", "~/boxtwin-mejoras/r11")).expanduser()
RAIZ_V8 = Path("~/boxtwin-gemelo/raiz-v8").expanduser()
DB_A = Path("~/boxtwin/datos/boxtwin.db").expanduser()
DB_Z = Path("~/boxtwinz-datos/boxtwinz.db").expanduser()
CANDADO = Path("/tmp/boxtwin-gpu.lock")
VIDEOS = {"sparring-3": ("anotacion-sparring-3", "sparring-3-rounds"),
          "pacquiao": ("anotacion-pacquiao", "pacquiao_margarito")}
REFERENCIA = "pytorch-l-640"
ESPERA_S = 30


def log(m: str) -> None:
    linea = f"{datetime.now().isoformat(timespec='seconds')} {m}"
    print(linea, flush=True)
    with (BASE / "r11.log").open("a") as f:
        f.write(linea + "\n")


def _cola(db: Path) -> tuple[int, int]:
    """(trabajos totales, trabajos en_cola o tomado), en solo lectura."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    try:
        return (int(con.execute("select count(*) from trabajos").fetchone()[0]),
                int(con.execute("select count(*) from trabajos where estado in ('en_cola','tomado')").fetchone()[0]))
    finally:
        con.close()


def pids_gpu() -> set[int]:
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=30)
    return {int(x) for x in r.stdout.split() if x.strip().isdigit()}


def foto() -> dict:
    return {"a": _cola(DB_A), "z": _cola(DB_Z), "pids": sorted(pids_gpu())}


def libre(f: dict) -> bool:
    return f["a"][1] == 0 and f["z"][1] == 0 and not f["pids"]


class Candado:
    def __init__(self):
        self.f = None

    def tomar(self):
        self.f = open(CANDADO, "a+")
        fcntl.flock(self.f, fcntl.LOCK_EX)

    def soltar(self):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()
        self.f = None


def esperar_libre(c: Candado) -> None:
    """Con trabajos en alguna cola: suelta el candado, espera y lo vuelve a tomar."""
    f = foto()
    if libre(f):
        return
    log(f"ocupado {f}: suelto el candado y espero")
    c.soltar()
    while not libre(f := foto()):
        time.sleep(ESPERA_S)
    c.tomar()
    log("libre: candado tomado otra vez")


def cmd_cuadros(_a) -> int:
    (BASE / "cuadros").mkdir(parents=True, exist_ok=True)
    for nombre, (dir_, base) in VIDEOS.items():
        raiz = RAIZ_V8 / dir_
        meta = json.loads((raiz / "cache" / f"{base}.meta.json").read_text())
        z = np.load(raiz / "cache" / f"{base}.pose.npz")
        off, conf = z["frame_index"], z["det_conf"]
        ok = np.array([(conf[off[i]:off[i + 1]] >= 0.5).sum() >= 2 for i in range(len(off) - 1)])
        tramos = regla.elegir_tramos(ok, meta["scene_cuts"]["frames"])
        doc = {"video": str(raiz / "videos" / f"{base}.mp4"), "tramos": tramos}
        doc["sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
        (BASE / "cuadros" / f"{nombre}.json").write_text(json.dumps(doc, indent=1))
        log(f"cuadros {nombre}: {len(tramos)} tramos, de {tramos[0][0]} a {tramos[-1][1]}, {ok.mean():.3f} ok")
    return 0


def ruta_modelo(k: str) -> Path:
    formato, t, s = k.split("-")
    pesos = BASE / "pesos" / f"yolov8{t}-pose.pt"
    if formato == "pytorch":
        return pesos
    d = BASE / "modelos" / k
    return d / f"yolov8{t}-pose.{formato}"


def exportar(k: str) -> None:
    """Cada celda en su carpeta: el export a engine escribe un .onnx con el mismo nombre."""
    destino = ruta_modelo(k)
    if destino.is_file() or k.startswith("pytorch"):
        return
    formato, t, s = k.split("-")
    destino.parent.mkdir(parents=True, exist_ok=True)
    pt = destino.parent / f"yolov8{t}-pose.pt"
    shutil.copy2(BASE / "pesos" / pt.name, pt)
    args = f"format='{formato}', imgsz={s}, device=0, half={formato == 'engine'}, simplify=True"
    codigo = f"from ultralytics import YOLO; YOLO({str(pt)!r}).export({args})"
    t0 = time.time()
    subprocess.run([sys.executable, "-c", codigo], check=True, cwd=destino.parent,
                   stdout=(destino.parent / "export.log").open("w"), stderr=subprocess.STDOUT,
                   env={**os.environ, "YOLO_AUTOINSTALL": "false"})
    log(f"exportado {k} en {time.time() - t0:.0f} s, {destino.stat().st_size >> 20} MiB")


def medir(k: str, video: str, ronda: int, c: Candado) -> None:
    salida = BASE / "crudo" / f"{k}.{video}.r{ronda}"
    if salida.with_suffix(".json").is_file():
        return
    formato, t, s = k.split("-")
    intentos = 0
    while True:
        esperar_libre(c)
        antes = foto()
        mon = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu",
             "--format=csv,noheader", "-l", "2"],
            stdout=(BASE / "crudo" / f"{salida.name}.gpu.csv").open("w"), stderr=subprocess.DEVNULL)
        apps = subprocess.Popen(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader", "-l", "5"],
            stdout=(BASE / "crudo" / f"{salida.name}.apps.csv").open("w"), stderr=subprocess.DEVNULL)
        r = subprocess.run(
            [sys.executable, str(AQUI / "medir_celda.py"), "--modelo", str(ruta_modelo(k)), "--imgsz", s,
             "--formato", formato, "--cuadros", str(BASE / "cuadros" / f"{video}.json"), "-o", str(salida) + ".tmp"],
            capture_output=True, text=True)
        mon.terminate()
        apps.terminate()
        mon.wait()
        apps.wait()
        despues = foto()
        if r.returncode != 0:
            log(f"FALLA {k} {video} r{ronda}: {r.stderr[-400:]}")
            (BASE / "crudo" / f"{salida.name}.fallo.txt").write_text(r.stderr[-4000:] + r.stdout[-1000:])
            return
        tmp_j = Path(str(salida) + ".tmp.json")
        tmp_n = Path(str(salida) + ".tmp.npz")
        otros = {x.split(",")[0].strip()
                 for x in (BASE / "crudo" / f"{salida.name}.apps.csv").read_text().splitlines()
                 if "yolo" not in x and "python" not in x and x.strip() and "No running" not in x}
        if antes["a"] != despues["a"] or antes["z"] != despues["z"] or otros:
            intentos += 1
            log(f"CONTAMINADA {k} {video} r{ronda} (intento {intentos}): {antes} -> {despues} {otros}")
            with (BASE / "descartadas.log").open("a") as f:
                f.write(f"{k} {video} r{ronda} {antes} {despues} {sorted(otros)}\n")
            tmp_j.unlink(missing_ok=True)
            tmp_n.unlink(missing_ok=True)
            continue
        tmp_n.rename(salida.with_suffix(".npz"))
        tmp_j.rename(salida.with_suffix(".json"))
        d = np.load(salida.with_suffix(".npz"))
        tot = (d["t_dec"] + d["t_pose"])[regla.WARMUP:]
        log(f"{k} {video} r{ronda}: mediana {np.median(tot):.1f} p95 {np.percentile(tot, 95):.1f}")
        return


def t_total(k: str, video: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    tot, pose, medianas = [], [], []
    for ronda in (1, 2):
        p = BASE / "crudo" / f"{k}.{video}.r{ronda}.npz"
        if p.is_file():
            d = np.load(p)
            x = (d["t_dec"] + d["t_pose"])[regla.WARMUP:]
            tot.append(x)
            pose.append(d["t_pose"][regla.WARMUP:])
            medianas.append(float(np.median(x)))
    if not tot:
        return np.array([]), np.array([]), []
    return np.concatenate(tot), np.concatenate(pose), medianas


def por_video(k: str) -> dict:
    """Solo con las dos rondas de ese video; una celda a medias no cuenta para decidir."""
    salida = {}
    for v in VIDEOS:
        tot, _, med = t_total(k, v)
        if len(med) == 2:
            salida[v] = regla.estadisticos(tot)
    return salida if len(salida) == len(VIDEOS) else {}


def correr_etapa(celdas: list[str], c: Candado, corte: datetime) -> bool:
    for ronda, orden in ((1, celdas), (2, celdas[::-1])):
        for k in orden:
            for v in VIDEOS:
                if datetime.now() >= corte:
                    log(f"corte alcanzado antes de {k} {v} r{ronda}")
                    return False
                esperar_libre(c)
                exportar(k)
                medir(k, v, ronda, c)
    return True


def cmd_correr(a) -> int:
    corte = datetime.fromisoformat(a.corte)
    (BASE / "crudo").mkdir(parents=True, exist_ok=True)
    c = Candado()
    log("esperando el candado")
    c.tomar()
    log(f"candado tomado; {foto()}")
    e1 = [regla.clave(f, t, 640) for f in regla.FORMATOS for t in regla.TAMANOS] + [REFERENCIA]
    if not correr_etapa(e1, c, corte):
        return 0
    etapa1 = {k: por_video(k) for k in e1 if k != REFERENCIA}
    e2 = regla.celdas_etapa2(etapa1)
    (BASE / "etapa2.json").write_text(json.dumps(e2, indent=1))
    log(f"etapa 2: {e2}")
    correr_etapa(e2, c, corte)
    log("correr terminado")
    return 0


def celdas_medidas() -> list[str]:
    ks = sorted({p.name.split(".")[0] for p in (BASE / "crudo").glob("*.r1.json")})
    return [k for k in ks if k != REFERENCIA]


def cmd_calidad(a) -> int:
    corte = datetime.fromisoformat(a.corte)
    yaml_ = BASE / "coco" / "coco-pose-val.yaml"
    c = Candado()
    c.tomar()
    salida = BASE / "calidad"
    salida.mkdir(exist_ok=True)
    ks = sorted(celdas_medidas(), key=lambda k: (not k.startswith("engine"), k))
    for k in ks:
        p = salida / f"{k}.json"
        if p.is_file():
            continue
        if datetime.now() >= corte:
            log(f"corte alcanzado antes de la calidad de {k}")
            break
        esperar_libre(c)
        s = k.split("-")[2]
        codigo = (f"import json,torch;from ultralytics import YOLO;"
                  f"m=YOLO({str(ruta_modelo(k))!r},task='pose').val(data={str(yaml_)!r},imgsz={s},batch=1,"
                  f"device=0,rect=False,workers=2,plots=False,verbose=False,project={str(salida / 'val')!r},name={k!r});"
                  f"json.dump({{'map':float(m.pose.map),'map50':float(m.pose.map50),'map_caja':float(m.box.map)}},open({str(p)!r},'w'))")
        t0 = time.time()
        r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True,
                           env={**os.environ, "YOLO_AUTOINSTALL": "false"})
        log(f"calidad {k}: rc {r.returncode}, {time.time() - t0:.0f} s, {p.read_text() if p.is_file() else r.stderr[-300:]}")
    return 0


def _concordancia(k: str, video: str) -> dict | None:
    p = BASE / "crudo" / f"{k}.{video}.r1.npz"
    if not p.is_file():
        return None
    dir_, base = VIDEOS[video]
    z = np.load(RAIZ_V8 / dir_ / "cache" / f"{base}.pose.npz")
    d = np.load(p)
    off = z["frame_index"]
    ini = np.r_[0, np.cumsum(d["n"])]

    def cuadros():
        for i in range(regla.WARMUP, len(d["cuadro"])):
            f = int(d["cuadro"][i])
            a, b = off[f], off[f + 1]
            x, y = ini[i], ini[i + 1]
            yield (z["bbox"][a:b], z["keypoints"][a:b], z["kp_score"][a:b],
                   d["cajas"][x:y], d["kp"][x:y], d["sc"][x:y])

    return regla.concordancia(cuadros())


def cmd_resumen(_a) -> int:
    ks = sorted({p.name.split(".")[0] for p in (BASE / "crudo").glob("*.json") if ".r" in p.name})
    celdas, detalle = [], {}
    for k in ks:
        formato, t, s = k.split("-")
        e = {"por_video": {}, "pose": {}, "rondas": {}, "extra": {}, "concordancia": {}}
        for v in VIDEOS:
            tot, pose, med = t_total(k, v)
            if not len(tot):
                continue
            e["por_video"][v] = regla.estadisticos(tot)
            e["pose"][v] = regla.estadisticos(pose)
            e["rondas"][v] = {"medianas": med,
                              "marca_10pct": len(med) == 2 and abs(med[0] - med[1]) / min(med) > 0.10}
            js = [json.loads(p.read_text()) for p in sorted((BASE / "crudo").glob(f"{k}.{v}.r?.json"))]
            e["extra"][v] = {
                "fps_sostenidos": [j["fps_sostenidos"] for j in js],
                "fraccion_dos_personas": [j["fraccion_dos_personas"] for j in js],
                "k_r11a": min(j["k_r11a_dos_personas_warmup"] for j in js),
                "proveedor": js[0]["proveedor"], "sha_lista": sorted({j["sha_lista"] for j in js})}
            e["concordancia"][v] = _concordancia(k, v)
        cal = BASE / "calidad" / f"{k}.json"
        e["calidad"] = json.loads(cal.read_text()) if cal.is_file() else None
        detalle[k] = e
        completa = por_video(k)
        k_ok = bool(e["extra"]) and all(x["k_r11a"] >= 0.9 for x in e["extra"].values())
        if k != REFERENCIA and completa and k_ok:
            celdas.append({"clave": k, "formato": formato, "tamano": t, "imgsz": int(s), "por_video": completa,
                           "mediana": max(x["mediana"] for x in completa.values()),
                           "map": e["calidad"]["map"] if e["calidad"] else None})
    doc = {"celdas": detalle, "elegibles": [c["clave"] for c in celdas], "perfiles": regla.asignar(celdas)}
    for p in (BASE / "resultado.json",):
        p.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc["perfiles"], indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cuadros").set_defaults(f=cmd_cuadros)
    for n, f in (("correr", cmd_correr), ("calidad", cmd_calidad)):
        p = sub.add_parser(n)
        p.add_argument("--corte", default="2026-11-04T04:00")
        p.set_defaults(f=f)
    sub.add_parser("resumen").set_defaults(f=cmd_resumen)
    a = ap.parse_args(argv)
    BASE.mkdir(parents=True, exist_ok=True)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
