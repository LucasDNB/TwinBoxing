"""
BoxTwin - Modo academico E7: 3D de HMR2.0 (4D-Humans) para cada golpe detectado de una sesion.

POR QUE EXISTE
  E7 mostro que, con el mismo modelo chico, el 3D de HMR2 mejora la familia sobre MotionBERT
  (0,777 contra 0,738, docs/experiments/2026-10-24-e7-4dhumans.md). La enmienda del 25-10
  (docs/specs/2026-10-25-e7-modo-academico/00-enmienda.md) lo deja entrar a la app como modo
  academico, apagado por defecto. HMR2 necesita su propio entorno (boxtwin_4dhumans) y SMPL, que
  solo admite investigacion: por eso corre como herramienta aparte, igual que PoseConv3D en v1
  corria en boxtwin_mmaction, y deja un archivo que `boxtwin clasificar --pose3d` lee.

  El producto no importa nada de scripts/. Lo que hace falta de scripts/clasificador_geo/
  (elevar_hmr2.py y elevar3d.py) esta copiado aca con el origen al lado, y CA-E7a
  (scripts/clasificador_geo/tests/test_equivalencia_e7.py) verifica que la copia no se desvio.

QUE HACE
  Lee segmentos.json (ruta y fps del video, segmentos) y poses.npz (keypoints con la identidad
  ya resuelta). Por segmento: ventana de E4 (inicio - 8 a fin + 8), cuadros de esa ventana con
  el paso de E5, caja del peleador desde sus keypoints con huecos rellenados, HMR2 por cuadro,
  las 24 articulaciones de SMPL llevadas a H36M y remuestreo a 32 pasos con mascara. Escribe
  <sesion>/pose3d_hmr2.npz (gid, P3 (n, 32, 17, 3), M3 (n, 32)) y pose3d_hmr2.meta.json. El gid
  es el id del golpe de la Fight-Card (A-izq-20).

USO (en boxtwin_4dhumans, con el modelo SMPL en ~/.cache/4DHumans/data/smpl/)
  ~/miniforge3/envs/boxtwin_4dhumans/bin/python boxtwin-annotator/tools/elevar_hmr2_sesion.py <sesion>
  ~/miniforge3/envs/boxtwin_4dhumans/bin/python boxtwin-annotator/tools/elevar_hmr2_sesion.py <sesion> --lote 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

# De scripts/clasificador_geo (pre-registro de E7, E2 y E3; elevar3d.py para E5 y E4).
SCORE_MIN = 0.3
MAX_HUECO = 10
LARGO = 243
PASOS = 32
MARGEN = 8
LOTE = 8
SMPL_A_H36M = {0: 0, 1: 2, 4: 1, 8: 12, 11: 16, 12: 18, 13: 20, 14: 17, 15: 19, 16: 21}
BRAZO_ID = {"left": "izq", "right": "der", "izq": "izq", "der": "der"}
INDICE = {"A": 0, "B": 1}
METODO = "hmr2.0b-4dhumans"


# ---------------------------------------------------------------------------
# Copias. CA-E7a verifica cada una contra su original.
# ---------------------------------------------------------------------------

def ventana(inicio: int, fin: int, T: int, margen: int = MARGEN) -> tuple[int, int]:
    """boxtwin.mvp.clasificador_geo.ventana (E4): [inicio - 8, fin + 8] recortada al video."""
    return max(0, inicio - margen), min(T - 1, fin + margen)


def rellenar_huecos(kp: np.ndarray, presente: np.ndarray, max_hueco: int = MAX_HUECO) -> tuple[np.ndarray, np.ndarray]:
    """elevar3d.rellenar_huecos: huecos internos cortos interpolados con mascara 1; el resto, vecino y mascara 0."""
    kp = np.array(kp, np.float32, copy=True)
    presente = np.asarray(presente, bool)
    mascara = presente.copy()
    T = len(kp)
    if not presente.any():
        return np.zeros_like(kp), mascara
    i = 0
    while i < T:
        if presente[i]:
            i += 1
            continue
        j = i
        while j < T and not presente[j]:
            j += 1
        izq, der = i - 1, j
        if izq >= 0 and der < T and (j - i) <= max_hueco:
            a = (np.arange(i, j) - izq) / (der - izq)
            kp[i:j] = kp[izq][None] * (1 - a)[:, None, None] + kp[der][None] * a[:, None, None]
            mascara[i:j] = True
        else:
            for f in range(i, j):
                if izq < 0:
                    kp[f] = kp[der]
                elif der >= T:
                    kp[f] = kp[izq]
                else:
                    kp[f] = kp[izq] if (f - izq) <= (der - f) else kp[der]
        i = j
    return kp, mascara


def paso_de(fps: float) -> int:
    """elevar3d.paso_de."""
    return 2 if fps > 45 else 1


def indices_contexto(ini: int, fin: int, T: int, paso: int, largo: int = LARGO) -> tuple[np.ndarray, np.ndarray]:
    """elevar3d.indices_contexto."""
    centro = (ini + fin) // 2
    crudo = centro + (np.arange(largo) - largo // 2) * paso
    if crudo[0] > ini or crudo[-1] < fin:
        raise ValueError(f"la ventana [{ini}, {fin}] no entra en el contexto de {largo} cuadros con paso {paso}")
    return crudo, np.clip(crudo, 0, T - 1)


def a_ventana(X: np.ndarray, crudo: np.ndarray, ini: int, fin: int, pasos: int = PASOS) -> np.ndarray:
    """elevar3d.a_ventana."""
    pos = np.linspace(ini, fin, pasos)
    flat = X.reshape(len(X), -1)
    out = np.stack([np.interp(pos, crudo, flat[:, j]) for j in range(flat.shape[1])], -1)
    return out.reshape(pasos, *X.shape[1:]).astype(np.float32)


def mascara_ventana(mascara: np.ndarray, ini: int, fin: int, pasos: int = PASOS) -> np.ndarray:
    """elevar3d.mascara_ventana."""
    pos = np.rint(np.linspace(ini, fin, pasos)).astype(int)
    return mascara[pos]


def smpl_a_h36m(J: np.ndarray) -> np.ndarray:
    """elevar_hmr2.smpl_a_h36m."""
    Q = np.zeros(J.shape[:-2] + (17, 3), np.float32)
    for h, s in SMPL_A_H36M.items():
        Q[..., h, :] = J[..., s, :]
    return Q


def caja(kp: np.ndarray, sc: np.ndarray, score_min: float = SCORE_MIN) -> np.ndarray | None:
    """elevar_hmr2.caja."""
    ok = sc >= score_min
    if not ok.any():
        return None
    p = kp[ok]
    return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()], np.float32)


def cajas_de(kp: np.ndarray, sc: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """elevar_hmr2.cajas_de."""
    T = len(kp)
    crudas = np.zeros((T, 2, 2), np.float32)
    presente = np.zeros(T, bool)
    for t in np.flatnonzero(valid):
        b = caja(kp[t], sc[t])
        if b is not None:
            crudas[t], presente[t] = b.reshape(2, 2), True
    rellenas, mascara = rellenar_huecos(crudas, presente)
    return rellenas.reshape(T, 4), mascara


def cuadros_ventana(ini: int, fin: int, T: int, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """elevar_hmr2.cuadros_ventana."""
    crudo, _ = indices_contexto(ini, fin, T, paso_de(fps))
    a = np.flatnonzero(crudo <= ini)[-1]
    b = np.flatnonzero(crudo >= fin)[0]
    sel = crudo[a:b + 1]
    return sel, np.clip(sel, 0, T - 1)


# ---------------------------------------------------------------------------
# Sesion
# ---------------------------------------------------------------------------

def gid(s: dict) -> str:
    """El id del golpe de fightcard.id_de_golpe y de clasificar_sesion."""
    return f"{s['peleador']}-{BRAZO_ID[s['brazo']]}-{s['inicio']}"


class Lector:
    """Cuadros por indice, leyendo hacia adelante y saltando solo si hace falta (C5, 20-10)."""

    def __init__(self, ruta: Path):
        import cv2
        self.cv2, self.cap, self.pos, self.ultimo = cv2, cv2.VideoCapture(str(ruta)), -1, None
        if not self.cap.isOpened():
            raise ValueError(f"no se pudo abrir el video {ruta}")

    def cuadro(self, i: int):
        if i == self.pos:
            return self.ultimo
        if i < self.pos or i > self.pos + 30:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, i)
            self.pos = i - 1
        while self.pos < i:
            ok, self.ultimo = self.cap.read()
            self.pos += 1
            if not ok:
                raise RuntimeError(f"no se pudo leer el cuadro {i}")
        return self.ultimo


class ElevadorHMR2:
    """HMR2.0b por cuadro y las 24 articulaciones de SMPL (elevar_hmr2.Elevador)."""

    def __init__(self, lote: int = LOTE):
        import smplx
        import torch
        from hmr2.datasets.vitdet_dataset import ViTDetDataset
        from hmr2.models import DEFAULT_CHECKPOINT, load_hmr2
        from hmr2.utils import recursive_to
        self.torch, self.smplx, self.Dataset, self.recursive_to = torch, smplx, ViTDetDataset, recursive_to
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo, self.cfg = load_hmr2(DEFAULT_CHECKPOINT)
        self.modelo = self.modelo.to(self.dev).eval()
        self.checkpoint, self.lote = Path(DEFAULT_CHECKPOINT), lote

    def j24(self, imagenes: list, cajas: list) -> np.ndarray:
        torch = self.torch
        items = [self.Dataset(self.cfg, img, b[None])[0] for img, b in zip(imagenes, cajas)]
        salida = []
        for k in range(0, len(items), self.lote):
            batch = self.recursive_to(torch.utils.data.default_collate(items[k:k + self.lote]), self.dev)
            with torch.no_grad():
                prm = self.modelo(batch)["pred_smpl_params"]
                n = prm["body_pose"].shape[0]
                crudo = self.smplx.SMPL.forward(self.modelo.smpl,
                                                global_orient=prm["global_orient"].reshape(n, -1, 3, 3),
                                                body_pose=prm["body_pose"].reshape(n, -1, 3, 3),
                                                betas=prm["betas"].reshape(n, -1), pose2rot=False)
            salida.append(crudo.joints[:, :24].float().cpu().numpy())
        return np.concatenate(salida) if salida else np.zeros((0, 24, 3), np.float32)


def elevar_sesion(directorio: Path, elevador, lector=None) -> dict:
    """Escribe pose3d_hmr2.npz y su meta en la sesion. Devuelve el resumen."""
    directorio = Path(directorio)
    t0 = time.perf_counter()
    d = json.loads((directorio / "segmentos.json").read_text())
    video = d.get("video") or {}
    if lector is None:
        if not video.get("ruta"):
            raise ValueError("segmentos.json no tiene la ruta del video (video.ruta)")
        lector = Lector(Path(video["ruta"]))
    fps = float(video.get("fps") or 30.0)
    z = np.load(directorio / d.get("poses", "poses.npz"))
    kp, sc = z["keypoints"], z["kp_score"]
    valid = z["valid"] if "valid" in z.files else np.ones(kp.shape[:2], bool)
    T = kp.shape[1]
    cajas = {}
    gids, P3, M3 = [], [], []
    orden = sorted(d["segmentos"], key=lambda s: s["inicio"])          # el video se lee hacia adelante
    for s in orden:
        p = INDICE.get(s["peleador"])
        if p is None or p >= kp.shape[0]:
            continue
        if p not in cajas:
            cajas[p] = cajas_de(kp[p], sc[p], valid[p])
        bx, mascara = cajas[p]
        ini, fin = ventana(int(s["inicio"]), int(s["fin"]), T)
        sel, idx = cuadros_ventana(ini, fin, T, fps)
        ok = mascara[idx]
        X = np.zeros((len(sel), 17, 3), np.float32)
        if ok.any():
            usar = np.flatnonzero(ok)
            J = elevador.j24([lector.cuadro(int(idx[i])) for i in usar], [bx[idx[i]] for i in usar])
            X[usar] = smpl_a_h36m(J)
            for i in np.flatnonzero(~ok):
                X[i] = X[usar[np.argmin(np.abs(usar - i))]]
        gids.append(gid(s))
        P3.append(a_ventana(X, sel, ini, fin))
        M3.append(mascara_ventana(mascara, ini, fin) & ok.any())
    # en el orden de segmentos.json, como los lee clasificar_sesion
    pos = {g: i for i, g in enumerate(gids)}
    final = [gid(s) for s in d["segmentos"] if gid(s) in pos]
    P3 = np.stack([P3[pos[g]] for g in final]) if final else np.zeros((0, PASOS, 17, 3), np.float32)
    M3 = np.stack([M3[pos[g]] for g in final]) if final else np.zeros((0, PASOS), bool)
    np.savez_compressed(directorio / "pose3d_hmr2.npz", gid=np.array(final), P3=P3, M3=M3)
    sin_3d = int((~M3.any(1)).sum()) if len(final) else 0
    resumen = {"metodo": METODO, "segmentos": len(final), "sin_3d": sin_3d,
               "pasos_con_3d": round(float(M3.mean()), 4) if len(final) else None,
               "segundos": round(time.perf_counter() - t0, 2)}
    ck = getattr(elevador, "checkpoint", None)
    if ck is not None and Path(ck).is_file():
        h = hashlib.sha256()
        with Path(ck).open("rb") as fh:
            for bloque in iter(lambda: fh.read(1 << 20), b""):
                h.update(bloque)
        resumen["checkpoint"], resumen["checkpoint_sha256"] = str(ck), h.hexdigest()
    (directorio / "pose3d_hmr2.meta.json").write_text(json.dumps(resumen, indent=2) + "\n")
    return resumen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sesion", type=Path)
    ap.add_argument("--lote", type=int, default=LOTE)
    args = ap.parse_args(argv)
    r = elevar_sesion(args.sesion, ElevadorHMR2(args.lote))
    print(f"{r['segmentos']} segmentos elevados con HMR2, {r['sin_3d']} sin 3D, {r['segundos']} s "
          f"(modo academico: SMPL, licencia de investigacion) -> {args.sesion / 'pose3d_hmr2.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
