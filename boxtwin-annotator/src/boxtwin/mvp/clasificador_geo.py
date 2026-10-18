"""
BoxTwin - Tipo de golpe v2: familia con el modelo geometrico de E4, jab o cross por brazo y guardia.

POR QUE EXISTE
  El tipo de v1 sale de PoseConv3D, que no generaliza a fuentes nuevas (0,656 de familia con
  una fuente afuera, E1) y obliga al worker a tener boxtwin_mmaction. E4, una red chica sobre
  las features que el detector ya calcula, acierta la familia en 0,727 con una fuente afuera
  y corre en este entorno. E6 mostro que el brazo del detector coincide con el anotado en
  0,979: con la guardia del boxeador, jab contra cross sale de una regla y no de un modelo.
  Specs en docs/specs/2026-10-16-fightcard-tipo-v2/ (02-diseno.md, secciones 2 y 3).

  El producto no importa nada de scripts/. Lo que hace falta de alli esta copiado aca, con
  el origen dicho al lado, y CA1 y CA2 verifican que la copia no se desvio.

QUE HACE
  - features_golpe: las 20 del detector (importadas de boxtwin_detector.features) sobre la
    serie entera del peleador, las 6 del rival, ventana [inicio - 8, fin + 8] y remuestreo
    a 32 pasos. Es lo que scripts/clasificador_geo/features2d.py hizo para E4.
  - ClasificadorGeo: carga las semillas del modelo final verificando su sha256 y promedia
    sus probabilidades de familia.
  - rol y tipo_de: adelantado o atrasado con core.types.arm_role, y el tipo de pantalla.
  - clasificar_sesion: de segmentos.json y poses.npz a tipos.json version 0.2.
  - Bolsa y sombra: no hay rival y E4 recibiria sus 6 features en cero todo el golpe, algo
    que casi no vio al entrenar. Para esas sesiones el worker pasa el modelo de E4a (solo las
    20 del brazo, familia 0,724 con una fuente afuera contra 0,727 de E4) y ClasificadorGeo
    recorta las columnas segun el n_features de su meta.json.

USO
  from boxtwin.mvp.clasificador_geo import clasificar_sesion
  clasificar_sesion(Path("sesion/"), Path("~/boxtwin/modelos/clasificador_geo_v1"), "ortodoxa", None)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

__all__ = [
    "FAMILIAS",
    "MODELO_POR_DEFECTO",
    "ClasificadorGeo",
    "clasificar_sesion",
    "features_golpe",
    "features_serie",
    "rol",
    "tipo_de",
]

MODELO_POR_DEFECTO = Path("~/boxtwin/modelos/clasificador_geo_v1")
FAMILIAS = ("straight", "hook", "uppercut")
METODO = "geometrico-v1"
VERSION_TIPOS = "0.2"

# Pre-registro de E4, seccion 2 (scripts/clasificador_geo/geo.py): margen y pasos.
MARGEN = 8
PASOS = 32
# Columnas de features_golpe: 20 del brazo (detector) y 6 del rival. E4 usa las 26, E4a las 20.
N_BRAZO, N_RIVAL = 20, 6

# De scripts/clasificador_geo/features2d.py, sin cambios.
MIN_SCORE = 0.3
HOMBROS, CADERAS = (5, 6), (11, 12)
MUNECA = {"left": 9, "right": 10}

GUARDIAS = {"ortodoxa": "orthodox", "zurda": "southpaw", "orthodox": "orthodox", "southpaw": "southpaw"}
LADOS = {"izq": "left", "der": "right", "left": "left", "right": "right"}
BRAZO_ID = {"left": "izq", "right": "der"}


# ---------------------------------------------------------------------------
# Modelo: copia exacta de scripts/clasificador_geo/modelo.py (E4). CA2 verifica que no se
# desvio. Si alguna vez cambia alla, cambia aca y se vuelve a correr CA2.
# ---------------------------------------------------------------------------

def _clases_modelo():
    import torch
    from torch import nn

    ancho, dilataciones, dropout = 64, (1, 2, 4), 0.2

    class Bloque(nn.Module):
        def __init__(self, ancho: int, dilatacion: int):
            super().__init__()
            self.conv = nn.Conv1d(ancho, ancho, kernel_size=3, dilation=dilatacion, padding=dilatacion)
            self.bn = nn.BatchNorm1d(ancho)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return x + self.drop(torch.relu(self.bn(self.conv(x))))

    class ClasificadorGolpe(nn.Module):
        def __init__(self, n_features: int, n_tipos: int = 3, n_alturas: int = 2):
            super().__init__()
            self.entrada = nn.Conv1d(n_features, ancho, 1)
            self.bloques = nn.Sequential(*[Bloque(ancho, d) for d in dilataciones])
            self.tipo = nn.Linear(2 * ancho, n_tipos)
            self.altura = nn.Linear(2 * ancho, n_alturas)

        def forward(self, x, m):
            mf = m.to(x.dtype)                                   # (B, T)
            h = (x * mf[..., None]).transpose(1, 2)              # (B, F, T)
            h = self.bloques(self.entrada(h))                    # (B, C, T)
            w = mf[:, None, :]
            n = w.sum(-1).clamp(min=1.0)
            prom = (h * w).sum(-1) / n
            maxi = h.masked_fill(w == 0, float("-inf")).amax(-1)
            maxi = torch.where(torch.isfinite(maxi), maxi, torch.zeros_like(maxi))   # golpe sin pasos usables
            z = torch.cat([prom, maxi], -1)
            return self.tipo(z), self.altura(z)

    return ClasificadorGolpe


def ClasificadorGolpe(n_features: int):  # noqa: N802 - se usa como la clase de E4
    return _clases_modelo()(n_features)


# ---------------------------------------------------------------------------
# Features: las de scripts/clasificador_geo/features2d.py. CA1 verifica la igualdad sobre los
# 1047 golpes anotados.
# ---------------------------------------------------------------------------

def _features_rival(kp: np.ndarray, sc: np.ndarray, valid: np.ndarray, p: int, side: str) -> np.ndarray:
    """(T, 6) en pixeles de imagen, sin espejar. Sin rival confiable: rival_ok 0 y el resto en 0."""
    T = kp.shape[1]
    if kp.shape[0] < 2:                     # B8: sesion de un solo peleador
        return np.zeros((T, 6), np.float32)
    r = 1 - p
    kr, sr = kp[r].astype(np.float64), sc[r]
    s = kr[:, list(HOMBROS)].mean(1)
    c = kr[:, list(CADERAS)].mean(1)
    v = s - c
    torso = np.linalg.norm(v, axis=-1)
    ok = (sr[:, list(HOMBROS + CADERAS)].min(1) >= MIN_SCORE) & valid[r].astype(bool) & (torso > 1e-6)
    t = np.where(ok, torso, 1.0)
    m = kp[p][:, MUNECA[side]].astype(np.float64)
    h = (m[:, 1] - s[:, 1]) / t
    dx = np.abs(m[:, 0] - s[:, 0]) / t
    d = np.linalg.norm(m - s, axis=-1) / t
    incl_cos = -v[:, 1] / t
    incl_sin = v[:, 0] / t
    f = np.stack([h, dx, d, incl_cos, incl_sin, ok.astype(np.float64)], axis=-1)
    f[~ok, :5] = 0.0
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _remuestrear(x: np.ndarray, pasos: int = PASOS) -> np.ndarray:
    L = len(x)
    pos = np.linspace(0, L - 1, pasos)
    return np.stack([np.interp(pos, np.arange(L), x[:, j]) for j in range(x.shape[1])], -1).astype(np.float32)


def _remuestrear_mascara(m: np.ndarray, pasos: int = PASOS) -> np.ndarray:
    pos = np.rint(np.linspace(0, len(m) - 1, pasos)).astype(int)
    return np.asarray(m, bool)[pos]


def features_serie(kp: np.ndarray, sc: np.ndarray, valid: np.ndarray, p: int,
                   brazo: str) -> tuple[np.ndarray, np.ndarray]:
    """(T, 26) y la mascara (T,) de un carril, sobre el video entero. Se calcula una vez por carril."""
    from boxtwin_detector.features import features_de

    side = LADOS[brazo]
    f, usable = features_de(kp[p], sc[p], brazo=side)
    fr = _features_rival(kp, sc, valid, p, side)
    return np.concatenate([f.astype(np.float32), fr], -1), usable.astype(bool)


def ventana(inicio: int, fin: int, T: int, margen: int = MARGEN) -> tuple[int, int]:
    """[inicio - margen, fin + margen] recortado al video. Inclusivo, como datos.ventana."""
    return max(0, int(inicio) - margen), min(T - 1, int(fin) + margen)


def features_golpe(kp, sc, valid, p: int, brazo: str, inicio: int, fin: int,
                   serie: tuple[np.ndarray, np.ndarray] | None = None,
                   margen: int = MARGEN) -> tuple[np.ndarray, np.ndarray]:
    """
    X (32, 26) y M (32,) de un golpe. `inicio` y `fin` son los cuadros del golpe (el margen lo
    agrega esta funcion); `serie` evita recalcular el carril para cada golpe.
    """
    f, us = serie if serie is not None else features_serie(kp, sc, valid, p, brazo)
    a, b = ventana(inicio, fin, len(f), margen)
    return _remuestrear(f[a:b + 1]), _remuestrear_mascara(us[a:b + 1])


# ---------------------------------------------------------------------------
# Ensamble
# ---------------------------------------------------------------------------

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


class ClasificadorGeo:
    """Las semillas del modelo final, cada una con su estandarizacion, en CPU."""

    def __init__(self, modelos: list, medias: list, desvios: list, meta: dict):
        self.modelos, self.medias, self.desvios, self.meta = modelos, medias, desvios, meta

    @classmethod
    def cargar(cls, directorio: Path) -> "ClasificadorGeo":
        import torch

        directorio = Path(directorio).expanduser()
        meta = json.loads((directorio / "meta.json").read_text())
        semillas = meta.get("semillas") or {}
        if not semillas:
            raise ValueError(f"{directorio}/meta.json no declara semillas")
        modelos, medias, desvios = [], [], []
        for nombre, info in sorted(semillas.items()):
            pt = directorio / info["archivo"]
            if _sha256(pt) != info["sha256"]:
                # Un .pt que no es el que se midio no se usa: el numero que muestra la
                # Fight-Card dejaria de ser el de este modelo.
                raise ValueError(f"sha256 de {pt.name} no coincide con meta.json")
            m = ClasificadorGolpe(int(meta["n_features"]))
            m.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
            m.eval()
            modelos.append(m)
            medias.append(np.asarray(info["media"], np.float32))
            desvios.append(np.asarray(info["desvio"], np.float32))
        return cls(modelos, medias, desvios, meta)

    @property
    def n_features(self) -> int:
        # Sin meta (como en las pruebas que arman el ensamble a mano), lo dice la estandarizacion.
        return int(self.meta.get("n_features") or len(self.medias[0]))

    @property
    def sin_rival(self) -> bool:
        """El modelo de E4a (solo las 20 del brazo): el de bolsa y sombra, donde no hay rival."""
        return self.n_features == N_BRAZO

    def familia(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        """(n, 3): promedio de las probabilidades de familia de las semillas."""
        import torch

        X = np.asarray(X, np.float32)
        if X.ndim == 2:
            X, M = X[None], np.asarray(M)[None]
        # features_golpe da siempre las 26 de E4; el modelo de E4a usa las 20 primeras, que
        # son las del brazo (scripts/clasificador_geo/entrenar.py, EXPERIMENTOS["E4a"]).
        if X.shape[-1] != self.n_features:
            if self.n_features != N_BRAZO or X.shape[-1] != N_BRAZO + N_RIVAL:
                raise ValueError(f"el modelo espera {self.n_features} features y llegan {X.shape[-1]}")
            X = X[..., :N_BRAZO]
        m = torch.tensor(np.asarray(M, bool), dtype=torch.float32)
        ps = []
        with torch.no_grad():
            for modelo, media, desvio in zip(self.modelos, self.medias, self.desvios):
                x = torch.tensor((X - media) / desvio, dtype=torch.float32)
                lt, _ = modelo(x, m)
                ps.append(torch.softmax(lt, -1).numpy())
        return np.mean(ps, axis=0)


def rol(guardia: str, brazo: str) -> str:
    """'lead' o 'rear' via core.types.arm_role. Acepta ortodoxa/zurda e izq/der."""
    from boxtwin.core.types import Guard, Side, arm_role

    return arm_role(Guard(GUARDIAS[guardia]), Side(LADOS[brazo])).value


def tipo_de(familia: str, rol_: str) -> str:
    """El tipo de pantalla: un recto adelantado es jab, uno atrasado es cross (RT4)."""
    if familia == "straight":
        return "jab" if rol_ == "lead" else "cross"
    return familia


# ---------------------------------------------------------------------------
# La sesion
# ---------------------------------------------------------------------------

def _guardia_efectiva(g: str | None) -> tuple[str, bool]:
    """(guardia, supuesta). Sin guardia cargada se supone ortodoxa y se dice (RT5)."""
    if g in ("ortodoxa", "zurda"):
        return g, False
    if g in ("orthodox", "southpaw"):
        return {"orthodox": "ortodoxa", "southpaw": "zurda"}[g], False
    return "ortodoxa", True


def clasificar_sesion(directorio: Path, modelo: Path = MODELO_POR_DEFECTO, guardia_a: str | None = None,
                      guardia_b: str | None = None, clasificador: ClasificadorGeo | None = None) -> dict:
    """
    Lee segmentos.json y poses.npz, clasifica cada segmento y escribe tipos.json 0.2.

    El id del golpe es el de tools/clasificar_sesion.py y fightcard.id_de_golpe, para que
    aplicar_tipos lo encuentre igual que con v1.
    """
    directorio = Path(directorio)
    d = json.loads((directorio / "segmentos.json").read_text())
    z = np.load(directorio / d.get("poses", "poses.npz"))
    kp, sc = z["keypoints"], z["kp_score"]
    valid = z["valid"] if "valid" in z.files else np.ones(kp.shape[:2], bool)
    clf = clasificador or ClasificadorGeo.cargar(modelo)

    guardia, supuesta = {}, {}
    for lado, g in (("A", guardia_a), ("B", guardia_b)):
        guardia[lado], supuesta[lado] = _guardia_efectiva(g)

    indices = {"A": 0, "B": 1}
    series: dict = {}
    claves, Xs, Ms = [], [], []
    saltados = 0
    for s in d["segmentos"]:
        p = indices.get(s["peleador"])
        if p is None or p >= kp.shape[0]:
            continue
        brazo = LADOS[s["brazo"]]
        if (p, brazo) not in series:
            series[(p, brazo)] = features_serie(kp, sc, valid, p, brazo)
        X, M = features_golpe(kp, sc, valid, p, brazo, s["inicio"], s["fin"], serie=series[(p, brazo)])
        # Un segmento sin ningun paso usable no se clasifica, como los de un cuadro en v1:
        # sale sin tipo, que la Fight-Card sabe mostrar.
        if not M.any():
            saltados += 1
            continue
        claves.append((s, brazo))
        Xs.append(X)
        Ms.append(M)

    tipos: dict[str, dict] = {}
    if Xs:
        P = clf.familia(np.stack(Xs), np.stack(Ms))
        for (s, brazo), pf in zip(claves, P):
            i = int(np.argmax(pf))
            fam = FAMILIAS[i]
            r = rol(guardia[s["peleador"]], brazo)
            gid = f"{s['peleador']}-{BRAZO_ID[brazo]}-{s['inicio']}"
            tipos[gid] = {"tipo": tipo_de(fam, r), "confianza": round(float(pf[i]), 4), "crudo": fam,
                          "p_familia": [round(float(x), 4) for x in pf], "rol": r}

    met = clf.meta.get("metricas_e4") or {}
    salida = {
        "version": VERSION_TIPOS,
        "metodo": METODO,
        "checkpoint": clf.meta.get("nombre", Path(modelo).name),
        "exactitud_familia_fuente_no_vista": met.get("familia_fuente_no_vista"),
        "referencia": met.get("referencia"),
        "clases": list(FAMILIAS),
        # "con_rival" (E4, sparring y combate) o "sin_rival" (E4a, bolsa y sombra)
        "entrada": "sin_rival" if clf.sin_rival else "con_rival",
        "guardia": guardia,
        "guardia_supuesta": supuesta,
        "segmentos_sin_clasificar": saltados,
        "tipos": tipos,
    }
    (directorio / "tipos.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False) + "\n")
    return salida
