"""
El camino de produccion: de keypoints identificados a golpes con su segundo.

Lo que se fija aca es lo que separa este camino del de medicion y por lo tanto lo que puede
divergir en silencio: el remuestreo al timebase del modelo, la mascara de identidad sin
resolver, y la vuelta del cuadro del modelo al cuadro del video.
"""

from __future__ import annotations

import numpy as np
import pytest

from boxtwin_detector.features import N_FEATURES
from boxtwin_detector.inferencia import Carriles, carriles_de, detectar


def _esqueleto(T: int, fps: float = 30.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dos cuerpos quietos y validos, con hombros separados para que el torso mida."""
    kp = np.zeros((2, T, 17, 2), np.float32)
    sc = np.ones((2, T, 17), np.float32)
    for p in range(2):
        base = 100.0 + p * 400.0
        kp[p, :, 5] = (base - 30, 100)     # hombro izquierdo
        kp[p, :, 6] = (base + 30, 100)     # hombro derecho
        kp[p, :, 7] = (base - 50, 150)     # codo izq
        kp[p, :, 8] = (base + 50, 150)
        kp[p, :, 9] = (base - 60, 190)     # muneca izq
        kp[p, :, 10] = (base + 60, 190)
        kp[p, :, 11] = (base - 25, 220)    # caderas
        kp[p, :, 12] = (base + 25, 220)
        kp[p, :, 0] = (base, 60)           # nariz
    valid = np.ones((2, T), bool)
    return kp, sc, valid


# -- remuestreo -------------------------------------------------------------


def test_un_video_a_60_fps_se_lleva_al_timebase_del_modelo():
    # El campo receptivo del modelo esta en cuadros. A 60 fps un golpe ocupa el doble y el
    # modelo lo ve como otra accion: el remuestreo no es cosmetico.
    kp, sc, valid = _esqueleto(120)
    c = carriles_de(kp, sc, valid, fps_origen=60.0)
    assert c.T == 60
    assert c.fps == 30.0
    assert c.features.shape == (4, 60, N_FEATURES)


def test_un_video_que_ya_esta_a_30_no_se_toca():
    kp, sc, valid = _esqueleto(90)
    c = carriles_de(kp, sc, valid, fps_origen=30.0)
    assert c.T == 90
    assert np.array_equal(c.indice, np.arange(90))


def test_el_indice_vuelve_al_cuadro_del_video():
    kp, sc, valid = _esqueleto(120)
    c = carriles_de(kp, sc, valid, fps_origen=60.0)
    # El cuadro 10 del modelo es el 20 del video: la mitad del tiempo, el doble de cuadros.
    assert c.indice[10] == 20
    assert c.indice[-1] < 120


# -- carriles ---------------------------------------------------------------


def test_son_cuatro_carriles_y_en_orden():
    kp, sc, valid = _esqueleto(40)
    c = carriles_de(kp, sc, valid, fps_origen=30.0)
    assert c.nombres == ["A-left", "A-right", "B-left", "B-right"]


def test_sin_identidad_resuelta_el_cuadro_no_es_usable():
    # Un cuadro sin identidad no es un cuadro sin golpe: es un cuadro sin dato. Si entrara
    # como usable, el decodificador podria abrir un segmento sobre keypoints en cero.
    kp, sc, valid = _esqueleto(40)
    valid[0, 10:20] = False
    c = carriles_de(kp, sc, valid, fps_origen=30.0)
    assert not c.usable[0, 10:20].any()
    assert not c.usable[1, 10:20].any(), "los dos carriles de A"
    assert c.usable[2, 10:20].all(), "B no se ve afectado"


def test_una_forma_que_no_es_la_de_dos_peleadores_se_rechaza():
    kp, sc, valid = _esqueleto(30)
    with pytest.raises(ValueError, match=r"\(2, T, K, 2\)"):
        carriles_de(kp[0], sc[0], valid[0], fps_origen=30.0)


def test_un_fps_invalido_se_rechaza_antes_de_calcular():
    kp, sc, valid = _esqueleto(30)
    with pytest.raises(ValueError, match="positivo"):
        carriles_de(kp, sc, valid, fps_origen=0.0)


# -- deteccion --------------------------------------------------------------


class _EnsambleFalso:
    """Un ensamble que no aprende nada: devuelve la senal que se le pide."""

    class _Est:
        @staticmethod
        def aplicar(x):
            return x

    def __init__(self, senal_por_carril):
        self.senal = senal_por_carril
        self.estandarizador = self._Est()
        self.modelos = [object()]
        self.config = None
        self.n = 1


@pytest.fixture
def sin_torch(monkeypatch):
    """Corta la inferencia real: lo que se prueba aca es el cableado, no la TCN."""
    import boxtwin_detector.inferencia as inf

    llamadas = {"i": 0}

    def fake_predecir(modelo, x, cfg, device):
        # probabilidad_de_golpe se aplica despues, asi que hay que devolver logits de 3
        # clases donde O es la 0 y B/I son golpe.
        senal = fake_predecir.senal[llamadas["i"]]
        llamadas["i"] += 1
        logits = np.zeros((len(senal), 3), np.float32)
        logits[:, 0] = np.where(senal > 0.5, -10.0, 10.0)
        logits[:, 2] = np.where(senal > 0.5, 10.0, -10.0)
        return logits

    monkeypatch.setattr(inf, "predecir_secuencia", fake_predecir)
    monkeypatch.setattr(inf, "probabilidad_de_golpe",
                        lambda lg: 1.0 / (1.0 + np.exp(-(lg[:, 2] - lg[:, 0]))))
    return fake_predecir


def test_un_golpe_sale_con_su_cuadro_y_su_segundo(sin_torch):
    kp, sc, valid = _esqueleto(90)
    c = carriles_de(kp, sc, valid, fps_origen=30.0)
    senal = [np.zeros(90) for _ in range(4)]
    senal[0][20:28] = 1.0            # ocho cuadros de golpe en A-left
    sin_torch.senal = senal

    golpes = detectar(_EnsambleFalso(senal), c, umbral=0.5, device="cpu")
    assert len(golpes) == 1
    g = golpes[0]
    assert (g.peleador, g.brazo) == ("A", "left")
    assert (g.inicio, g.fin) == (20, 27)
    assert g.t_inicio == pytest.approx(20 / 30)
    assert g.t_fin == pytest.approx(28 / 30)


def test_el_cuadro_sale_en_el_timebase_del_video_y_no_en_el_del_modelo(sin_torch):
    # El error que este test existe para atrapar: devolver el cuadro del modelo. Sobre un
    # video a 60 fps eso pone cada golpe al doble de velocidad y el enlace al video cae a
    # la mitad de la sesion.
    kp, sc, valid = _esqueleto(180)
    c = carriles_de(kp, sc, valid, fps_origen=60.0)
    senal = [np.zeros(c.T) for _ in range(4)]
    senal[3][30:38] = 1.0
    sin_torch.senal = senal

    golpes = detectar(_EnsambleFalso(senal), c, umbral=0.5, device="cpu")
    assert len(golpes) == 1
    g = golpes[0]
    assert (g.peleador, g.brazo) == ("B", "right")
    assert g.inicio == 60, "cuadro 30 del modelo es el 60 del video"
    assert g.t_inicio == pytest.approx(1.0)


def test_un_tramo_sin_identidad_no_genera_golpe(sin_torch):
    kp, sc, valid = _esqueleto(90)
    valid[0, 15:40] = False
    c = carriles_de(kp, sc, valid, fps_origen=30.0)
    senal = [np.zeros(90) for _ in range(4)]
    senal[0][20:30] = 1.0
    sin_torch.senal = senal

    assert detectar(_EnsambleFalso(senal), c, umbral=0.5, device="cpu") == []


def test_los_golpes_salen_ordenados_en_el_tiempo(sin_torch):
    kp, sc, valid = _esqueleto(120)
    c = carriles_de(kp, sc, valid, fps_origen=30.0)
    senal = [np.zeros(120) for _ in range(4)]
    senal[2][10:18] = 1.0    # B-left primero
    senal[0][60:68] = 1.0    # A-left despues
    sin_torch.senal = senal

    golpes = detectar(_EnsambleFalso(senal), c, umbral=0.5, device="cpu")
    assert [g.inicio for g in golpes] == sorted(g.inicio for g in golpes)
    assert golpes[0].peleador == "B"


def test_un_segmento_mas_corto_que_el_minimo_no_es_un_golpe(sin_torch):
    # Tres cuadros a 30 fps son 100 ms. El p10 de la duracion medida es 5 cuadros.
    kp, sc, valid = _esqueleto(60)
    c = carriles_de(kp, sc, valid, fps_origen=30.0)
    senal = [np.zeros(60) for _ in range(4)]
    senal[1][20:23] = 1.0
    sin_torch.senal = senal

    assert detectar(_EnsambleFalso(senal), c, umbral=0.5, device="cpu") == []


def test_la_forma_de_carriles_es_la_que_espera_el_modelo():
    kp, sc, valid = _esqueleto(50)
    c = carriles_de(kp, sc, valid, fps_origen=30.0)
    assert isinstance(c, Carriles)
    assert c.features.dtype == np.float32
    assert c.usable.dtype == np.bool_
