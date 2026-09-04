import json

import numpy as np
import pytest

from boxtwin_detector.bio import B, I, Segmento
from boxtwin_detector.dataset import (
    construir, escribir, indice_remuestreo, leer, remuestrear_segmentos,
)
from boxtwin_detector.features import N_FEATURES

L_HOM, R_HOM = 5, 6


# -- remuestreo ------------------------------------------------------------

def test_indice_de_60_a_30_toma_uno_de_cada_dos():
    idx = indice_remuestreo(100, 60.0, 30.0)
    assert len(idx) == 50
    assert idx[:4].tolist() == [0, 2, 4, 6]


def test_indice_no_se_pasa_del_final():
    idx = indice_remuestreo(7, 59.94, 30.0)
    assert idx.max() <= 6


def test_indice_a_mismo_fps_es_la_identidad():
    idx = indice_remuestreo(50, 30.0, 30.0)
    assert idx.tolist() == list(range(50))


def test_fps_invalido_es_error():
    with pytest.raises(ValueError):
        indice_remuestreo(10, 0.0, 30.0)


def test_los_segmentos_conservan_su_duracion_en_segundos():
    # 12 cuadros a 60 fps son 200 ms; a 30 fps tienen que ser 6 cuadros
    segs = [Segmento(100, 111, 0)]
    r, aj = remuestrear_segmentos(segs, 60.0, 30.0, 1000)
    assert aj == 0
    assert r[0].largo == 6
    assert r[0].inicio == 50


def test_a_60_y_30_dos_golpes_adyacentes_no_colisionan():
    # Con el mapeo semiabierto, lo que no se solapaba no se solapa. Es la propiedad que
    # hace que el remuestreo real no pierda golpes.
    segs = [Segmento(10, 13, 0), Segmento(14, 17, 0)]
    r, aj = remuestrear_segmentos(segs, 60.0, 30.0, 100)
    assert aj == 0
    assert [(s.inicio, s.fin) for s in r] == [(5, 6), (7, 8)]


def test_si_un_golpe_se_comprime_a_cero_se_recorta_el_siguiente_y_se_cuenta():
    # Caso extremo, con una razon de fps que no existe en el proyecto: es la unica via por
    # la que dos segmentos pueden terminar encima.
    segs = [Segmento(10, 10, 0), Segmento(11, 11, 0)]
    r, aj = remuestrear_segmentos(segs, 300.0, 30.0, 100)
    assert aj == 1
    assert r[0].fin < r[1].inicio, "los segmentos no pueden solaparse"


def test_el_remuestreo_nunca_deja_segmentos_solapados():
    rng = np.random.default_rng(7)
    for _ in range(200):
        origen = float(rng.choice([30.0, 50.0, 59.94, 120.0]))
        destino = float(rng.choice([15.0, 25.0, 30.0]))
        segs, f = [], 0
        for _ in range(rng.integers(1, 12)):
            f += int(rng.integers(1, 8))
            largo = int(rng.integers(0, 10))
            segs.append(Segmento(f, f + largo, 0))
            f += largo + 1
        r, _ = remuestrear_segmentos(segs, origen, destino, 500)
        for x, y in zip(r, r[1:]):
            assert x.fin < y.inicio


def test_un_segmento_que_cae_fuera_del_nuevo_largo_se_descarta():
    r, aj = remuestrear_segmentos([Segmento(90, 95, 0)], 60.0, 30.0, 40)
    assert r == [] and aj == 1


def test_el_remuestreo_conserva_la_cantidad_de_golpes():
    segs = [Segmento(i * 20, i * 20 + 11, 0) for i in range(10)]
    r, aj = remuestrear_segmentos(segs, 60.0, 30.0, 200)
    assert len(r) == 10 and aj == 0


# -- construccion completa -------------------------------------------------

def _export_falso(tmp_path, fps, T, segmentos_por_carril, clases=None):
    """Un export `sequence` sintetico, con la forma que escribe boxtwin."""
    clases = clases or [f"c{i}" for i in range(13)] + ["feint"]
    rng = np.random.default_rng(0)
    kp = rng.normal(size=(2, T, 17, 2)) * 20 + 300
    kp[:, :, L_HOM] = kp[:, :, R_HOM] + np.array([50.0, 0.0])
    sc = np.ones((2, T, 17))
    labels = np.zeros((2, 2, T), np.int16)
    for (p, c), segs in segmentos_por_carril.items():
        for s in segs:
            labels[p, c, s.inicio] = 1 + 2 * s.clase
            if s.fin > s.inicio:
                labels[p, c, s.inicio + 1 : s.fin + 1] = 2 + 2 * s.clase
    npz = tmp_path / "falso.sequence.npz"
    np.savez_compressed(
        npz, labels=labels, valid=np.ones((2, T), bool),
        interpolated=np.zeros((2, T), bool), keypoints=kp.astype(np.float32),
        kp_score=sc.astype(np.float32), fighters=np.array(["fighter_A", "fighter_B"]),
        lanes=np.array(["left", "right"]), bio_names=np.array(["O"]),
        classes=np.array(clases),
    )
    (tmp_path / "falso.sequence.meta.json").write_text(json.dumps({
        "video": {"name": "falso.mp4", "sha256": "abc", "fps": fps, "total_frames": T},
        "annot_sha256": "def", "channels": "per-arm", "label_space": "side",
        "classes": clases,
    }))
    return npz


def test_construye_cuatro_carriles(tmp_path):
    npz = _export_falso(tmp_path, 30.0, 200, {(0, 0): [Segmento(50, 56, 0)]})
    f = construir(npz)
    assert f.features.shape == (4, 200, N_FEATURES)
    assert f.carriles == ["A-left", "A-right", "B-left", "B-right"]
    assert f.conteos["golpes"] == 1


def test_el_amague_se_enmascara_y_no_queda_como_fondo(tmp_path):
    # Es la decision central del armado: un amague no es "no hay golpe", es "no sabemos".
    npz = _export_falso(tmp_path, 30.0, 200, {(0, 0): [Segmento(50, 56, 13)]})
    f = construir(npz)
    assert f.conteos["golpes"] == 0
    assert f.conteos["amagues_enmascarados"] == 1
    assert not f.usable[0, 50:57].any()
    assert (f.labels[0, 50:57] == 0).all()


def test_pacquiao_a_60_queda_con_la_mitad_de_cuadros(tmp_path):
    npz = _export_falso(tmp_path, 59.94, 1000, {(0, 0): [Segmento(100, 111, 0)]})
    f = construir(npz)
    assert f.T == int(np.floor(1000 * 30.0 / 59.94))
    segs = np.where(f.labels[0] == B)[0]
    assert len(segs) == 1
    largo = int((f.labels[0] != 0).sum())
    assert largo == 6, f"12 cuadros a 60 fps tienen que ser 6 a 30, no {largo}"


def test_export_por_peleador_es_rechazado(tmp_path):
    npz = _export_falso(tmp_path, 30.0, 100, {})
    meta = tmp_path / "falso.sequence.meta.json"
    d = json.loads(meta.read_text())
    d["channels"] = "per-fighter"
    meta.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="per-arm"):
        construir(npz)


def test_export_sin_amagues_es_rechazado(tmp_path):
    npz = _export_falso(tmp_path, 30.0, 100, {}, clases=[f"c{i}" for i in range(6)])
    with pytest.raises(ValueError, match="amagues"):
        construir(npz)


def test_falta_la_metadata(tmp_path):
    npz = _export_falso(tmp_path, 30.0, 100, {})
    (tmp_path / "falso.sequence.meta.json").unlink()
    with pytest.raises(FileNotFoundError):
        construir(npz)


def test_ida_y_vuelta_por_disco(tmp_path):
    npz = _export_falso(tmp_path, 30.0, 300, {(1, 1): [Segmento(80, 88, 2)]})
    f = construir(npz)
    salida, _ = escribir(f, tmp_path / "data")
    g = leer(salida)
    assert np.array_equal(f.features, g.features)
    assert np.array_equal(f.labels, g.labels)
    assert np.array_equal(f.usable, g.usable)
    assert g.carriles == f.carriles
    assert g.procedencia["annot_sha256"] == "def"


def test_los_conteos_cierran(tmp_path):
    npz = _export_falso(tmp_path, 30.0, 400, {
        (0, 0): [Segmento(50, 56, 0), Segmento(100, 108, 1)],
        (1, 1): [Segmento(200, 205, 2)],
    })
    f = construir(npz)
    c = f.conteos
    assert c["golpes"] == 3
    assert c["cuadros_B"] == 3, "una B por golpe"
    assert c["cuadros_O"] + c["cuadros_B"] + c["cuadros_I"] == c["cuadros_usables"]


# -- cobertura -------------------------------------------------------------

def test_fuera_del_tramo_anotado_no_queda_como_fondo(tmp_path):
    # De Pacquiao solo esta anotado el round 1 de 12. Sin esto, los otros 11 entrarian
    # como "no hay golpe" sobre metraje donde nadie miro.
    npz = _export_falso(tmp_path, 30.0, 1000, {(0, 0): [Segmento(400, 406, 0)]})
    f = construir(npz)
    assert f.conteos["cobertura"] == [400, 406]
    assert not f.usable[:, :400].any()
    assert not f.usable[:, 407:].any()
    assert f.usable[:, 400:407].any()
    assert f.conteos["cuadros_fuera_de_cobertura"] > 0


def test_la_cobertura_es_de_la_fuente_y_no_de_cada_carril(tmp_path):
    # El anotador mira los dos peleadores sobre el mismo tramo de video.
    npz = _export_falso(tmp_path, 30.0, 500, {
        (0, 0): [Segmento(100, 106, 0)],
        (1, 1): [Segmento(300, 306, 0)],
    })
    f = construir(npz)
    assert f.conteos["cobertura"] == [100, 306]
    assert f.usable[2, 150:250].all(), "el fondo entre dos golpes sigue siendo fondo"


def test_fuente_sin_golpes_queda_toda_enmascarada(tmp_path):
    npz = _export_falso(tmp_path, 30.0, 200, {})
    f = construir(npz)
    assert f.conteos["cuadros_usables"] == 0


# -- pose interpolada ------------------------------------------------------

def _con_interpolados(tmp_path, marcados):
    """Como _export_falso pero marcando cuadros de pose rellenada."""
    npz = _export_falso(tmp_path, 30.0, 300, {(0, 0): [Segmento(100, 108, 0)]})
    d = dict(np.load(npz))
    interp = np.zeros_like(d["interpolated"])
    interp[0, marcados] = True
    d["interpolated"] = interp
    np.savez_compressed(npz, **d)
    return npz


def test_por_defecto_la_pose_interpolada_entra(tmp_path):
    npz = _con_interpolados(tmp_path, slice(100, 105))
    f = construir(npz)
    assert f.conteos["interpolados_en_mascara"] is True
    assert f.usable[0, 100:105].all()


def test_sin_interpolados_salen_de_la_mascara(tmp_path):
    # No son un dato observado: una recta entre dos puntos no tiene la firma temporal
    # que el detector busca.
    npz = _con_interpolados(tmp_path, slice(100, 105))
    f = construir(npz, interpolados=False)
    assert f.conteos["interpolados_en_mascara"] is False
    assert not f.usable[0, 100:105].any()
    # el resto del mismo golpe, que si se midio, sigue en la mascara
    assert f.usable[0, 105:109].all(), "solo salen los cuadros rellenados"


def test_sacar_interpolados_no_toca_al_otro_peleador(tmp_path):
    npz = _con_interpolados(tmp_path, slice(100, 105))
    f = construir(npz, interpolados=False)
    assert f.usable[2, 100:105].all(), "los carriles de B no se marcaron"


def test_el_conteo_de_interpolados_se_reporta(tmp_path):
    npz = _con_interpolados(tmp_path, slice(50, 70))
    f = construir(npz)
    assert f.conteos["cuadros_interpolados"] == 20
