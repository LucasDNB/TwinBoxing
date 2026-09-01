import numpy as np
import pytest

torch = pytest.importorskip("torch")

from boxtwin_detector.dataset import Fuente
from boxtwin_detector.entrenamiento import (
    Config, Estandarizador, entrenar, metricas, pesos_de_clase, predecir_secuencia,
    sortear_ventanas,
)
from boxtwin_detector.modelo import TCN


def fuente(nombre="f", T=1000, n_carriles=4, F=20, semilla=0, cobertura=None, densidad=0.05):
    rng = np.random.default_rng(semilla)
    feats = rng.normal(size=(n_carriles, T, F)).astype(np.float32)
    labels = np.zeros((n_carriles, T), np.int8)
    usable = np.ones((n_carriles, T), bool)
    for c in range(n_carriles):
        for s in range(20, T - 20, int(1 / densidad)):
            labels[c, s] = 1
            labels[c, s + 1 : s + 7] = 2
            # la senal: las features de un golpe son distintas
            feats[c, s : s + 7] += 3.0
    cob = cobertura or [0, T - 1]
    usable[:, : cob[0]] = False
    usable[:, cob[1] + 1 :] = False
    return Fuente(nombre=nombre, features=feats, labels=labels, usable=usable,
                  carriles=[f"c{i}" for i in range(n_carriles)], fps=30.0,
                  conteos={"cobertura": cob})


# -- estandarizacion -------------------------------------------------------

def test_la_estandarizacion_usa_solo_los_cuadros_usables():
    f = fuente(T=200)
    f.features[:, :100] = 1000.0     # basura, toda enmascarada
    f.usable[:, :100] = False
    est = Estandarizador.ajustar([f])
    assert np.abs(est.media).max() < 50, "la basura enmascarada no puede mover la media"


def test_una_feature_constante_no_divide_por_cero():
    f = fuente(T=200)
    f.features[:, :, 3] = 7.0
    est = Estandarizador.ajustar([f])
    assert est.desvio[3] == 1.0
    assert np.isfinite(est.aplicar(f.features[0])).all()


def test_estandarizar_centra_y_escala():
    f = fuente(T=500)
    est = Estandarizador.ajustar([f])
    z = est.aplicar(f.features[0][f.usable[0]])
    assert np.abs(z.mean(0)).max() < 0.3
    assert np.abs(z.std(0) - 1).max() < 0.3


def test_sin_cuadros_usables_es_error():
    f = fuente(T=100)
    f.usable[:] = False
    with pytest.raises(ValueError):
        Estandarizador.ajustar([f])


def test_ida_y_vuelta_del_estandarizador():
    est = Estandarizador.ajustar([fuente(T=200)])
    otro = Estandarizador.de_dict(est.a_dict())
    assert np.allclose(est.media, otro.media) and np.allclose(est.desvio, otro.desvio)


# -- pesos -----------------------------------------------------------------

def test_los_pesos_favorecen_a_la_clase_rara():
    w = pesos_de_clase([fuente(T=2000)], alpha=0.5)
    assert w[1] > w[2] > w[0], "B es la mas rara, O la mas comun"
    assert np.isclose(w.mean(), 1.0, atol=1e-5)


def test_alpha_cero_no_corrige_nada():
    w = pesos_de_clase([fuente(T=1000)], alpha=0.0)
    assert np.allclose(w, 1.0)


def test_alpha_mas_alto_estira_la_razon_entre_la_rara_y_la_comun():
    f = fuente(T=2000)
    razon = lambda a: pesos_de_clase([f], a)[1] / pesos_de_clase([f], a)[0]
    assert razon(0.0) == pytest.approx(1.0)
    assert razon(0.25) < razon(0.5) < razon(1.0)


def test_con_el_desbalance_real_el_inverso_puro_se_desmadra():
    # Reparto medido del dataset: O 95,31%, I 4,05%, B 0,64%.
    f = fuente(T=100)
    f.labels[:] = 0
    f.labels[0, :3] = 1
    f.labels[0, 3:23] = 2
    f.usable[:] = True
    r_medio = pesos_de_clase([f], 0.5)[1] / pesos_de_clase([f], 0.5)[0]
    r_uno = pesos_de_clase([f], 1.0)[1] / pesos_de_clase([f], 1.0)[0]
    assert r_uno > 10 * r_medio, "por esto alpha es 0,5 y no 1"


# -- ventanas --------------------------------------------------------------

def test_las_ventanas_caen_dentro_del_tramo_anotado():
    # El tensor de Pacquiao tiene 68.250 cuadros y el round anotado son 5.000.
    f = fuente(T=20000, cobertura=[1000, 6000])
    cfg = Config(ventana=256, ventanas_por_epoca=200)
    for _, _, s in sortear_ventanas([f], cfg, np.random.default_rng(0)):
        assert 1000 <= s <= 6000 - 256


def test_se_descartan_las_ventanas_casi_vacias():
    f = fuente(T=3000)
    f.usable[:, 1000:2000] = False
    cfg = Config(ventana=256, ventanas_por_epoca=200, min_usable_en_ventana=0.9)
    for _, c, s in sortear_ventanas([f], cfg, np.random.default_rng(1)):
        assert f.usable[c, s : s + 256].mean() >= 0.9


def test_sin_tramo_suficiente_es_error():
    with pytest.raises(ValueError):
        sortear_ventanas([fuente(T=100)], Config(ventana=256), np.random.default_rng(0))


# -- inferencia por trozos -------------------------------------------------

def test_inferir_por_trozos_da_lo_mismo_que_de_una():
    # Si el solape estuviera mal, aparecerian artefactos cada N cuadros y no se verian
    # en ninguna metrica agregada.
    m = TCN(n_features=8, canales=16).eval()
    x = np.random.default_rng(0).normal(size=(700, 8)).astype(np.float32)
    dev = torch.device("cpu")
    completo = predecir_secuencia(m, x, Config(trozo_inferencia=10_000), dev)
    troceado = predecir_secuencia(m, x, Config(trozo_inferencia=128), dev)
    assert np.abs(completo - troceado).max() < 1e-4


def test_inferir_devuelve_un_logit_por_cuadro():
    m = TCN(n_features=8, canales=16).eval()
    x = np.zeros((333, 8), np.float32)
    assert predecir_secuencia(m, x, Config(), torch.device("cpu")).shape == (333, 3)


# -- metricas --------------------------------------------------------------

def test_las_metricas_ignoran_lo_enmascarado():
    y = np.array([0, 0, 1, 2])
    pred = np.array([0, 9, 1, 2])          # el cuadro 1 esta mal pero enmascarado
    m = metricas(pred, y, np.array([True, False, True, True]))
    assert m["n"] == 3 and m["f1_macro"] == 1.0


def test_decir_siempre_O_queda_expuesto():
    y = np.array([0] * 95 + [1] * 2 + [2] * 3)
    m = metricas(np.zeros(100, int), y, np.ones(100, bool))
    assert m["siempre_O"] == 0.95
    assert m["B_recall"] == 0.0
    assert m["f1_macro"] < 0.35, "F1 macro no se deja enganar por la clase mayoritaria"


# -- loop ------------------------------------------------------------------

def test_entrenar_aprende_una_senal_sintetica():
    tr = [fuente("tr", T=3000, semilla=1)]
    va = [fuente("va", T=1200, semilla=2)]
    est = Estandarizador.ajustar(tr)
    cfg = Config(epocas=6, ventanas_por_epoca=64, batch=8, ventana=256, trozo_inferencia=4096)
    m = TCN(n_features=20, canales=32)
    hist = entrenar(m, tr, va, est, cfg, torch.device("cpu"), verbose=False)
    assert len(hist["epocas"]) == 6
    primera = hist["epocas"][0]["perdida"]
    ultima = hist["epocas"][-1]["perdida"]
    assert ultima < primera, f"la perdida no bajo: {primera} -> {ultima}"
    assert hist["mejor"]["val"]["f1_macro"] > 0.4


def test_entrenar_deja_el_mejor_estado_y_no_el_ultimo():
    tr = [fuente("tr", T=2000, semilla=3)]
    va = [fuente("va", T=800, semilla=4)]
    est = Estandarizador.ajustar(tr)
    cfg = Config(epocas=4, ventanas_por_epoca=32, batch=8, ventana=256)
    m = TCN(n_features=20, canales=32)
    hist = entrenar(m, tr, va, est, cfg, torch.device("cpu"), verbose=False)
    mejor = max(e["val"]["f1_macro"] for e in hist["epocas"])
    assert hist["mejor"]["val"]["f1_macro"] == mejor


def test_la_parada_temprana_corta_cuando_deja_de_mejorar():
    tr = [fuente("tr", T=1500, semilla=5)]
    va = [fuente("va", T=600, semilla=6)]
    est = Estandarizador.ajustar(tr)
    cfg = Config(epocas=50, ventanas_por_epoca=16, batch=8, ventana=256, paciencia=2)
    hist = entrenar(TCN(n_features=20, canales=16), tr, va, est, cfg,
                    torch.device("cpu"), verbose=False)
    assert len(hist["epocas"]) < 50
    assert "parada_temprana" in hist


def test_sin_paciencia_corre_todas_las_epocas():
    tr = [fuente("tr", T=1200, semilla=7)]
    va = [fuente("va", T=500, semilla=8)]
    est = Estandarizador.ajustar(tr)
    cfg = Config(epocas=3, ventanas_por_epoca=16, batch=8, ventana=256, paciencia=0)
    hist = entrenar(TCN(n_features=20, canales=16), tr, va, est, cfg,
                    torch.device("cpu"), verbose=False)
    assert len(hist["epocas"]) == 3 and "parada_temprana" not in hist


def test_dos_corridas_con_la_misma_semilla_dan_lo_mismo():
    # Sin sembrar antes de construir el modelo, los pesos se inicializan con el estado
    # global que hubiera y dos corridas "iguales" difieren hasta 0,1 de F1 por evento.
    from boxtwin_detector.entrenamiento import sembrar

    tr = [fuente("tr", T=1200, semilla=11)]
    va = [fuente("va", T=500, semilla=12)]
    est = Estandarizador.ajustar(tr)

    def corrida(semilla):
        cfg = Config(epocas=3, ventanas_por_epoca=16, batch=8, ventana=256, semilla=semilla)
        sembrar(semilla)
        m = TCN(n_features=20, canales=32)
        h = entrenar(m, tr, va, est, cfg, torch.device("cpu"), verbose=False)
        return [e["perdida"] for e in h["epocas"]]

    assert corrida(42) == corrida(42)
    assert corrida(42) != corrida(7)


def test_sembrar_fija_la_inicializacion_de_los_pesos():
    from boxtwin_detector.entrenamiento import sembrar

    sembrar(3); a = TCN(n_features=20, canales=32).entrada.weight.detach().clone()
    sembrar(3); b = TCN(n_features=20, canales=32).entrada.weight.detach().clone()
    sembrar(4); c = TCN(n_features=20, canales=32).entrada.weight.detach().clone()
    assert torch.equal(a, b) and not torch.equal(a, c)
