import numpy as np
import pytest

torch = pytest.importorskip("torch")

from boxtwin_detector.dataset import Fuente
from boxtwin_detector.entrenamiento import Config
from boxtwin_detector.ensamble import (
    Ensamble, cargar, entrenar_ensamble, guardar, probabilidad,
)
from boxtwin_detector.modelo import TCN

CPU = torch.device("cpu")


def fuente(nombre="f", T=900, n_carriles=2, F=20, semilla=0):
    rng = np.random.default_rng(semilla)
    feats = rng.normal(size=(n_carriles, T, F)).astype(np.float32)
    labels = np.zeros((n_carriles, T), np.int8)
    for c in range(n_carriles):
        for s in range(20, T - 20, 40):
            labels[c, s] = 1
            labels[c, s + 1 : s + 7] = 2
            feats[c, s : s + 7] += 3.0
    return Fuente(nombre=nombre, features=feats, labels=labels,
                  usable=np.ones((n_carriles, T), bool),
                  interpolado=np.zeros((n_carriles, T), bool),
                  carriles=[f"c{i}" for i in range(n_carriles)], fps=30.0,
                  conteos={"cobertura": [0, T - 1]})


def _cfg(**kw):
    base = dict(epocas=2, ventanas_por_epoca=8, batch=4, ventana=256, canales=16)
    return Config(**{**base, **kw})


# -- la salida no esta cuantizada -------------------------------------------

def test_el_promedio_no_queda_cuantizado_en_pasos_de_1_sobre_n():
    # Parecia que promediar n modelos saturados daba pasos de 1/n, y sobre esa teoria se
    # barrieron solo n umbrales: el barrido se salteaba el optimo y el ensamble parecia
    # peor que una corrida sola. Los modelos no estan tan saturados.
    f = fuente("v", T=700, semilla=9)
    ens = entrenar_ensamble([fuente()], [f], _cfg(), [1, 2, 3], CPU, verbose=False)
    p = np.concatenate(probabilidad(ens, f, CPU))
    escalones = np.array([0.0, 1 / 3, 2 / 3, 1.0])
    lejos = np.abs(p[:, None] - escalones[None, :]).min(axis=1) > 0.02
    assert lejos.mean() > 0.05, (
        "si el promedio cayera siempre sobre los escalones del voto, barrer n umbrales "
        "alcanzaria; no es el caso"
    )


# -- construccion ----------------------------------------------------------

def test_un_ensamble_sin_modelos_es_error():
    from boxtwin_detector.entrenamiento import Estandarizador
    est = Estandarizador.ajustar([fuente()])
    with pytest.raises(ValueError):
        Ensamble([], est, _cfg(), [])


def test_semillas_y_modelos_tienen_que_coincidir():
    from boxtwin_detector.entrenamiento import Estandarizador
    est = Estandarizador.ajustar([fuente()])
    with pytest.raises(ValueError):
        Ensamble([TCN(n_features=20, canales=16)], est, _cfg(), [1, 2])


def test_sin_semillas_es_error():
    with pytest.raises(ValueError):
        entrenar_ensamble([fuente()], [fuente("v", semilla=1)], _cfg(), [], CPU)


# -- entrenamiento y prediccion --------------------------------------------

def test_entrena_un_modelo_por_semilla():
    ens = entrenar_ensamble([fuente()], [fuente("v", semilla=1)], _cfg(), [1, 2, 3],
                            CPU, verbose=False)
    assert ens.n == 3 and len(ens.historiales) == 3


def test_las_semillas_dan_modelos_distintos():
    # Si dieran el mismo modelo, promediar no aportaria nada.
    ens = entrenar_ensamble([fuente()], [fuente("v", semilla=1)], _cfg(), [1, 2],
                            CPU, verbose=False)
    a = ens.modelos[0].entrada.weight.detach()
    b = ens.modelos[1].entrada.weight.detach()
    assert not torch.equal(a, b)


def test_la_estandarizacion_es_unica_para_todo_el_ensamble():
    tr = [fuente()]
    ens = entrenar_ensamble(tr, [fuente("v", semilla=1)], _cfg(), [1, 2], CPU, verbose=False)
    from boxtwin_detector.entrenamiento import Estandarizador
    esperada = Estandarizador.ajustar(tr)
    assert np.allclose(ens.estandarizador.media, esperada.media)


def test_la_probabilidad_es_una_senal_por_carril_en_cero_uno():
    f = fuente("v", T=600, semilla=2)
    ens = entrenar_ensamble([fuente()], [f], _cfg(), [1, 2], CPU, verbose=False)
    p = probabilidad(ens, f, CPU)
    assert len(p) == f.features.shape[0]
    assert all(x.shape == (f.T,) for x in p)
    todo = np.concatenate(p)
    assert todo.min() >= 0.0 and todo.max() <= 1.0


def test_la_probabilidad_es_el_promedio_y_no_otra_cosa():
    from boxtwin_detector.decodificacion import probabilidad_de_golpe
    from boxtwin_detector.entrenamiento import predecir_secuencia
    f = fuente("v", T=400, semilla=3)
    ens = entrenar_ensamble([fuente()], [f], _cfg(), [1, 2], CPU, verbose=False)
    p = probabilidad(ens, f, CPU)[0]
    x = ens.estandarizador.aplicar(f.features[0])
    esperado = np.mean([
        probabilidad_de_golpe(predecir_secuencia(m, x, ens.config, CPU))
        for m in ens.modelos
    ], axis=0)
    assert np.abs(p - esperado).max() < 1e-5


def test_un_ensamble_de_uno_es_el_modelo_solo():
    f = fuente("v", T=400, semilla=4)
    ens = entrenar_ensamble([fuente()], [f], _cfg(), [7], CPU, verbose=False)
    assert ens.n == 1
    p = probabilidad(ens, f, CPU)[0]
    assert np.isfinite(p).all()


def test_el_ensamble_es_determinista():
    tr, va = [fuente()], [fuente("v", semilla=5)]
    a = entrenar_ensamble(tr, va, _cfg(), [1, 2], CPU, verbose=False)
    b = entrenar_ensamble(tr, va, _cfg(), [1, 2], CPU, verbose=False)
    pa, pb = probabilidad(a, va[0], CPU)[0], probabilidad(b, va[0], CPU)[0]
    assert np.abs(pa - pb).max() == 0.0


# -- persistencia ----------------------------------------------------------

def test_ida_y_vuelta_por_disco(tmp_path):
    f = fuente("v", T=500, semilla=6)
    ens = entrenar_ensamble([fuente()], [f], _cfg(), [1, 2, 3], CPU, verbose=False)
    p = guardar(ens, tmp_path / "ens.pt")
    otro = cargar(p, CPU)
    assert otro.n == 3 and otro.semillas == [1, 2, 3]
    assert np.abs(probabilidad(ens, f, CPU)[0] - probabilidad(otro, f, CPU)[0]).max() < 1e-6


def test_cargar_un_checkpoint_que_no_es_ensamble(tmp_path):
    p = tmp_path / "otro.pt"
    torch.save({"kind": "otra.cosa"}, p)
    with pytest.raises(ValueError, match="no es un ensamble"):
        cargar(p, CPU)
