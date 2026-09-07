import numpy as np
import pytest

from boxtwin_detector.splits import BANDA_POR_DEFECTO, Fold, en_distribucion, leave_one_source_out


def test_un_fold_por_fuente():
    folds = leave_one_source_out(["a", "b", "c"])
    assert len(folds) == 3
    assert folds[0] == Fold("sin-a", ("b", "c"), ("a",))


def test_ninguna_fuente_esta_en_train_y_val_a_la_vez():
    for f in leave_one_source_out(["sparring-3", "Sparring", "pacquiao"]):
        assert not set(f.train) & set(f.val)
        assert len(f.train) + len(f.val) == 3


def test_una_sola_fuente_es_error():
    with pytest.raises(ValueError):
        leave_one_source_out(["a"])


def test_en_distribucion_no_se_pisan_y_dejan_la_banda():
    tr, va = en_distribucion(1000, fraccion=0.25, banda=60)
    assert not (tr & va).any()
    assert va.sum() == 250
    assert tr.sum() == 690
    # la banda es un hueco continuo entre los dos
    assert not tr[690:750].any() and not va[690:750].any()


def test_la_banda_impide_que_una_ventana_toque_los_dos_lados():
    banda = BANDA_POR_DEFECTO
    tr, va = en_distribucion(2000, fraccion=0.2, banda=banda)
    ultimo_train = np.where(tr)[0][-1]
    primer_val = np.where(va)[0][0]
    assert primer_val - ultimo_train > banda


def test_sin_banda_quedan_pegados():
    tr, va = en_distribucion(100, fraccion=0.2, banda=0)
    assert np.where(tr)[0][-1] + 1 == np.where(va)[0][0]


def test_fraccion_invalida():
    for f in (0.0, 1.0, -0.3, 2.0):
        with pytest.raises(ValueError):
            en_distribucion(1000, fraccion=f)


def test_si_no_queda_entrenamiento_es_error():
    with pytest.raises(ValueError):
        en_distribucion(50, fraccion=0.9, banda=60)


def test_partir_en_distribucion_no_comparte_cuadros():
    import numpy as np
    from boxtwin_detector.dataset import Fuente
    from boxtwin_detector.splits import partir_en_distribucion

    T = 2000
    f = Fuente(nombre="x", features=np.zeros((4, T, 20), np.float32),
               labels=np.zeros((4, T), np.int8), usable=np.ones((4, T), bool),
               interpolado=np.zeros((4, T), bool),
               carriles=["a", "b", "c", "d"], fps=30.0,
               conteos={"cobertura": [100, 1899]})
    tr, va = partir_en_distribucion(f, fraccion=0.25, banda=60)
    assert not (tr.usable & va.usable).any()
    assert tr.usable.sum() > 0 and va.usable.sum() > 0
    # nada fuera de la cobertura original
    assert not tr.usable[:, :100].any() and not va.usable[:, 1900:].any()
    # la banda muerta existe
    ultimo_tr = np.where(tr.usable[0])[0][-1]
    primer_va = np.where(va.usable[0])[0][0]
    assert primer_va - ultimo_tr > 60
