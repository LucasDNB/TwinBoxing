import numpy as np
import pytest


def _bt(geo):
    fuentes = [f for f in geo.FUENTES for _ in range(3)]
    return {"frame_dir": np.array([f"{f}_{i}" for i, f in enumerate(fuentes)]), "fuente": np.array(fuentes)}


def _bw(n_rondas=8, por_ronda=2):
    fuentes = [f"r{i}" for i in range(n_rondas) for _ in range(por_ronda)]
    part = ["test" if int(f[1:]) < 2 else "train" for f in fuentes]
    return {"frame_dir": np.array([f"bw{i}" for i in range(len(fuentes))]), "fuente": np.array(fuentes),
            "particion": np.array(part)}


def test_bm1b_deja_una_fuente_afuera_sin_boxingweb(trabajo):
    import entrenar
    from bm import geo
    bt = _bt(geo)
    ps = entrenar.particiones("BM-1b", bt, None)
    assert [p[0] for p in ps] == list(geo.FUENTES)
    for nombre, tr, trbw, te in ps:
        assert len(trbw) == 0 and set(bt["fuente"][te]) == {nombre} and nombre not in set(bt["fuente"][tr])
        assert len(tr) + len(te) == len(bt["frame_dir"])


def test_bm2_suma_todo_boxingweb_y_bm1_entrena_una_vez(trabajo):
    import entrenar
    from bm import geo
    bt, bw = _bt(geo), _bw()
    for _, _, trbw, _ in entrenar.particiones("BM-2", bt, bw):
        assert len(trbw) == len(bw["frame_dir"])
    ps = entrenar.particiones("BM-1", bt, bw)
    assert len(ps) == 1 and ps[0][0] == entrenar.TODAS and len(ps[0][3]) == len(bt["frame_dir"])


def test_bm0b_usa_la_particion_de_boxmind(trabajo):
    import entrenar
    from bm import geo
    bw = _bw()
    (_, trbt, tr, te), = entrenar.particiones("BM-0b", _bt(geo), bw)
    assert len(trbt) == 0 and set(bw["particion"][tr]) == {"train"} and set(bw["particion"][te]) == {"test"}


def test_curva_sortea_rondas_completas_una_sola_vez(trabajo):
    import entrenar
    from bm import geo
    bt, bw = _bt(geo), _bw(n_rondas=8)
    r25 = entrenar.rondas_de_fraccion(list(bw["fuente"]), 0.25)
    assert r25 == entrenar.rondas_de_fraccion(list(bw["fuente"]), 0.25) and len(r25) == 2
    _, _, trbw, _ = entrenar.particiones("BM-2-50", bt, bw)[0]
    assert len(set(bw["fuente"][trbw])) == 4
    with pytest.raises(SystemExit):
        entrenar.particiones("BM-2-30", bt, bw)


def test_p_altura_suma_pecho_y_abdomen():
    import entrenar
    po = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    np.testing.assert_allclose(entrenar.p_altura_de(po), [[0.6, 0.4], [0.2, 0.8]])
