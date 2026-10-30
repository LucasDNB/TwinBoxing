import numpy as np

import features2d


def _esqueleto(T=10, x0=100.0, y_hombros=200.0, torso=100.0):
    """Un peleador de frente: hombros en y_hombros, caderas torso pixeles mas abajo."""
    kp = np.zeros((T, 17, 2), np.float32)
    kp[:, 5] = (x0 - 20, y_hombros); kp[:, 6] = (x0 + 20, y_hombros)
    kp[:, 11] = (x0 - 15, y_hombros + torso); kp[:, 12] = (x0 + 15, y_hombros + torso)
    kp[:, 7] = (x0 - 30, y_hombros + 40); kp[:, 8] = (x0 + 30, y_hombros + 40)
    kp[:, 9] = (x0 - 30, y_hombros + 80); kp[:, 10] = (x0 + 30, y_hombros + 80)
    kp[:, 0] = (x0, y_hombros - 40)
    return kp


def _par(T=10):
    kp = np.stack([_esqueleto(T, x0=100), _esqueleto(T, x0=300)])
    sc = np.ones((2, T, 17), np.float32)
    valid = np.ones((2, T), bool)
    return kp, sc, valid


def test_rival_a_la_altura_de_los_hombros_da_h_cero():
    kp, sc, valid = _par()
    kp[0, :, 9] = (250.0, 200.0)                 # muneca izquierda de A a la altura de los hombros de B
    f = features2d.features_rival(kp, sc, valid, p=0, side="left")
    assert np.allclose(f[:, 0], 0.0)
    assert np.allclose(f[:, 1], 50 / 100)        # |250 - 300| / torso
    assert np.allclose(f[:, 3], 1.0) and np.allclose(f[:, 4], 0.0)   # torso vertical
    assert np.allclose(f[:, 5], 1.0)
    kp[0, :, 9] = (250.0, 250.0)                 # medio torso mas abajo
    assert np.allclose(features2d.features_rival(kp, sc, valid, 0, "left")[:, 0], 0.5)


def test_rival_sin_pose_da_rival_ok_cero_y_ceros():
    kp, sc, valid = _par()
    valid[1, 3:6] = False
    sc[1, 7, 11] = 0.1                           # una cadera floja
    f = features2d.features_rival(kp, sc, valid, p=0, side="left")
    malos = [3, 4, 5, 7]
    assert np.all(f[malos, 5] == 0) and np.all(f[malos, :5] == 0)
    buenos = [i for i in range(10) if i not in malos]
    assert np.all(f[buenos, 5] == 1)


def test_remuestreo_de_rampa_conserva_la_rampa():
    x = np.arange(17, dtype=np.float32)          # rampa de 17 cuadros
    y = features2d.remuestrear(x, 32)
    assert y.shape == (32,)
    assert np.allclose(y, np.linspace(0, 16, 32))
    X = np.stack([x, 2 * x + 1], -1)
    Y = features2d.remuestrear(X, 32)
    assert np.allclose(Y[:, 1], 2 * Y[:, 0] + 1)


def test_mascara_por_vecino_mas_cercano():
    m = np.array([1, 1, 0, 0, 1], bool)
    r = features2d.remuestrear_mascara(m, 9)
    assert r.dtype == bool and r.astype(int).tolist() == [1, 1, 1, 0, 0, 0, 0, 1, 1]
    assert r[0] and r[-1] and not r[4]


def test_construir_formas():
    kp, sc, valid = _par(T=60)
    golpes = [{"fuente": "F", "p": 0, "side": "left", "ini": 5, "fin": 30, "tipo": "hook",
               "altura": "body", "rol": "lead", "frame_dir": "F_ev_0001"},
              {"fuente": "F", "p": 1, "side": "right", "ini": 0, "fin": 59, "tipo": "straight",
               "altura": "head", "rol": "rear", "frame_dir": "F_ev_0002"}]
    d = features2d.construir(golpes, {"F": {"kp": kp, "sc": sc, "valid": valid}})
    assert d["X"].shape == (2, 32, 26) and d["M"].shape == (2, 32)
    assert d["y_tipo"].tolist() == [1, 0] and d["y_altura"].tolist() == [1, 0] and d["y_rol"].tolist() == [0, 1]


# -- T3 y T5 (mejoras post-E8): umbrales y relleno en E4, apagados por defecto ---------------

import json  # noqa: E402

GOLPES = [{"fuente": "F", "p": 0, "side": "left", "ini": 5, "fin": 30, "tipo": "hook",
           "altura": "body", "rol": "lead", "frame_dir": "F_ev_0001"},
          {"fuente": "F", "p": 1, "side": "right", "ini": 0, "fin": 59, "tipo": "straight",
           "altura": "head", "rol": "rear", "frame_dir": "F_ev_0002"}]
ARTIC = ("nariz", "hombro", "codo", "muneca", "cadera")


def _serie_con_huecos(T=60):
    kp, sc, valid = _par(T)
    rng = np.random.default_rng(5)
    kp = kp + rng.normal(size=kp.shape).astype(np.float32) * 3
    sc = (rng.uniform(0, 1, size=sc.shape) ** 0.3).astype(np.float32)
    sc[0, 12:14, 9] = 0.05                       # hueco de 2 en la muneca izquierda de A
    return {"F": {"kp": kp, "sc": sc, "valid": valid, "fps": 30.0}}


def _umbrales(tmp_path, **cambios):
    u = {a: 0.3 for a in ARTIC}
    u.update(cambios)
    p = tmp_path / "umbrales.json"
    p.write_text(json.dumps({"umbrales": u}))
    return p


def test_apagado_da_los_mismos_tensores(tmp_path):
    series = _serie_con_huecos()
    base = features2d.construir(GOLPES, series)
    u, _ = features2d.leer_umbrales(_umbrales(tmp_path))
    igual = features2d.construir(GOLPES, series, umbrales=u)
    assert base["X"].tobytes() == igual["X"].tobytes()
    assert base["M"].tobytes() == igual["M"].tobytes()
    assert "R" not in base


def test_los_umbrales_llegan_al_brazo_y_al_rival(tmp_path):
    series = _serie_con_huecos()
    base = features2d.construir(GOLPES, series)
    u, _ = features2d.leer_umbrales(_umbrales(tmp_path, muneca=0.95, cadera=0.95))
    otro = features2d.construir(GOLPES, series, umbrales=u)
    assert otro["M"].sum() < base["M"].sum()
    assert otro["X"][..., 25].sum() < base["X"][..., 25].sum()     # rival_ok


def test_con_relleno_suma_la_mascara_r(tmp_path):
    series = _serie_con_huecos()
    d = features2d.construir(GOLPES, series, relleno=True)
    assert d["R"].shape == d["M"].shape and d["R"].any()


def test_opciones_desde_el_entorno(tmp_path, monkeypatch):
    monkeypatch.delenv("BOXTWIN_UMBRALES", raising=False)
    monkeypatch.delenv("BOXTWIN_RELLENO", raising=False)
    assert features2d.opciones_de_entorno() == (None, None, False)
    p = _umbrales(tmp_path, muneca=0.2)
    monkeypatch.setenv("BOXTWIN_UMBRALES", str(p))
    monkeypatch.setenv("BOXTWIN_RELLENO", "1")
    u, sha, rell = features2d.opciones_de_entorno()
    assert u["muneca"] == 0.2 and len(sha) == 64 and rell


def test_main_registra_el_sha256_en_el_meta(trabajo, tmp_path, monkeypatch):
    import importlib
    import features2d as f2
    importlib.reload(f2)
    d = trabajo / "datos"
    (d / "series").mkdir(parents=True)
    s = _serie_con_huecos()["F"]
    np.savez_compressed(d / "series" / "F.npz", kp=s["kp"], sc=s["sc"], valid=s["valid"], fps=30.0)
    import datos
    datos.escribir_csv([{**g, "event_id": "e", "fighter": "fighter_A", "guardia": "orthodox",
                         "fps": 30.0, "start_frame": g["ini"], "end_frame": g["fin"]}
                        for g in GOLPES], d / "golpes.csv")
    monkeypatch.setenv("BOXTWIN_UMBRALES", str(_umbrales(tmp_path, muneca=0.2)))
    monkeypatch.setenv("BOXTWIN_RELLENO", "1")
    assert f2.main([]) == 0
    meta = json.loads((d / "golpes2d.meta.json").read_text())
    assert len(meta["umbrales_sha256"]) == 64 and meta["relleno"] is True
    assert "R" in np.load(d / "golpes.npz").files
