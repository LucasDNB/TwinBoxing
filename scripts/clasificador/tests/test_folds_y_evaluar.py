import json
import pickle

import numpy as np
import pytest

import comun

CLASES12 = [f"{t}-{e}-{a}" for t in comun.TIPOS for e in ("left", "right") for a in comun.ALTURAS]


def _exports(trabajo, variante, clases, por_fuente=6, seed=0):
    rng = np.random.default_rng(seed)
    d = trabajo / "exports" / variante
    d.mkdir(parents=True)
    for f in comun.FUENTES:
        ms = [{"frame_dir": f"{f}_ev_{i:04d}", "label": int(rng.integers(len(clases))),
               "keypoint": np.zeros((1, 4, 17, 2), np.float32)} for i in range(por_fuente)]
        with (d / f"{f}.mmaction.pkl").open("wb") as fh:
            pickle.dump({"split": {f: [m["frame_dir"] for m in ms]}, "annotations": ms}, fh)
        (d / f"{f}.mmaction.meta.json").write_text(json.dumps({"classes": clases}))


def test_folds_sin_fuga_y_remapeo(trabajo):
    import armar_folds
    _exports(trabajo, "v12s_at", CLASES12)
    m = armar_folds.armar("E2", trabajo / "exports", trabajo / "folds" / "E2")
    assert m["clases"] == comun.CLASES_TIPO_ALTURA
    assert set(m["folds"]) == set(comun.FUENTES)
    for f, v in m["folds"].items():
        assert v["test"] == 6 and v["train"] == 6 * (len(comun.FUENTES) - 1)
        with (trabajo / "folds" / "E2" / f"fold_{f}.pkl").open("rb") as fh:
            d = pickle.load(fh)
        assert all(n.startswith(f + "_") for n in d["split"]["test"])
        assert not any(n.startswith(f + "_") for n in d["split"]["train"])
    # el remapeo conserva tipo y altura y tira el lado
    with (trabajo / "exports" / "v12s_at" / "Sparring.mmaction.pkl").open("rb") as fh:
        orig = {m["frame_dir"]: CLASES12[m["label"]] for m in pickle.load(fh)["annotations"]}
    with (trabajo / "folds" / "E2" / "fold_Sparring.pkl").open("rb") as fh:
        nuevo = {m["frame_dir"]: comun.CLASES_TIPO_ALTURA[m["label"]] for m in pickle.load(fh)["annotations"]}
    for k, v in orig.items():
        assert nuevo[k] == comun.clase_tipo_altura(v)


def test_folds_rechaza_clases_distintas(trabajo):
    import armar_folds
    _exports(trabajo, "v12s_at", CLASES12)
    meta = trabajo / "exports" / "v12s_at" / "Sparring.mmaction.meta.json"
    meta.write_text(json.dumps({"classes": list(reversed(CLASES12))}))
    with pytest.raises(SystemExit):
        armar_folds.armar("E2", trabajo / "exports", trabajo / "folds" / "E2")


def _dumps(trabajo, exp, perfecto=True, desordenar=None):
    import evaluar
    manifest = json.loads((trabajo / "folds" / exp / "manifest.json").read_text())
    C = len(manifest["clases"])
    for f in comun.FUENTES:
        _, y = evaluar.esperado(trabajo / "folds" / exp / f"fold_{f}.pkl")
        for s in comun.SEMILLAS:
            P = np.full((len(y), C), 0.01)
            P[np.arange(len(y)), y if perfecto else (y + 1) % C] = 0.9
            gt = y[::-1] if desordenar == f else y
            items = [{"pred_score": P[i], "gt_label": np.array([gt[i]])} for i in range(len(y))]
            wd = trabajo / "corridas" / exp / f / f"s{s}"
            wd.mkdir(parents=True)
            with (wd / "pred.pkl").open("wb") as fh:
                pickle.dump(items, fh)


def test_evaluar_perfecto_da_uno(trabajo):
    import armar_folds
    import evaluar
    _exports(trabajo, "v12s_at", CLASES12)
    armar_folds.armar("E2", trabajo / "exports", trabajo / "folds" / "E2")
    _dumps(trabajo, "E2")
    r = evaluar.evaluar("E2")
    assert r["completo"]
    t = r["ensamble_total"]
    assert t["top1"] == 1.0 and t["familia"]["exactitud"] == 1.0
    assert t["altura"]["balanceada"] == 1.0
    assert "eje" not in t                                   # E2 no tiene lado
    assert r["ensamble_fuentes_c8"]["golpes"] == 6 * len(comun.FUENTES_C8)
    assert r["dispersion_entre_semillas"]["top1"]["desvio"] == 0.0


def test_evaluar_detecta_desorden(trabajo):
    import armar_folds
    import evaluar
    _exports(trabajo, "v12s_at", CLASES12, seed=3)
    armar_folds.armar("E2", trabajo / "exports", trabajo / "folds" / "E2")
    _dumps(trabajo, "E2", desordenar="03-sparring")
    with pytest.raises(SystemExit):
        evaluar.evaluar("E2")


def test_evaluar_e1_reporta_eje_y_familia(trabajo):
    import armar_folds
    import evaluar
    clases6 = [f"{t}-{e}" for t in comun.TIPOS for e in ("lead", "rear")]
    _exports(trabajo, "v6lr_at", clases6)
    armar_folds.armar("E1", trabajo / "exports", trabajo / "folds" / "E1")
    _dumps(trabajo, "E1", perfecto=False)                   # (y+1) % 6: mismo tipo, rol cambiado o tipo siguiente
    r = evaluar.evaluar("E1")
    t = r["ensamble_total"]
    assert "altura" not in t and "eje" in t
    assert t["top1"] == 0.0
    assert t["eje"]["exactitud"] == 0.0                      # +1 siempre invierte lead/rear


def test_evaluar_incompleto_lo_dice(trabajo):
    import armar_folds
    import evaluar
    _exports(trabajo, "v12s_at", CLASES12)
    armar_folds.armar("E2", trabajo / "exports", trabajo / "folds" / "E2")
    _dumps(trabajo, "E2")
    (trabajo / "corridas" / "E2" / "Sparring" / "s2" / "pred.pkl").unlink()
    r = evaluar.evaluar("E2")
    assert not r["completo"] and r["faltan"] == ["Sparring/s2"]
    assert r["por_fold"]["Sparring"]["semillas"] == 2
    assert "s2" not in r["por_semilla_total"]


def test_generar_configs_toca_solo_lo_previsto(tmp_path):
    import generar_configs

    class D(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    cfg = D(model=D(cls_head=D(num_classes=6)),
            train_dataloader=D(dataset=D(dataset=D(ann_file="x", split=None), times=10)),
            test_dataloader=D(dataset=D(ann_file="y")),
            val_dataloader=D(), val_cfg=D(), val_evaluator=[], train_cfg=D(max_epochs=10, val_interval=1),
            default_hooks=D(checkpoint=D(save_best="auto")), load_from="rel.pth", optim_wrapper="igual")
    out = generar_configs.ajustar(cfg, 6, tmp_path / "f.pkl", tmp_path / "wd", 2, tmp_path / "ini.pth")
    assert out.val_dataloader is None and out.val_cfg is None and out.val_evaluator is None
    assert out.train_cfg == {"type": "EpochBasedTrainLoop", "max_epochs": 10}
    assert out.default_hooks.checkpoint["save_best"] is None
    assert out.randomness == {"seed": 2, "deterministic": False}
    assert out.train_dataloader.dataset.dataset.split == "train"
    assert out.test_dataloader.dataset.split == "test"
    assert out.optim_wrapper == "igual"
    assert out.train_dataloader.dataset.times == 10


CLASES6_LR = [f"{t}-{e}" for t in comun.TIPOS for e in ("lead", "rear")]


def test_clase_ocho_parte_el_recto_por_rol():
    assert comun.clase_ocho("straight-head", "lead") == "jab-head"
    assert comun.clase_ocho("straight-body", "rear") == "cross-body"
    assert comun.clase_ocho("hook-body", "lead") == "hook-body"
    with pytest.raises(ValueError):
        comun.clase_ocho("straight-head", None)
    with pytest.raises(ValueError):
        comun.clase_ocho("straight-left-head", "lead")


def test_evaluar_ocho_clases_usa_el_rol_de_e1(trabajo):
    import armar_folds
    import evaluar
    _exports(trabajo, "v12s_at", CLASES12)
    _exports(trabajo, "v6lr_at", CLASES6_LR, seed=1)
    armar_folds.armar("E1", trabajo / "exports", trabajo / "folds" / "E1")
    armar_folds.armar("E2", trabajo / "exports", trabajo / "folds" / "E2")
    _dumps(trabajo, "E2")
    t = evaluar.evaluar("E2")["ensamble_total"]
    assert t["ocho_clases"]["exactitud"] == 1.0
    # el rol de cada recto real es el de la etiqueta de E1 del mismo frame_dir
    roles = evaluar.roles_e1()
    n_jab = n_cross = 0
    for f in comun.FUENTES:
        nombres, y = evaluar.esperado(trabajo / "folds" / "E2" / f"fold_{f}.pkl")
        for n, i in zip(nombres, y):
            c = comun.clase_ocho(comun.CLASES_TIPO_ALTURA[i], roles[n])
            n_jab += c.startswith("jab")
            n_cross += c.startswith("cross")
    rec = t["ocho_clases"]["recall"]
    assert sum(rec[c]["n"] for c in ("jab-head", "jab-body")) == n_jab
    assert sum(rec[c]["n"] for c in ("cross-head", "cross-body")) == n_cross
    assert "ocho_clases" in t and "eje" not in t


def test_evaluar_ocho_clases_recto_predicho_toma_el_rol_real(trabajo):
    import armar_folds
    import evaluar
    _exports(trabajo, "v12s_at", CLASES12)
    _exports(trabajo, "v6lr_at", CLASES6_LR, seed=1)
    armar_folds.armar("E1", trabajo / "exports", trabajo / "folds" / "E1")
    armar_folds.armar("E2", trabajo / "exports", trabajo / "folds" / "E2")
    # el modelo dice siempre straight-head: acierta las 8 clases justo en los rectos a la cabeza
    sh = comun.CLASES_TIPO_ALTURA.index("straight-head")
    total = aciertos = 0
    for f in comun.FUENTES:
        _, y = evaluar.esperado(trabajo / "folds" / "E2" / f"fold_{f}.pkl")
        total += len(y)
        aciertos += int((y == sh).sum())
        for s in comun.SEMILLAS:
            P = np.full((len(y), 6), 0.01)
            P[:, sh] = 0.9
            wd = trabajo / "corridas" / "E2" / f / f"s{s}"
            wd.mkdir(parents=True)
            with (wd / "pred.pkl").open("wb") as fh:
                pickle.dump([{"pred_score": P[i], "gt_label": np.array([y[i]])} for i in range(len(y))], fh)
    t = evaluar.evaluar("E2")["ensamble_total"]
    assert t["ocho_clases"]["exactitud"] == round(aciertos / total, 4)


def test_evaluar_ocho_clases_sin_e1_lo_avisa_y_con_hueco_frena(trabajo):
    import armar_folds
    import evaluar
    _exports(trabajo, "v12s_at", CLASES12)
    armar_folds.armar("E2", trabajo / "exports", trabajo / "folds" / "E2")
    _dumps(trabajo, "E2")
    r = evaluar.evaluar("E2")
    assert "ocho_clases" not in r["ensamble_total"] and r["avisos"]
    _exports(trabajo, "v6lr_at", CLASES6_LR, por_fuente=5)  # un golpe por fuente sin etiqueta de E1
    armar_folds.armar("E1", trabajo / "exports", trabajo / "folds" / "E1")
    with pytest.raises(SystemExit):
        evaluar.evaluar("E2")
