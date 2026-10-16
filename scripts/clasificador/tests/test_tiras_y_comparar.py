import csv
import json

import cv2
import numpy as np

from extraer_tiras import Golpe, dibujar, escribir_lote, indices_muestra, region


def _video(ruta, n=40, ancho=320, alto=240):
    w = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"MJPG"), 30, (ancho, alto))
    for i in range(n):
        img = np.full((alto, ancho, 3), 40, np.uint8)
        cv2.putText(img, str(i), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        w.write(img)
    w.release()
    return ruta


def _golpe(video, fuente="f1", ev="ev_0001", inicio=10, fin=18, tipo="hook", alt="body"):
    T = fin - inicio + 1 + 4
    kp = np.zeros((T, 17, 2), np.float32)
    kp[:, :, 0] = np.linspace(80, 140, 17)
    kp[:, :, 1] = np.linspace(40, 200, 17)
    kp_r = kp.copy()
    kp_r[:, :, 0] += 100
    sc = np.ones((T, 17), np.float32)
    return Golpe(fuente=fuente, event_id=ev, fighter="fighter_A", side="left", start_frame=inicio,
                 end_frame=fin, video=video, kp_atac=kp, sc_atac=sc, kp_rival=kp_r, sc_rival=sc.copy(),
                 primer_cuadro=inicio - 2, punch_type=tipo, target=alt)


def test_indices_muestra():
    assert indices_muestra(13) == [0, 2, 3, 5, 7, 9, 10, 12]
    assert indices_muestra(3) == [0, 1, 2]


def test_region_sin_pose_es_cuadro_entero(tmp_path):
    g = _golpe(tmp_path / "v.avi")
    g.sc_atac[:] = 0
    g.sc_rival[:] = 0
    assert region(g, 320, 240) == (0, 0, 320, 240)


def test_lote_ciego_y_reanudable(trabajo, tmp_path):
    video = _video(tmp_path / "v.avi")
    golpes = [_golpe(video, ev=f"ev_{k:04d}", inicio=5 + k, fin=12 + k,
                     tipo=["straight", "hook", "uppercut"][k % 3]) for k in range(5)]
    lote = trabajo / "lotes" / "prueba"
    assert escribir_lote(golpes, lote, seed=1) == 5
    tiras = sorted(p.name for p in (lote / "tiras").iterdir())
    assert len(tiras) == 5 and all(t.startswith("T") for t in tiras)
    # nada de la etiqueta en lo que ve el etiquetador
    for linea in (lote / "lote.jsonl").read_text().splitlines():
        assert set(json.loads(linea)) == {"id", "brazo"}
    for t in tiras:
        assert not any(x in t for x in ("hook", "straight", "uppercut", "head", "body", "ev_"))
    img = cv2.imread(str(lote / "tiras" / tiras[0]))
    assert img is not None and img.shape[0] > 40
    # reanudar no redibuja ni cambia ids
    clave1 = (lote / "clave" / "clave.csv").read_text()
    assert escribir_lote(golpes, lote, seed=1) == 0
    assert (lote / "clave" / "clave.csv").read_text() == clave1


def test_dibujar_marca_el_brazo(tmp_path):
    video = _video(tmp_path / "v.avi")
    g = _golpe(video)
    from extraer_tiras import leer_cuadros
    cuadros = leer_cuadros(video, g.primer_cuadro, indices_muestra(len(g.sc_atac)))
    img = dibujar(g, "T0001", cuadros)
    amarillo = np.all(np.abs(img.astype(int) - (0, 230, 255)) < 30, axis=2)
    assert amarillo.sum() > 50


def _clave(lote, filas):
    (lote / "clave").mkdir(parents=True)
    with (lote / "clave" / "clave.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "origen", "fuente", "event_id", "fighter", "side",
                                           "start_frame", "end_frame", "punch_type", "target"])
        w.writeheader()
        for i, (fu, t, a) in enumerate(filas):
            w.writerow({"id": f"T{i:04d}", "origen": "anotado", "fuente": fu, "event_id": f"ev{i}",
                        "fighter": "fighter_A", "side": "left", "start_frame": 0, "end_frame": 5,
                        "punch_type": t, "target": a})


def test_comparar_lectura_estricta(trabajo):
    from comparar_etiquetas import comparar
    lote = trabajo / "lotes" / "p"
    _clave(lote, [("f1", "hook", "head"), ("f1", "straight", "body"), ("f2", "uppercut", "head"),
                  ("f2", "hook", "body")])
    (lote / "respuestas").mkdir()
    (lote / "respuestas" / "a.jsonl").write_text("\n".join([
        json.dumps({"id": "T0000", "tipo": "hook", "altura": "head", "confianza": "alta"}),
        json.dumps({"id": "T0001", "tipo": "straight", "altura": "body"}),
        json.dumps({"id": "T0002", "tipo": "no_se", "altura": "head"}),
        json.dumps({"id": "T0000", "tipo": "straight", "altura": "body"}),   # duplicado: vale el primero
    ]) + "\n")
    r = comparar(lote / "clave" / "clave.csv", lote / "respuestas", 0.85, 0.80)
    t = r["total"]
    assert t["familia"]["exactitud"] == 0.5             # T0002 no_se y T0003 sin respuesta son error
    assert t["cobertura_tipo"] == 0.5
    assert t["altura"]["recall_cuerpo"] == 0.5          # T0003 sin respuesta
    assert t["altura"]["balanceada"] == 0.75
    assert not r["pasa"]
    assert any("dos veces" in a for a in r["avisos"])
    assert set(r["por_fuente"]) == {"f1", "f2"}


def test_comparar_rechaza_lote_sin_etiqueta(trabajo):
    import pytest
    from comparar_etiquetas import comparar
    lote = trabajo / "lotes" / "nuevo"
    _clave(lote, [("s", "", "")])
    (lote / "respuestas").mkdir()
    with pytest.raises(SystemExit):
        comparar(lote / "clave" / "clave.csv", lote / "respuestas", 0.85, 0.8)
