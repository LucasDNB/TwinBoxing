"""T8: recache.py (01-percepcion.md §4, reglas duras 1, 2 y 4)."""

import json

import pytest


def _original(tmp_path, dir_="anotacion-spar-01", fuente="01-sparring"):
    """Un proyecto de A con su video, como en la copia principal."""
    raiz = tmp_path / "TwinBoxing"
    video = raiz / dir_ / "videos" / f"{fuente}.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    (raiz / dir_ / "annotations").mkdir()
    (raiz / dir_ / "annotations" / f"{fuente}.annot.json").write_text(
        json.dumps({"video": {"path": str(video)}}))
    return raiz, video


def test_el_comando_lleva_project_explicito_y_los_parametros_fijos(gemelo, tmp_path):
    import recache
    proj = gemelo / "raiz-y26" / "anotacion-spar-01"
    cmd = recache.comando(proj / "videos" / "01-sparring.mp4", proj, gemelo / "modelos" / "yolo26l-pose.pt",
                          tracker=tmp_path / "botsort.yaml", device="0")
    assert cmd[cmd.index("--project") + 1] == str(proj)
    for flag, valor in (("--imgsz", "640"), ("--conf", "0.25"), ("--iou", "0.7"),
                        ("--cut-threshold", "0.3"), ("--model", str(gemelo / "modelos" / "yolo26l-pose.pt"))):
        assert cmd[cmd.index(flag) + 1] == valor
    assert "--half" not in cmd and "--force" not in cmd and "--restart" not in cmd


def test_preparar_enlaza_el_video_original_sin_copiarlo(gemelo, tmp_path):
    import recache
    raiz_a, video = _original(tmp_path)
    proj, enlace = recache.preparar("01-sparring", gemelo / "raiz-y26", raiz_a)
    assert proj == gemelo / "raiz-y26" / "anotacion-spar-01"
    assert enlace.is_symlink() and enlace.resolve() == video.resolve()
    assert recache.preparar("01-sparring", gemelo / "raiz-y26", raiz_a)[1] == enlace   # idempotente


def test_proyecto_fuera_del_gemelo_se_niega(gemelo, tmp_path):
    import guardas
    import recache
    raiz_a, _ = _original(tmp_path)
    with pytest.raises(guardas.GuardaError):
        recache.preparar("01-sparring", raiz_a, raiz_a)       # el proyecto seria el de A


def test_destino_con_annot_que_no_escribio_el_puente_se_niega(gemelo, tmp_path):
    import guardas
    import recache
    raiz_a, _ = _original(tmp_path)
    ann = gemelo / "raiz-y26" / "anotacion-spar-01" / "annotations" / "01-sparring.annot.json"
    ann.parent.mkdir(parents=True)
    ann.write_text(json.dumps({"identity": {"assignments": [{"origin": {"annotator": "lucasb"}}]}}))
    with pytest.raises(guardas.GuardaError):
        recache.preparar("01-sparring", gemelo / "raiz-y26", raiz_a)


def test_correr_deja_tiempo_y_fps_y_saltea_lo_hecho(gemelo, tmp_path):
    import recache
    raiz_a, _ = _original(tmp_path)
    llamadas = []

    def ejecutar(cmd):
        llamadas.append(cmd)
        proj = __import__("pathlib").Path(cmd[cmd.index("--project") + 1])
        (proj / "cache").mkdir(exist_ok=True)
        (proj / "cache" / "01-sparring.meta.json").write_text(json.dumps(
            {"video": {"total_frames": 600}, "inference": {"model_sha256": "ab" * 32}}))
        (proj / "cache" / "01-sparring.pose.npz").write_bytes(b"npz")

    reloj = iter([100.0, 130.0])
    registro = gemelo / "e8" / "recache.json"
    recache.correr(["01-sparring"], gemelo / "raiz-y26", raiz_a, gemelo / "modelos" / "yolo26l-pose.pt",
                   registro, ejecutar=ejecutar, reloj=lambda: next(reloj))
    r = json.loads(registro.read_text())["01-sparring"]
    assert r["segundos"] == 30.0 and r["cuadros"] == 600 and r["fps"] == 20.0
    assert r["model_sha256"] == "ab" * 32
    recache.correr(["01-sparring"], gemelo / "raiz-y26", raiz_a, gemelo / "modelos" / "yolo26l-pose.pt",
                   registro, ejecutar=ejecutar, reloj=lambda: 0.0)
    assert len(llamadas) == 1                                  # reanudable: lo hecho no se rehace
