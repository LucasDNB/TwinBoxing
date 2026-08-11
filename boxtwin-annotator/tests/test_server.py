"""
Servidor: API y transporte.

La API se testea sin socket, llamando a los metodos, y el transporte se testea con un
servidor real en un puerto libre. Lo segundo importa mas de lo que parece: el cache
inmutable de los cuadros es lo que hace que retroceder sobre lo ya visto no genere ni una
peticion, y eso vive en las cabeceras, no en la logica.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from boxtwin.core.posecache import N_KEYPOINTS, FrameStatus, PoseArrays, write_pose_cache  # noqa: E402
from boxtwin.server.api import MAX_POSE_RANGE, Api  # noqa: E402
from boxtwin.server.frames import FrameRenderer  # noqa: E402
from boxtwin.server.http import build_handler  # noqa: E402
from boxtwin.server.session import Session  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="hace falta ffmpeg")

W, H, N = 64, 48, 25
SHA = "d" * 64


@pytest.fixture(scope="module")
def proyecto(tmp_path_factory: pytest.TempPathFactory) -> Path:
    raiz = tmp_path_factory.mktemp("proy")
    (raiz / "videos").mkdir()
    (raiz / "cache").mkdir()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc=duration=1:size={W}x{H}:rate=25",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "12",
         str(raiz / "videos" / "mini.mp4")],
        check=True,
    )
    meta = {
        "format_version": 1, "kind": "boxtwin.pose_meta", "keypoint_format": "coco17",
        "video": {
            "path": "videos/mini.mp4", "sha256": SHA, "size_bytes": 1, "width": W, "height": H,
            "mtime": "2026-08-11T10:00:00-03:00", "codec": "h264", "fps": 25.0,
            "fps_declared": 25.0, "fps_source": "container_verified", "duration_s": 1.0,
            "total_frames": N, "total_frames_declared": N,
        },
        "counts": {"n_detections": N}, "resume_seams": [{"frame": 10, "id_offset": 3}],
    }
    arrays = PoseArrays(
        frame_index=np.arange(N + 1, dtype=np.int64),
        frame_status=np.full(N, FrameStatus.OK, np.uint8),
        track_id=np.ones(N, np.int32),
        bbox=np.tile(np.array([1.0, 1.0, 20.0, 30.0], np.float32), (N, 1)),
        det_conf=np.full(N, 0.9, np.float32),
        keypoints=np.arange(N * N_KEYPOINTS * 2, dtype=np.float32).reshape(N, N_KEYPOINTS, 2),
        kp_score=np.full((N, N_KEYPOINTS), 0.8, np.float32),
    )
    write_pose_cache(raiz / "cache" / "mini.pose.npz", arrays, meta)
    (raiz / "cache" / "mini.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return raiz


@pytest.fixture
def api(proyecto: Path):
    s = Session.open(proyecto / "videos" / "mini.mp4")
    yield Api(s, FrameRenderer(s))
    s.close()


# -- API -------------------------------------------------------------------


def test_meta_trae_lo_que_el_cliente_necesita(api: Api) -> None:
    m = api.meta()
    assert m["video"]["total_frames"] == N
    assert m["video"]["fps"] == 25.0
    assert m["seams"] == [10]
    assert set(m["fighters"]) == {"fighter_A", "fighter_B"}


def test_meta_manda_esqueleto_y_paleta(api: Api) -> None:
    """
    El overlay se dibuja en el navegador. Si el cliente tuviera su propia copia de los
    indices COCO, tarde o temprano dejaria de coincidir y el esqueleto saldria cruzado.
    """
    m = api.meta()
    assert len(m["skeleton"]["names"]) == 19  # 17 + los dos guantes
    assert len(m["skeleton"]["edges"]) == 19
    assert m["skeleton"]["glove_edges"] == [[9, 17], [10, 18]]
    assert set(m["colors"]) >= {"fighter_A", "fighter_B", "ignore", "unassigned", "punch"}


def test_poses_por_rango(api: Api) -> None:
    d = api.poses(5, 10)
    assert (d["from"], d["to"]) == (5, 10)
    assert len(d["frames"]) == 5
    det = d["frames"][0][0]
    assert det["track_id"] == 1
    assert len(det["kp"]) == 34  # 17 pares (x, y) aplanados
    assert len(det["ks"]) == 17


def test_poses_recorta_el_rango(api: Api) -> None:
    """Un pedido enorme no puede hacer que el servidor arme un JSON de cien megas."""
    d = api.poses(0, 100000)
    assert d["to"] - d["from"] <= MAX_POSE_RANGE
    assert d["to"] <= N


def test_poses_fuera_del_video(api: Api) -> None:
    d = api.poses(N + 50, N + 100)
    assert d["frames"] == []


def test_poses_redondea_para_achicar_el_json(api: Api) -> None:
    det = api.poses(3, 4)["frames"][0][0]
    for v in det["kp"]:
        assert round(v, 1) == v
    for v in det["ks"]:
        assert round(v, 2) == v


def test_annotation_e_issues(api: Api) -> None:
    a = api.annotation()
    assert a["kind"] == "boxtwin.annot"
    assert isinstance(api.issues()["issues"], list)


# -- render ----------------------------------------------------------------


def test_jpeg_valido_y_cacheado(api: Api) -> None:
    datos = api.renderer.jpeg(5)
    assert datos is not None
    assert datos[:2] == b"\xff\xd8"  # marca de inicio JPEG
    antes = api.renderer.stats()["encoded"]
    api.renderer.jpeg(5)
    assert api.renderer.stats()["encoded"] == antes  # el segundo salio del cache
    assert api.renderer.stats()["cache_hits"] >= 1


def test_jpeg_fuera_de_rango(api: Api) -> None:
    assert api.renderer.jpeg(-1) is None
    assert api.renderer.jpeg(N + 10) is None


def test_hires_sin_video_original(proyecto: Path, tmp_path: Path) -> None:
    """Sin el original el cliente se queda con el proxy, no se rompe."""
    copia = tmp_path / "p"
    shutil.copytree(proyecto, copia)
    (copia / "videos" / "mini.mp4").unlink()
    # sin video no hay proxy tampoco en esta fixture, asi que se copia uno
    shutil.copy(proyecto / "videos" / "mini.mp4", copia / "cache" / "mini.proxy.mp4")
    s = Session.open(copia / "videos" / "mini.mp4")
    try:
        assert FrameRenderer(s).hires_jpeg(3) is None
    finally:
        s.close()


# -- transporte ------------------------------------------------------------


@pytest.fixture
def servidor(api: Api):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(api))
    srv.daemon_threads = True
    hilo = threading.Thread(target=srv.serve_forever, daemon=True)
    hilo.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def pedir(url: str):
    return urllib.request.urlopen(url, timeout=5)


def test_sirve_el_cliente(servidor: str) -> None:
    for ruta, marca in (("/", b"<canvas"), ("/static/app.js", b"function"),
                        ("/static/style.css", b"#lienzo")):
        r = pedir(servidor + ruta)
        assert r.status == 200
        assert marca in r.read()


def test_cuadros_con_cache_inmutable(servidor: str) -> None:
    """
    Es lo que convierte el cache del navegador en buffer del cliente: retroceder sobre lo
    ya visto no genera ni una peticion.
    """
    r = pedir(f"{servidor}/api/frame/4.jpg")
    assert r.status == 200
    assert r.headers["Content-Type"] == "image/jpeg"
    assert "immutable" in r.headers["Cache-Control"]
    assert r.read()[:2] == b"\xff\xd8"


def test_json_no_se_cachea(servidor: str) -> None:
    """La anotacion cambia mientras se trabaja: cachearla mostraria datos viejos."""
    r = pedir(f"{servidor}/api/meta")
    assert r.headers["Cache-Control"] == "no-store"


def test_poses_por_query(servidor: str) -> None:
    d = json.load(pedir(f"{servidor}/api/poses?from=2&to=6"))
    assert d["from"] == 2 and d["to"] == 6


def test_cuadro_inexistente_da_404(servidor: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        pedir(f"{servidor}/api/frame/9999.jpg")
    assert exc.value.code == 404


def test_ruta_desconocida_da_404(servidor: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        pedir(f"{servidor}/api/inventada")
    assert exc.value.code == 404


def test_no_se_escapa_del_directorio_web(servidor: str) -> None:
    """Sin este chequeo, /static/../../../etc/passwd serviria cualquier archivo."""
    with pytest.raises(urllib.error.HTTPError) as exc:
        pedir(f"{servidor}/static/../../../../etc/passwd")
    assert exc.value.code in (400, 404)


def test_parametro_invalido_da_400(servidor: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        pedir(f"{servidor}/api/poses?from=abc&to=xyz")
    assert exc.value.code == 400
