import json
import zipfile

import pytest

from boxtwin_guantes import roboflow as rf


# -- clip de origen ---------------------------------------------------------


def test_dos_cuadros_del_mismo_video_dan_el_mismo_clip():
    a = rf.clip_de_origen("Trimed Box Match _mp4-2092.jpg")
    b = rf.clip_de_origen("Trimed Box Match _mp4-2167.jpg")
    assert a == b == "trimed box match"


def test_el_nombre_renombrado_por_el_export_agrupa_igual():
    # El export mete el hash entre el numero de cuadro y la extension. Sin sacarlo, cada
    # cuadro quedaba en su propio grupo y 853 cuadros vecinos de la misma pelea se
    # repartian entre train, valid y test.
    a = rf.clip_de_origen("Trimed-Box-Match-_mp4-0020_jpg.rf.15c99360dd7b8b256f829593e2dd8c6d.jpg")
    b = rf.clip_de_origen("Trimed-Box-Match-_mp4-0407_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg")
    c = rf.clip_de_origen("Trimed Box Match _mp4-2092.jpg")
    assert a == b == c == "trimed box match"


def test_videos_distintos_dan_clips_distintos():
    assert rf.clip_de_origen("fight2_mp4-14.jpg") != rf.clip_de_origen("fight3_mp4-14.jpg")


def test_un_nombre_sin_forma_de_cuadro_queda_en_su_propio_grupo():
    # El lado conservador: agrupar de menos reparte de mas, pero nunca mezcla dos clips.
    assert rf.clip_de_origen("foto_suelta.jpg") == "foto suelta.jpg"


# -- cruce de nombres entre el export y el catalogo -------------------------


def test_el_nombre_del_export_cruza_con_el_del_catalogo():
    # El export renombra y le pega un hash; las dos formas tienen que caer en la misma clave.
    a = rf.clave_de_nombre("Trimed-Box-Match-_mp4-0431_jpg.rf.5a1b2c3d4e5f.jpg")
    b = rf.clave_de_nombre("Trimed Box Match _mp4-0431.jpg")
    assert a == b == "trimedboxmatchmp40431"


def test_cuadros_distintos_del_mismo_clip_no_cruzan_entre_si():
    assert rf.clave_de_nombre("a_mp4-0431.jpg") != rf.clave_de_nombre("a_mp4-0432.jpg")


# -- reparto ----------------------------------------------------------------


def test_ningun_clip_cae_en_dos_splits():
    # Es la propiedad que justifica todo el modulo: los cuadros vecinos de un video son
    # casi identicos, y repartirlos al azar inflaria la validacion.
    nombres = {f"i{n}": f"pelea{n % 7}_mp4-{n}.jpg" for n in range(200)}
    splits = rf.repartir(nombres)
    por_clip = {}
    for ident, nombre in nombres.items():
        por_clip.setdefault(rf.clip_de_origen(nombre), set()).add(splits[ident])
    assert all(len(s) == 1 for s in por_clip.values())


def test_el_reparto_es_determinista_entre_corridas():
    nombres = {f"i{n}": f"pelea{n % 5}_mp4-{n}.jpg" for n in range(50)}
    assert rf.repartir(nombres) == rf.repartir(dict(reversed(list(nombres.items()))))


def test_el_reparto_cubre_todas_las_imagenes():
    nombres = {f"i{n}": f"pelea{n % 5}_mp4-{n}.jpg" for n in range(50)}
    splits = rf.repartir(nombres)
    assert set(splits) == set(nombres)
    assert set(splits.values()) <= {"train", "valid", "test"}


# -- catalogo ---------------------------------------------------------------


def test_una_imagen_sin_dimensiones_no_se_construye():
    # El tamano original es lo unico que el catalogo aporta; sin el, el registro no sirve.
    assert rf._imagen_de({"id": "x", "url": "u", "width": 0, "height": 10}) is None


def test_el_catalogo_conserva_el_tamano_original():
    im = rf._imagen_de({"id": "x", "url": "u", "width": 1920, "height": 1080, "name": "a.jpg"})
    assert (im.ancho, im.alto, im.nombre) == (1920, 1080, "a.jpg")


# -- labels -----------------------------------------------------------------


def test_los_labels_se_leen_en_formato_yolo(tmp_path):
    t = tmp_path / "a.txt"
    t.write_text("0 0.5 0.25 0.1 0.2\n0 0.1 0.1 0.05 0.05\n")
    assert rf._leer_labels(t) == [(0, 0.5, 0.25, 0.1, 0.2), (0, 0.1, 0.1, 0.05, 0.05)]


def test_un_poligono_se_reduce_a_su_caja_envolvente(tmp_path):
    # La v3 anota los cuadros de video con poligonos y el resto con cajas. Un lector que
    # exija cinco campos tira el 100% del material de dominio en silencio.
    t = tmp_path / "a.txt"
    t.write_text("0 0.2 0.4 0.6 0.4 0.6 0.8 0.2 0.8\n")
    assert rf._leer_labels(t) == [(0, 0.4, 0.6000000000000001, 0.39999999999999997, 0.4)]


def test_cajas_y_poligonos_conviven_en_el_mismo_archivo(tmp_path):
    t = tmp_path / "a.txt"
    t.write_text("0 0.5 0.5 0.1 0.1\n0 0.2 0.4 0.6 0.4 0.6 0.8 0.2 0.8\n")
    assert len(rf._leer_labels(t)) == 2


@pytest.mark.parametrize("linea", [
    "0 0.1 0.2 0.3 0.4 0.5",        # cantidad impar de coordenadas
    "0 0.1 0.2 0.1 0.2 0.1 0.2",    # poligono degenerado: todos los vertices iguales
    "0 0.1 0.2",                     # muy corto
])
def test_un_poligono_malformado_se_descarta(tmp_path, linea):
    t = tmp_path / "a.txt"
    t.write_text(linea + "\n")
    assert rf._leer_labels(t) == []


def test_una_linea_corrupta_se_saltea_y_el_resto_se_lee(tmp_path):
    t = tmp_path / "a.txt"
    t.write_text("0 0.5 0.25 0.1 0.2\nbasura x y z w\n0 1 2\n")
    assert rf._leer_labels(t) == [(0, 0.5, 0.25, 0.1, 0.2)]


def test_un_txt_que_no_existe_da_lista_vacia(tmp_path):
    assert rf._leer_labels(tmp_path / "no-esta.txt") == []


# -- desestirado ------------------------------------------------------------


def test_desestirar_devuelve_la_relacion_de_aspecto_original(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    origen = tmp_path / "in.jpg"
    cv2.imwrite(str(origen), np.zeros((640, 640, 3), dtype=np.uint8))
    destino = tmp_path / "out.jpg"

    assert rf._desestirar(origen, destino, 1920, 1080) is True
    alto, ancho = cv2.imread(str(destino)).shape[:2]
    # 16:9, con el lado largo conservado en 640: reescalar hacia arriba no inventa detalle.
    assert (ancho, alto) == (640, 360)


def test_desestirar_avisa_cuando_no_puede_leer(tmp_path):
    pytest.importorskip("cv2")
    roto = tmp_path / "roto.jpg"
    roto.write_bytes(b"no soy un jpeg")
    assert rf._desestirar(roto, tmp_path / "out.jpg", 1920, 1080) is False


# -- bajada completa, con la red simulada -----------------------------------


def _export_falso(tmp_path, nombres, ancho=640, alto=640):
    """Arma un zip con la forma de un export YOLOv8 de Roboflow."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    raiz = tmp_path / "export"
    for split in ("train", "valid", "test"):
        (raiz / split / "images").mkdir(parents=True, exist_ok=True)
        (raiz / split / "labels").mkdir(parents=True, exist_ok=True)
    (raiz / "data.yaml").write_text("names: ['Boxing-Glove']\nnc: 1\n")
    for i, n in enumerate(nombres):
        split = ("train", "valid", "test")[i % 3]
        stem = f"{n.replace(' ', '-').replace('.jpg', '')}_jpg.rf.{i:032x}"
        cv2.imwrite(str(raiz / split / "images" / f"{stem}.jpg"),
                    np.zeros((alto, ancho, 3), dtype=np.uint8))
        (raiz / split / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    z = tmp_path / "export.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in raiz.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(raiz))
    return z


def _simular_red(monkeypatch, zip_falso, catalogo=None):
    monkeypatch.setattr(rf, "link_de_export", lambda *a, **k: ("http://falso", {"imagenes": 9}))
    monkeypatch.setattr(rf, "_catalogo_por_clave", lambda key: catalogo or {})

    def falsa_bajada(url, destino, timeout=600):
        destino.write_bytes(zip_falso.read_bytes())
        return destino.stat().st_size

    monkeypatch.setattr(rf, "_bajar_archivo", falsa_bajada)


def test_la_bajada_deja_el_layout_de_ultralytics(tmp_path, monkeypatch):
    nombres = [f"pelea{n % 4}_mp4-{n}.jpg" for n in range(12)]
    z = _export_falso(tmp_path, nombres)
    _simular_red(monkeypatch, z)

    salida = tmp_path / "salida"
    m = rf.descargar_dataset("clave-falsa", salida, version=1)

    assert (salida / "data.yaml").exists()
    assert json.loads((salida / "manifiesto.json").read_text())["kind"] == "boxtwin.guantes.dataset"
    assert m["imagenes"] == 12
    assert m["cajas"] == 12
    assert m["clips_en_mas_de_un_split"] == []
    assert sum(c["imagenes"] for c in m["conteos"].values()) == 12


def test_el_reparto_del_export_se_rehace_por_clip(tmp_path, monkeypatch):
    # El export trae sus propios splits; los ignoramos porque no sabemos si respetan la
    # procedencia de los videos.
    nombres = [f"pelea{n % 3}_mp4-{n}.jpg" for n in range(12)]
    z = _export_falso(tmp_path, nombres)
    _simular_red(monkeypatch, z)
    salida = tmp_path / "salida"
    rf.descargar_dataset("clave-falsa", salida, version=1)

    por_clip = {}
    for split in ("train", "valid", "test"):
        for img in (salida / "images" / split).iterdir():
            por_clip.setdefault(rf.clip_de_origen(img.stem), set()).add(split)
    assert all(len(s) == 1 for s in por_clip.values())


def test_no_vuelve_a_bajar_el_zip_si_ya_esta(tmp_path, monkeypatch):
    nombres = [f"pelea{n % 3}_mp4-{n}.jpg" for n in range(6)]
    z = _export_falso(tmp_path, nombres)
    veces = []
    _simular_red(monkeypatch, z)
    original = rf._bajar_archivo

    def contando(url, destino, timeout=600):
        veces.append(url)
        return original(url, destino, timeout)

    monkeypatch.setattr(rf, "_bajar_archivo", contando)
    salida = tmp_path / "salida"
    rf.descargar_dataset("clave-falsa", salida, version=1)
    rf.descargar_dataset("clave-falsa", salida, version=1)
    assert len(veces) == 1, "el zip no se vuelve a bajar en la segunda corrida"


def test_un_zip_bajado_a_mano_no_le_pide_nada_a_la_api(tmp_path, monkeypatch):
    nombres = [f"pelea{n % 3}_mp4-{n}.jpg" for n in range(9)]
    z = _export_falso(tmp_path, nombres)

    def explota(*a, **k):
        raise AssertionError("no tendria que hablar con la API")

    monkeypatch.setattr(rf, "link_de_export", explota)
    monkeypatch.setattr(rf, "_bajar_archivo", explota)
    monkeypatch.setattr(rf, "_catalogo_por_clave", explota)

    m = rf.descargar_dataset(None, tmp_path / "salida", version=1, zip_local=z)
    assert m["imagenes"] == 9
    # Sin catalogo no se puede deshacer el estirado, y no se inventa: quedan como vinieron
    # y el manifiesto lo declara.
    assert m["sin_cruce_con_catalogo"] == 9


def test_un_zip_que_no_existe_falla_claro(tmp_path):
    with pytest.raises(rf.ErrorRoboflow, match="no existe el zip"):
        rf.descargar_dataset(None, tmp_path / "s", zip_local=tmp_path / "no-esta.zip")


def test_sin_catalogo_la_imagen_no_se_deforma(tmp_path, monkeypatch):
    # Asumir 16:9 para una foto de producto cuadrada la deformaria. Sin el tamano original
    # se copia tal cual, que es el unico lado que no inventa.
    cv2 = pytest.importorskip("cv2")
    z = _export_falso(tmp_path, ["producto_mp4-1.jpg"], ancho=640, alto=640)
    _simular_red(monkeypatch, z, catalogo={})
    salida = tmp_path / "salida"
    rf.descargar_dataset(None, salida, version=1, zip_local=z)
    img = next(f for s in ("train", "valid", "test")
               for f in (salida / "images" / s).iterdir())
    assert cv2.imread(str(img)).shape[:2] == (640, 640)


def test_con_catalogo_se_le_devuelve_el_aspecto(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    z = _export_falso(tmp_path, ["video_mp4-1.jpg"], ancho=640, alto=640)
    cat = {rf.clave_de_nombre("video_mp4-1.jpg"): rf.ImagenRemota(
        id="x", nombre="video_mp4-1.jpg", ancho=1920, alto=1080, url="u")}
    _simular_red(monkeypatch, z, catalogo=cat)
    salida = tmp_path / "salida"
    # Con clave el catalogo si se consulta, aun usando un zip local.
    rf.descargar_dataset("clave-falsa", salida, version=1, zip_local=z)
    img = next(f for s in ("train", "valid", "test")
               for f in (salida / "images" / s).iterdir())
    assert cv2.imread(str(img)).shape[:2] == (360, 640)


def test_un_export_vacio_falla_claro(tmp_path, monkeypatch):
    raiz = tmp_path / "vacio"
    raiz.mkdir()
    z = tmp_path / "vacio.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("data.yaml", "names: ['Boxing-Glove']\n")
    _simular_red(monkeypatch, z)
    with pytest.raises(rf.ErrorRoboflow, match="no trajo imagenes"):
        rf.descargar_dataset("clave-falsa", tmp_path / "salida", version=1)


# -- clave ------------------------------------------------------------------


def test_sin_variable_de_entorno_el_error_dice_que_hacer(monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(rf.ErrorRoboflow, match="ROBOFLOW_API_KEY"):
        rf.clave_api()
