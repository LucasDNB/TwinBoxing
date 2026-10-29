"""T6: el puente de identidad (01-percepcion.md §5), sobre caches sinteticos."""

import json

import pytest

from sinteticos import caja, cache, documento, guardar

T = 10
A, B = caja(300), caja(800)


def _orig(tmp_path, cuadros=None, asignaciones=None, **kw):
    cuadros = cuadros or [[(1, A), (2, B)] for _ in range(T)]
    npz = cache(tmp_path / "orig" / "cache" / "spar.pose.npz", cuadros)
    doc = documento(npz, len(cuadros), asignaciones or [(1, "A", 0, T), (2, "B", 0, T)], **kw)
    return doc, npz


def _nuevo(tmp_path, cuadros):
    return cache(tmp_path / "nuevo" / "cache" / "spar.pose.npz", cuadros, modelo="yolo26l-pose.pt")


def _abrir(npz):
    from boxtwin.core.posecache import PoseCache
    return PoseCache.open(npz)


def _roles(doc):
    return sorted((a.track_id, a.role.value, a.start_frame, a.end_frame_excl)
                  for a in doc.identity.assignments)


def test_k1_contra_si_mismo_da_100(gemelo, tmp_path):
    import puente_identidad as pi
    cuadros = [[(1, A), (2, B)] for _ in range(5)] + [[(3, A), (2, B)] for _ in range(5)]
    doc, npz = _orig(tmp_path, cuadros, [(1, "A", 0, 5), (3, "A", 5, 10), (2, "B", 0, 10)])
    k = pi.k1(doc, _abrir(npz))
    assert k["cuadros"] == 20 and k["iguales"] == 20 and k["conflictos"] == 0 and k["pasa"]


def test_caja_corrida_con_iou_bajo_queda_sin_emparejar(gemelo, tmp_path):
    import puente_identidad as pi
    doc, npz = _orig(tmp_path)
    corrida = caja(300 + 60)                                    # IoU 40/160 = 0,25
    nuevo = [[(7, corrida if f == 4 else A), (8, B)] for f in range(T)]
    _, inf, asign = pi.puente(doc, _abrir(npz), _abrir(_nuevo(tmp_path, nuevo)))
    assert 4 not in asign["A"] and set(asign["A"]) == set(range(T)) - {4}
    assert inf["emparejados"] == {"A": 9, "B": 10}
    assert inf["cuadros_con_verdad"] == {"A": 10, "B": 10}


def test_conflicto_a_b_lo_gana_el_de_mayor_iou(gemelo, tmp_path):
    import puente_identidad as pi
    cerca = (250.0, 100.0, 350.0, 400.0)
    cuadros = [[(1, cerca), (2, (270.0, 100.0, 370.0, 400.0))] for _ in range(T)]
    doc, npz = _orig(tmp_path, cuadros)
    # YOLO26 ve una sola persona, encimada con las dos cajas (IoU 0,90 con A y 0,74 con B)
    nuevo = [[(9, (255.0, 100.0, 355.0, 400.0))] for _ in range(T)]
    _, inf, asign = pi.puente(doc, _abrir(npz), _abrir(_nuevo(tmp_path, nuevo)))
    assert set(asign["A"]) == set(range(T)) and not asign["B"]
    assert inf["conflictos"] == T


def test_un_track_mixto_da_dos_asignaciones(gemelo, tmp_path):
    import puente_identidad as pi
    doc, npz = _orig(tmp_path)
    nuevo = [[(7, A), (8, B)] if f < 5 else [(8, A), (7, B)] for f in range(T)]
    doc2, inf, _ = pi.puente(doc, _abrir(npz), _abrir(_nuevo(tmp_path, nuevo)))
    assert _roles(doc2) == [(7, "fighter_A", 0, 5), (7, "fighter_B", 5, 10),
                            (8, "fighter_A", 5, 10), (8, "fighter_B", 0, 5)]
    assert all(a.origin.op.value == "puente" and a.origin.annotator == "puente-e8"
               for a in doc2.identity.assignments)
    assert inf["tracks_nuevos_usados"] == 2


def test_interpolacion_transportada_y_no_transportada(gemelo, tmp_path):
    import puente_identidad as pi
    # A se pierde en 4..6 en el original y Lucas acepto la interpolacion de 1 a 3
    cuadros = [[(1 if f < 4 else 3, A), (2, B)] if not 4 <= f < 7 else [(2, B)] for f in range(T)]
    doc, npz = _orig(tmp_path, cuadros, [(1, "A", 0, 4), (3, "A", 7, 10), (2, "B", 0, 10)],
                     interps=[("in_0001", "A", 1, 3, 4, 7)])
    # YOLO26 tambien lo pierde en el hueco y lo tiene en 3 y en 7: se transporta
    nuevo = [[(11, A), (12, B)] if not 4 <= f < 7 else [(12, B)] for f in range(T)]
    doc2, inf, _ = pi.puente(doc, _abrir(npz), _abrir(_nuevo(tmp_path, nuevo)))
    (it,) = doc2.identity.interpolations
    assert (it.from_track_id, it.to_track_id, it.gap_start_frame, it.gap_end_frame_excl) == (11, 11, 4, 7)
    assert it.accepted_by == "puente-e8 (de in_0001)" and it.method.value == "linear"
    assert inf["interpolaciones"] == {"transportadas": 1, "no_transportadas": 0}
    # en el nuevo cache A no esta emparejado en el cuadro 7: no se transporta ni se inventa otra
    nuevo = [[(11, A), (12, B)] if f < 4 or f > 7 else [(12, B)] for f in range(T)]
    doc3, inf, _ = pi.puente(doc, _abrir(npz), _abrir(_nuevo(tmp_path / "otra", nuevo)))
    assert doc3.identity.interpolations == []
    assert inf["interpolaciones"] == {"transportadas": 0, "no_transportadas": 1}


def test_todo_lo_demas_se_copia_y_pose_apunta_al_cache_nuevo(gemelo, tmp_path):
    import puente_identidad as pi
    from boxtwin.core.annotations import load
    doc, npz = _orig(tmp_path, eventos=3)
    npz_n = _nuevo(tmp_path, [[(7, A), (8, B)] for _ in range(T)])
    doc2, inf, _ = pi.puente(doc, _abrir(npz), _abrir(npz_n))
    destino = gemelo / "raiz-y26" / "anotacion" / "annotations" / "spar.annot.json"
    pi.escribir(doc2, inf, destino, npz_n)
    cargado, _ = load(destino)
    assert cargado.events == doc.events and cargado.fighters == doc.fighters
    assert cargado.video == doc.video and cargado.counters == doc.counters
    assert cargado.pose.npz_path == str(npz_n)
    assert cargado.pose.keypoint_sources == {"0-16": "yolo26l-pose"}
    assert len(cargado.pose.meta_sha256) == 64 and cargado.pose.meta_sha256 != "0" * 64
    informe = json.loads((destino.parents[1] / "puente.json").read_text())
    assert informe["emparejados"] == {"A": 10, "B": 10} and informe["iou_mediana"] == 1.0
    assert informe["puente_debil"] is False


def test_se_niega_a_escribir_dentro_del_repo(gemelo, tmp_path):
    import guardas
    import puente_identidad as pi
    doc, npz = _orig(tmp_path)
    npz_n = _nuevo(tmp_path, [[(7, A), (8, B)] for _ in range(T)])
    doc2, inf, _ = pi.puente(doc, _abrir(npz), _abrir(npz_n))
    with pytest.raises(guardas.GuardaError):
        pi.escribir(doc2, inf, guardas.REPO / "anotacion" / "annotations" / "x.annot.json", npz_n)
    with pytest.raises(guardas.GuardaError):                    # fuera del gemelo, tampoco
        pi.escribir(doc2, inf, tmp_path / "suelto" / "x.annot.json", npz_n)


def test_no_pisa_una_anotacion_que_no_escribio_el_puente(gemelo, tmp_path):
    import guardas
    import puente_identidad as pi
    doc, npz = _orig(tmp_path)
    npz_n = _nuevo(tmp_path, [[(7, A), (8, B)] for _ in range(T)])
    destino = gemelo / "raiz-y26" / "anotacion" / "annotations" / "spar.annot.json"
    guardar(doc, destino)                                      # una anotacion de Lucas
    doc2, inf, _ = pi.puente(doc, _abrir(npz), _abrir(npz_n))
    with pytest.raises(guardas.GuardaError):
        pi.escribir(doc2, inf, destino, npz_n)


def test_con_manual_tracks_se_frena(gemelo, tmp_path):
    import puente_identidad as pi
    from boxtwin.core.schema import ManualBox, ManualTrack
    from boxtwin.core.types import TrackRole
    from sinteticos import AHORA
    doc, npz = _orig(tmp_path)
    doc.identity.manual_tracks.append(ManualTrack(
        track_id=-1, role=TrackRole.A, boxes=[ManualBox(frame=2, xyxy=A)], annotator="lucasb",
        created_at=AHORA))
    with pytest.raises(pi.PuenteError):
        pi.puente(doc, _abrir(npz), _abrir(npz))


def test_k4_marca_el_puente_debil(gemelo, tmp_path):
    """K4: mediana del IoU de los emparejados de la fuente, A y B juntos, debajo de 0,70."""
    import puente_identidad as pi
    doc, npz = _orig(tmp_path)
    # cajas que se emparejan pero dibujadas distinto: IoU 80/120 = 0,667 y 75/125 = 0,60
    corrida = lambda f, cx: caja(cx + (20 if f % 2 else 25))
    nuevo = [[(7, corrida(f, 300)), (8, corrida(f, 800))] for f in range(T)]
    _, inf, _ = pi.puente(doc, _abrir(npz), _abrir(_nuevo(tmp_path, nuevo)))
    assert inf["iou_mediana"] == pytest.approx((0.6 + 2 / 3) / 2, abs=1e-3)
    assert inf["puente_debil"] is True
    # solo A corrida: la mediana de la fuente queda en (0,667 + 1) / 2 y no es debil
    nuevo = [[(7, corrida(f, 300)), (8, B)] for f in range(T)]
    _, inf, _ = pi.puente(doc, _abrir(npz), _abrir(_nuevo(tmp_path / "otra", nuevo)))
    assert inf["puente_debil"] is False and inf["iou_p10"] == pytest.approx(0.6, abs=1e-3)


def test_detecciones_de_yolo26_sin_verdad_se_cuentan_aparte(gemelo, tmp_path):
    import puente_identidad as pi
    cuadros = [[(1, A), (2, B)] if f < 8 else [(1, A)] for f in range(T)]
    doc, npz = _orig(tmp_path, cuadros)
    # YOLO26 ve a B en 8 y 9 donde v8 no lo veia: no hay verdad para premiarlo
    nuevo = [[(7, A), (8, B)] for _ in range(T)]
    _, inf, _ = pi.puente(doc, _abrir(npz), _abrir(_nuevo(tmp_path, nuevo)))
    assert inf["cuadros_con_deteccion_sin_verdad"] == {"A": 0, "B": 2}
