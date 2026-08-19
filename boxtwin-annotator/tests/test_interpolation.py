"""
Deteccion de huecos e interpolacion.

Lo que se fija es que la propuesta sea conservadora y ordenada por confianza. Unir dos
tracks es afirmar que son la misma persona, y en un clinch esa afirmacion puede ser falsa
justamente cuando mas se parecen las cajas.
"""

from __future__ import annotations

import numpy as np
import pytest

from boxtwin.core.interpolation import (
    GapCandidate,
    detectar_huecos,
    detectar_huecos_internos,
    interpolar_tramo,
    iou,
    tramos,
)
from boxtwin.core.posecache import N_KEYPOINTS, FrameStatus, PoseArrays, PoseCache
from boxtwin.core.types import TrackRole


def cache_con(por_frame: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
              n_frames: int) -> PoseCache:
    """Cache sintetico: por cuadro, lista de (track_id, bbox)."""
    conteos = [len(por_frame.get(f, [])) for f in range(n_frames)]
    total = sum(conteos)
    frame_index = np.zeros(n_frames + 1, np.int64)
    frame_index[1:] = np.cumsum(conteos)

    ids, cajas = [], []
    for f in range(n_frames):
        for tid, caja in por_frame.get(f, []):
            ids.append(tid)
            cajas.append(caja)

    kp = np.zeros((total, N_KEYPOINTS, 2), np.float32)
    for i, caja in enumerate(cajas):
        # keypoints en el centro de la caja, para que la interpolacion sea comprobable
        kp[i, :, 0] = (caja[0] + caja[2]) / 2
        kp[i, :, 1] = (caja[1] + caja[3]) / 2

    return PoseCache(
        PoseArrays(
            frame_index=frame_index,
            frame_status=np.full(n_frames, FrameStatus.OK, np.uint8),
            track_id=np.asarray(ids or [], np.int32).reshape(total),
            bbox=np.asarray(cajas or [], np.float32).reshape(total, 4),
            det_conf=np.full(total, 0.9, np.float32),
            keypoints=kp,
            kp_score=np.full((total, N_KEYPOINTS), 0.8, np.float32),
        ),
        meta={},
    )


CAJA = (100.0, 100.0, 200.0, 300.0)


# -- IoU -------------------------------------------------------------------


def test_iou_identica() -> None:
    a = np.array(CAJA, np.float32)
    assert iou(a, a) == pytest.approx(1.0)


def test_iou_disjuntas() -> None:
    a = np.array([0.0, 0.0, 10.0, 10.0], np.float32)
    b = np.array([50.0, 50.0, 60.0, 60.0], np.float32)
    assert iou(a, b) == 0.0


def test_iou_parcial() -> None:
    a = np.array([0.0, 0.0, 10.0, 10.0], np.float32)
    b = np.array([5.0, 0.0, 15.0, 10.0], np.float32)
    # interseccion 50, union 150
    assert iou(a, b) == pytest.approx(50 / 150)


def test_iou_caja_degenerada() -> None:
    a = np.array([0.0, 0.0, 0.0, 0.0], np.float32)
    b = np.array([0.0, 0.0, 10.0, 10.0], np.float32)
    assert iou(a, b) == 0.0


# -- deteccion -------------------------------------------------------------


def test_detecta_hueco_con_cajas_compatibles() -> None:
    """El track 1 termina en 10 y el 2 arranca en 14, casi en el mismo lugar."""
    por_frame = {f: [(1, CAJA)] for f in range(0, 11)}
    por_frame.update({f: [(2, CAJA)] for f in range(14, 20)})
    cands = detectar_huecos(cache_con(por_frame, 20), max_gap=5, min_iou=0.5)
    assert len(cands) == 1
    c = cands[0]
    assert (c.from_track_id, c.to_track_id) == (1, 2)
    assert (c.last_frame, c.first_frame) == (10, 14)
    assert c.gap_len == 3
    assert c.gap_start == 11 and c.gap_end_excl == 14
    assert not c.es_continuacion


def test_no_detecta_si_el_hueco_es_muy_largo() -> None:
    por_frame = {f: [(1, CAJA)] for f in range(0, 11)}
    por_frame.update({f: [(2, CAJA)] for f in range(30, 40)})
    assert detectar_huecos(cache_con(por_frame, 40), max_gap=5) == []


def test_no_detecta_si_las_cajas_no_se_parecen() -> None:
    """
    Es la salvaguarda que importa: dos personas distintas no se unen aunque el tracker
    pierda una justo cuando aparece la otra.
    """
    lejos = (800.0, 100.0, 900.0, 300.0)
    por_frame = {f: [(1, CAJA)] for f in range(0, 11)}
    por_frame.update({f: [(2, lejos)] for f in range(13, 20)})
    assert detectar_huecos(cache_con(por_frame, 20), max_gap=5, min_iou=0.5) == []


def test_continuacion_sin_hueco() -> None:
    """Cambio de id sin cuadros faltantes: no hay nada que interpolar."""
    por_frame = {f: [(1, CAJA)] for f in range(0, 11)}
    por_frame.update({f: [(2, CAJA)] for f in range(11, 20)})
    cands = detectar_huecos(cache_con(por_frame, 20), max_gap=5)
    assert len(cands) == 1
    assert cands[0].es_continuacion
    assert cands[0].gap_len == 0


def test_ordenados_por_confianza() -> None:
    """El mas parecido primero: es el que se confirma sin pensar, los dudosos van al final."""
    casi = (105.0, 100.0, 205.0, 300.0)
    por_frame: dict = {}
    for f in range(0, 6):
        por_frame[f] = [(1, CAJA), (10, CAJA)]
    for f in range(8, 12):
        por_frame[f] = [(2, CAJA), (11, casi)]
    cands = detectar_huecos(cache_con(por_frame, 12), max_gap=5, min_iou=0.3)
    assert cands == sorted(cands, key=lambda c: -c.iou)
    assert cands[0].iou >= cands[-1].iou


def test_un_track_solo_no_produce_candidatos() -> None:
    por_frame = {f: [(1, CAJA)] for f in range(0, 20)}
    assert detectar_huecos(cache_con(por_frame, 20)) == []


# -- interpolacion ---------------------------------------------------------


def test_interpola_linealmente() -> None:
    izq = (0.0, 0.0, 100.0, 100.0)
    der = (100.0, 0.0, 200.0, 100.0)
    por_frame = {0: [(1, izq)], 4: [(2, der)]}
    cache = cache_con(por_frame, 5)
    cand = GapCandidate(from_track_id=1, to_track_id=2, last_frame=0, first_frame=4, iou=0.6)

    tramo = interpolar_tramo(cache, cand)
    assert [f for f, *_ in tramo] == [1, 2, 3]
    # A un cuarto, a la mitad y a tres cuartos del camino.
    assert tramo[0][1][0] == pytest.approx(25.0)
    assert tramo[1][1][0] == pytest.approx(50.0)
    assert tramo[2][1][0] == pytest.approx(75.0)
    # Los keypoints acompanan: estaban en el centro de cada caja.
    assert tramo[1][2][0][0] == pytest.approx(100.0)


def test_el_score_interpolado_es_el_minimo_de_sus_insumos() -> None:
    """
    Un punto inventado no puede tener mas confianza que los datos con que se invento.

    Los dos extremos llevan scores distintos a proposito: con el mismo valor en los dos, el
    test pasaria aunque el codigo tomara el maximo o el promedio.
    """
    por_frame = {0: [(1, CAJA)], 3: [(2, CAJA)]}
    cache = cache_con(por_frame, 4)
    cache.detections(0).kp_score[0][:] = 0.9
    cache.detections(3).kp_score[0][:] = 0.4

    cand = GapCandidate(from_track_id=1, to_track_id=2, last_frame=0, first_frame=3, iou=0.9)
    for _, _, _, score in interpolar_tramo(cache, cand):
        np.testing.assert_allclose(score, 0.4, rtol=1e-6)


def test_sin_hueco_no_interpola_nada() -> None:
    por_frame = {0: [(1, CAJA)], 1: [(2, CAJA)]}
    cache = cache_con(por_frame, 2)
    cand = GapCandidate(from_track_id=1, to_track_id=2, last_frame=0, first_frame=1, iou=0.9)
    assert interpolar_tramo(cache, cand) == []


def test_falla_si_los_tracks_no_estan_donde_dice() -> None:
    por_frame = {0: [(1, CAJA)], 4: [(2, CAJA)]}
    cache = cache_con(por_frame, 5)
    malo = GapCandidate(from_track_id=99, to_track_id=2, last_frame=0, first_frame=4, iou=0.9)
    with pytest.raises(ValueError):
        interpolar_tramo(cache, malo)


# -- tramos contiguos ------------------------------------------------------


def test_tramos_parte_un_track_agujereado() -> None:
    """
    Un track no es un intervalo. Con track_buffer alto el mismo id aparece, desaparece y
    vuelve, y mirar solo el primer y ultimo cuadro finge una continuidad que no existe.
    """
    por_frame = {f: [(1, CAJA)] for f in list(range(0, 5)) + list(range(8, 12))}
    assert tramos(cache_con(por_frame, 12)) == {1: [(0, 4), (8, 11)]}


def test_tramos_de_un_track_continuo_es_uno_solo() -> None:
    por_frame = {f: [(1, CAJA)] for f in range(0, 10)}
    assert tramos(cache_con(por_frame, 10)) == {1: [(0, 9)]}


# -- huecos internos -------------------------------------------------------


def test_detecta_hueco_interno_del_mismo_track() -> None:
    por_frame = {f: [(1, CAJA)] for f in list(range(0, 6)) + list(range(9, 15))}
    cands = detectar_huecos_internos(cache_con(por_frame, 15))
    assert len(cands) == 1
    c = cands[0]
    assert (c.from_track_id, c.to_track_id) == (1, 1)
    assert c.es_mismo_track
    assert (c.last_frame, c.first_frame) == (5, 9)
    assert c.gap_len == 3


def test_hueco_interno_respeta_el_tope() -> None:
    por_frame = {f: [(1, CAJA)] for f in list(range(0, 6)) + list(range(40, 45))}
    assert detectar_huecos_internos(cache_con(por_frame, 45), max_gap=20) == []
    assert len(detectar_huecos_internos(cache_con(por_frame, 45), max_gap=40)) == 1


def test_hueco_interno_no_mira_otros_tracks() -> None:
    """Un track que termina y otro que empieza es una union, no un hueco interno."""
    por_frame = {f: [(1, CAJA)] for f in range(0, 6)}
    por_frame.update({f: [(2, CAJA)] for f in range(9, 15)})
    assert detectar_huecos_internos(cache_con(por_frame, 15)) == []


def test_uniones_no_devuelven_huecos_internos() -> None:
    """La simetrica: el detector de uniones nunca propone unir un track consigo mismo."""
    por_frame = {f: [(1, CAJA)] for f in list(range(0, 6)) + list(range(9, 15))}
    cands = detectar_huecos(cache_con(por_frame, 15), max_gap=20)
    assert all(not c.es_mismo_track for c in cands)


def test_hueco_interno_con_salto_absurdo_se_descarta() -> None:
    lejos = (800.0, 100.0, 900.0, 300.0)
    por_frame = {f: [(1, CAJA)] for f in range(0, 6)}
    por_frame.update({f: [(1, lejos)] for f in range(9, 15)})
    assert detectar_huecos_internos(cache_con(por_frame, 15), min_iou=0.2) == []


def test_interpola_un_hueco_interno() -> None:
    """from == to tiene que funcionar en la interpolacion, no solo en la deteccion."""
    por_frame = {f: [(1, CAJA)] for f in list(range(0, 6)) + list(range(9, 15))}
    cache = cache_con(por_frame, 15)
    (c,) = detectar_huecos_internos(cache)
    filas = interpolar_tramo(cache, c)
    assert [f for f, *_ in filas] == [6, 7, 8]


# -- filtro por rol --------------------------------------------------------


def test_no_propone_unir_dos_peleadores_distintos() -> None:
    """
    En un clinch las cajas de los dos boxeadores se superponen y la geometria no distingue
    una union buena de una que fusionaria a las dos personas. Lo que las distingue es que el
    anotador ya dijo de quien es cada track.
    """
    por_frame = {f: [(1, CAJA)] for f in range(0, 11)}
    por_frame.update({f: [(2, CAJA)] for f in range(14, 20)})
    cache = cache_con(por_frame, 20)
    roles = {1: TrackRole.A, 2: TrackRole.B}

    assert len(detectar_huecos(cache, max_gap=5)) == 1
    assert detectar_huecos(cache, max_gap=5, rol_en=lambda t, f: roles[t]) == []


def test_si_coinciden_en_el_cuadro_de_union_si_se_propone() -> None:
    """
    El rol se consulta en el cuadro donde se unirian, no en abstracto. Un track puede ser
    fighter_B al principio y fighter_A despues, que es lo que deja un swap; colapsarlo a un
    conjunto de roles hace que el filtro no filtre nada. Caso real de Sparring.mp4: el track
    3 es B hasta el cuadro 614 y A despues, y se une con el 12 en el 829 siendo los dos A.
    """
    por_frame = {f: [(1, CAJA)] for f in range(0, 11)}
    por_frame.update({f: [(2, CAJA)] for f in range(14, 20)})
    cache = cache_con(por_frame, 20)

    def rol_en(track: int, frame: int) -> TrackRole:
        if track == 1:
            return TrackRole.B if frame < 5 else TrackRole.A
        return TrackRole.A

    assert len(detectar_huecos(cache, max_gap=5, rol_en=rol_en)) == 1


def test_un_track_sin_rol_no_bloquea_la_propuesta() -> None:
    """Sin assignment no hay afirmacion que contradecir, asi que se propone igual."""
    por_frame = {f: [(1, CAJA)] for f in range(0, 11)}
    por_frame.update({f: [(2, CAJA)] for f in range(14, 20)})
    cache = cache_con(por_frame, 20)
    roles = {1: TrackRole.A, 2: None}
    assert len(detectar_huecos(cache, max_gap=5, rol_en=lambda t, f: roles[t])) == 1


def test_ignore_no_cuenta_como_peleador_para_el_filtro() -> None:
    por_frame = {f: [(1, CAJA)] for f in range(0, 11)}
    por_frame.update({f: [(2, CAJA)] for f in range(14, 20)})
    cache = cache_con(por_frame, 20)
    roles = {1: TrackRole.A, 2: TrackRole.IGNORE}
    assert len(detectar_huecos(cache, max_gap=5, rol_en=lambda t, f: roles[t])) == 1
