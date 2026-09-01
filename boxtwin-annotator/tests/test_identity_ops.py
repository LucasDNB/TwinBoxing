"""
Operaciones de identidad.

Es la parte del sistema donde un error no se ve. Si un rol queda mal, el esqueleto sigue
dibujandose sobre un cuerpo y el overlay se ve perfecto; lo unico que cambia es a que
peleador se le atribuyen los keypoints en el export. Por eso los tests miran el resultado
de la resolucion y no solo la forma de los registros.
"""

from __future__ import annotations

import numpy as np
import pytest

from boxtwin.core.identity import IdentityResolver
from boxtwin.core.identity_ops import (
    AcceptJoin,
    AssignRole,
    FillInternalGaps,
    MarkUnreliable,
    Reseed,
    SwapFromFrame,
    boundary_after,
)
from boxtwin.core.interpolation import GapCandidate, detectar_huecos_internos
from boxtwin.core.posecache import N_KEYPOINTS, FrameStatus, PoseArrays, PoseCache
from boxtwin.core.schema import AnnotationDoc
from boxtwin.core.types import FighterId, TrackRole, UnreliableReason
from boxtwin.core.undo import UndoStack

CAJA = (100.0, 100.0, 200.0, 300.0)


def cache_con(por_frame: dict, n_frames: int) -> PoseCache:
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


@pytest.fixture
def doc(doc_min: AnnotationDoc) -> AnnotationDoc:
    doc_min.video.total_frames = 1000
    return doc_min


@pytest.fixture
def pila(doc: AnnotationDoc) -> UndoStack:
    return UndoStack(doc)


def asignar(pila: UndoStack, track: int, rol: TrackRole, ini: int, fin: int) -> None:
    pila.do(AssignRole(track_id=track, role=rol, start_frame=ini, end_frame_excl=fin))


# -- asignacion ------------------------------------------------------------


def test_asignar_y_deshacer(doc: AnnotationDoc, pila: UndoStack) -> None:
    asignar(pila, 1, TrackRole.A, 0, 500)
    assert len(doc.identity.assignments) == 1
    pila.undo()
    assert doc.identity.assignments == []


def test_asignar_parte_el_intervalo_previo(doc: AnnotationDoc, pila: UndoStack) -> None:
    """
    Los intervalos de un mismo track no se pueden solapar: el resolver busca uno solo y con
    dos superpuestos el resultado dependeria del orden de la lista.
    """
    asignar(pila, 1, TrackRole.A, 0, 1000)
    asignar(pila, 1, TrackRole.B, 400, 600)

    res = IdentityResolver(doc, cache_con({}, 1000))
    assert res.role_of(399, 1) is TrackRole.A
    assert res.role_of(400, 1) is TrackRole.B
    assert res.role_of(599, 1) is TrackRole.B
    assert res.role_of(600, 1) is TrackRole.A
    # tres tramos: antes, el nuevo, y despues
    assert len(doc.identity.assignments) == 3


def test_asignar_sobre_el_borde(doc: AnnotationDoc, pila: UndoStack) -> None:
    asignar(pila, 1, TrackRole.A, 0, 500)
    asignar(pila, 1, TrackRole.B, 0, 500)
    assert len(doc.identity.assignments) == 1
    assert doc.identity.assignments[0].role is TrackRole.B


# -- frontera --------------------------------------------------------------


def test_boundary_sin_decisiones_posteriores(doc: AnnotationDoc) -> None:
    assert boundary_after(doc, 100) == doc.video.total_frames


def test_boundary_corta_en_la_proxima_decision(doc: AnnotationDoc, pila: UndoStack) -> None:
    """
    Sin este limite, corregir un intercambio en el minuto dos pisaria la correccion que ya
    se habia hecho en el minuto cinco.
    """
    asignar(pila, 2, TrackRole.A, 700, 1000)
    assert boundary_after(doc, 100) == 700


# -- intercambio -----------------------------------------------------------


def test_intercambio_desde_un_cuadro(doc: AnnotationDoc, pila: UndoStack) -> None:
    asignar(pila, 1, TrackRole.A, 0, 1000)
    asignar(pila, 2, TrackRole.B, 0, 1000)
    pila.do(SwapFromFrame(400))

    res = IdentityResolver(doc, cache_con({}, 1000))
    assert res.role_of(399, 1) is TrackRole.A
    assert res.role_of(399, 2) is TrackRole.B
    assert res.role_of(400, 1) is TrackRole.B
    assert res.role_of(400, 2) is TrackRole.A


def test_los_dos_lados_del_intercambio_comparten_op_id(doc: AnnotationDoc, pila: UndoStack) -> None:
    """En un diff se lee como un solo gesto y no como dos cambios que hay que correlacionar."""
    asignar(pila, 1, TrackRole.A, 0, 1000)
    asignar(pila, 2, TrackRole.B, 0, 1000)
    pila.do(SwapFromFrame(400))

    nuevos = [a for a in doc.identity.assignments if a.start_frame == 400]
    assert len(nuevos) == 2
    assert nuevos[0].origin.op_id == nuevos[1].origin.op_id
    assert nuevos[0].origin.op.value == "swap"


def test_intercambio_respeta_la_proxima_decision(doc: AnnotationDoc, pila: UndoStack) -> None:
    asignar(pila, 1, TrackRole.A, 0, 1000)
    asignar(pila, 2, TrackRole.B, 0, 1000)
    asignar(pila, 1, TrackRole.A, 800, 1000)  # decision posterior explicita
    pila.do(SwapFromFrame(400))

    res = IdentityResolver(doc, cache_con({}, 1000))
    assert res.role_of(400, 1) is TrackRole.B
    assert res.role_of(800, 1) is TrackRole.A  # la correccion posterior sobrevive


def test_intercambio_sin_dos_peleadores_falla(doc: AnnotationDoc, pila: UndoStack) -> None:
    asignar(pila, 1, TrackRole.A, 0, 1000)
    with pytest.raises(ValueError, match="no hay un track por peleador"):
        pila.do(SwapFromFrame(400))


def test_deshacer_el_intercambio(doc: AnnotationDoc, pila: UndoStack) -> None:
    asignar(pila, 1, TrackRole.A, 0, 1000)
    asignar(pila, 2, TrackRole.B, 0, 1000)
    antes = doc.identity.model_dump(mode="json")
    pila.do(SwapFromFrame(400))
    pila.undo()
    assert doc.identity.model_dump(mode="json") == antes


# -- re-seed ---------------------------------------------------------------


def test_reseed_sobre_un_track_existente(doc: AnnotationDoc, pila: UndoStack) -> None:
    """El caso comun: el tracker no perdio al peleador, le cambio el id."""
    pila.do(Reseed(frame=300, role=TrackRole.A, bbox=list(CAJA), track_id=7, iou=0.91))
    res = IdentityResolver(doc, cache_con({300: [(7, CAJA)]}, 1000))
    assert res.role_of(300, 7) is TrackRole.A
    assert doc.identity.manual_tracks == []
    assert doc.identity.assignments[0].origin.seed_iou == 0.91


def test_reseed_sin_track_debajo_crea_caja_manual(doc: AnnotationDoc, pila: UndoStack) -> None:
    pila.do(Reseed(frame=300, role=TrackRole.B, bbox=list(CAJA)))
    assert len(doc.identity.manual_tracks) == 1
    mt = doc.identity.manual_tracks[0]
    assert mt.track_id < 0  # negativo, no colisiona con los de BoT-SORT
    assert mt.role is TrackRole.B


def test_la_caja_manual_se_marca_no_confiable_sola(doc: AnnotationDoc, pila: UndoStack) -> None:
    """Sin keypoints no hay nada que exportar: se declara y no se descubre despues."""
    pila.do(Reseed(frame=300, role=TrackRole.B, bbox=list(CAJA)))
    assert len(doc.unreliable_segments) == 1
    seg = doc.unreliable_segments[0]
    assert seg.fighter is FighterId.B
    assert seg.reason is UnreliableReason.POSE_UNRELIABLE
    assert seg.source.value == "auto_manual_track"


def test_los_ids_manuales_no_se_repiten(doc: AnnotationDoc, pila: UndoStack) -> None:
    pila.do(Reseed(frame=300, role=TrackRole.A, bbox=list(CAJA)))
    pila.do(Reseed(frame=400, role=TrackRole.B, bbox=list(CAJA)))
    ids = [m.track_id for m in doc.identity.manual_tracks]
    assert ids == [-1, -2]


def test_deshacer_el_reseed(doc: AnnotationDoc, pila: UndoStack) -> None:
    pila.do(Reseed(frame=300, role=TrackRole.B, bbox=list(CAJA)))
    pila.undo()
    assert doc.identity.manual_tracks == []


# -- tramos no confiables --------------------------------------------------


def test_marcar_tramo(doc: AnnotationDoc, pila: UndoStack) -> None:
    pila.do(MarkUnreliable(fighter=FighterId.A, start_frame=100, end_frame_excl=200))
    res = IdentityResolver(doc, cache_con({}, 1000))
    assert res.is_reliable(150, FighterId.A) is False
    assert res.is_reliable(250, FighterId.A) is True
    assert res.is_reliable(150, FighterId.B) is True


def test_marcar_tramo_no_borra_la_pose(doc: AnnotationDoc, pila: UndoStack) -> None:
    """Excluirlo del export es politica del export, no de la lectura."""
    asignar(pila, 1, TrackRole.A, 0, 1000)
    pila.do(MarkUnreliable(fighter=FighterId.A, start_frame=100, end_frame_excl=200))
    res = IdentityResolver(doc, cache_con({150: [(1, CAJA)]}, 1000))
    pose = res.resolve_frame(150)[0]
    assert pose.role is TrackRole.A
    assert pose.reliable is False
    assert pose.keypoints is not None


def test_deshacer_el_tramo(doc: AnnotationDoc, pila: UndoStack) -> None:
    pila.do(MarkUnreliable(fighter=FighterId.A, start_frame=100, end_frame_excl=200))
    pila.undo()
    assert doc.unreliable_segments == []


# -- union de tracks -------------------------------------------------------


def test_aceptar_union_con_hueco(doc: AnnotationDoc, pila: UndoStack) -> None:
    cand = GapCandidate(from_track_id=1, to_track_id=2, last_frame=10, first_frame=14, iou=0.88)
    pila.do(AcceptJoin(candidato=cand, role=TrackRole.A))

    assert len(doc.identity.interpolations) == 1
    interp = doc.identity.interpolations[0]
    assert (interp.gap_start_frame, interp.gap_end_frame_excl, interp.gap_len) == (11, 14, 3)
    assert interp.iou_at_join == 0.88
    # y el track nuevo queda asignado
    res = IdentityResolver(doc, cache_con({}, 1000))
    assert res.role_of(14, 2) is TrackRole.A


def test_union_sin_hueco_no_declara_interpolacion(doc: AnnotationDoc, pila: UndoStack) -> None:
    """
    Declarar una interpolacion vacia seria decir que se sintetizo algo que no se sintetizo.
    """
    cand = GapCandidate(from_track_id=1, to_track_id=2, last_frame=10, first_frame=11, iou=0.95)
    pila.do(AcceptJoin(candidato=cand, role=TrackRole.A))
    assert doc.identity.interpolations == []
    res = IdentityResolver(doc, cache_con({}, 1000))
    assert res.role_of(11, 2) is TrackRole.A


def test_deshacer_la_union(doc: AnnotationDoc, pila: UndoStack) -> None:
    cand = GapCandidate(from_track_id=1, to_track_id=2, last_frame=10, first_frame=14, iou=0.88)
    pila.do(AcceptJoin(candidato=cand, role=TrackRole.A))
    pila.undo()
    assert doc.identity.interpolations == []
    assert doc.identity.assignments == []


# -- sintesis al leer ------------------------------------------------------


def test_el_resolver_sintetiza_el_hueco(doc: AnnotationDoc, pila: UndoStack) -> None:
    """
    Los cuadros del hueco se inventan al leer y nunca se escriben en el npz: el cache guarda
    lo que el modelo observo.
    """
    izq = (0.0, 0.0, 100.0, 100.0)
    der = (100.0, 0.0, 200.0, 100.0)
    cache = cache_con({0: [(1, izq)], 4: [(2, der)]}, 10)
    asignar(pila, 1, TrackRole.A, 0, 1)
    cand = GapCandidate(from_track_id=1, to_track_id=2, last_frame=0, first_frame=4, iou=0.6)
    pila.do(AcceptJoin(candidato=cand, role=TrackRole.A))

    res = IdentityResolver(doc, cache)
    for f in (1, 2, 3):
        poses = res.resolve_frame(f)
        assert len(poses) == 1
        p = poses[0]
        assert p.role is TrackRole.A
        assert p.interpolated is True
        assert p.det_conf == 0.0  # no es una deteccion
    # a la mitad del camino
    assert res.resolve_frame(2)[0].bbox[0] == pytest.approx(50.0)


def test_lo_observado_gana_sobre_lo_sintetizado(doc: AnnotationDoc, pila: UndoStack) -> None:
    cache = cache_con({0: [(1, CAJA)], 2: [(1, CAJA)], 4: [(2, CAJA)]}, 10)
    asignar(pila, 1, TrackRole.A, 0, 3)
    cand = GapCandidate(from_track_id=1, to_track_id=2, last_frame=0, first_frame=4, iou=0.9)
    pila.do(AcceptJoin(candidato=cand, role=TrackRole.A))

    res = IdentityResolver(doc, cache)
    poses = res.resolve_frame(2)
    assert len(poses) == 1
    assert poses[0].interpolated is False  # habia deteccion real, no se inventa nada


def test_una_interpolacion_rota_no_rompe_la_lectura(doc: AnnotationDoc, pila: UndoStack) -> None:
    """Si el cache no tiene los extremos que declara, se ignora en vez de explotar."""
    cand = GapCandidate(from_track_id=99, to_track_id=98, last_frame=0, first_frame=4, iou=0.9)
    pila.do(AcceptJoin(candidato=cand, role=TrackRole.A))
    res = IdentityResolver(doc, cache_con({}, 10))
    assert res.resolve_frame(2) == []


# -- relleno de huecos internos --------------------------------------------


def cache_agujereado() -> PoseCache:
    """El track 1 se ve en 0-5 y en 9-14; faltan 6, 7 y 8."""
    por_frame = {f: [(1, CAJA)] for f in list(range(0, 6)) + list(range(9, 15))}
    return cache_con(por_frame, 15)


def test_rellenar_huecos_internos(doc: AnnotationDoc, pila: UndoStack) -> None:
    cache = cache_agujereado()
    asignar(pila, 1, TrackRole.A, 0, 15)
    pila.do(FillInternalGaps(candidatos=detectar_huecos_internos(cache)))

    assert len(doc.identity.interpolations) == 1
    interp = doc.identity.interpolations[0]
    assert (interp.from_track_id, interp.to_track_id) == (1, 1)
    assert (interp.gap_start_frame, interp.gap_end_frame_excl) == (6, 9)
    assert interp.role is TrackRole.A


def test_los_cuadros_rellenados_aparecen_en_el_resolver(
    doc: AnnotationDoc, pila: UndoStack
) -> None:
    """
    El unico test que prueba lo que importa. Que se escriba el registro no sirve de nada si
    el cuadro sigue sin pose al leerlo.
    """
    cache = cache_agujereado()
    asignar(pila, 1, TrackRole.A, 0, 15)

    res = IdentityResolver(doc, cache)
    assert res.by_fighter(7)[FighterId.A] is None

    pila.do(FillInternalGaps(candidatos=detectar_huecos_internos(cache)))
    res = IdentityResolver(doc, cache)
    pose = res.by_fighter(7)[FighterId.A]
    assert pose is not None
    assert pose.interpolated


def test_deshacer_el_relleno(doc: AnnotationDoc, pila: UndoStack) -> None:
    cache = cache_agujereado()
    asignar(pila, 1, TrackRole.A, 0, 15)
    pila.do(FillInternalGaps(candidatos=detectar_huecos_internos(cache)))
    pila.undo()
    assert doc.identity.interpolations == []


def test_no_rellena_tracks_sin_rol_de_peleador(doc: AnnotationDoc, pila: UndoStack) -> None:
    """Interpolar un track ignorado seria inventar pose para alguien que se dejo afuera."""
    cache = cache_agujereado()
    asignar(pila, 1, TrackRole.IGNORE, 0, 15)
    pila.do(FillInternalGaps(candidatos=detectar_huecos_internos(cache)))
    assert doc.identity.interpolations == []


def test_no_rellena_tracks_sin_asignar(doc: AnnotationDoc, pila: UndoStack) -> None:
    cache = cache_agujereado()
    pila.do(FillInternalGaps(candidatos=detectar_huecos_internos(cache)))
    assert doc.identity.interpolations == []


def test_rellenar_dos_veces_no_duplica(doc: AnnotationDoc, pila: UndoStack) -> None:
    """Se puede apretar el boton de nuevo sin pensar: es idempotente."""
    cache = cache_agujereado()
    asignar(pila, 1, TrackRole.A, 0, 15)
    cands = detectar_huecos_internos(cache)
    pila.do(FillInternalGaps(candidatos=cands))
    segundo = FillInternalGaps(candidatos=cands)
    pila.do(segundo)
    assert segundo.aplicados == 0
    assert len(doc.identity.interpolations) == 1


def test_rechaza_candidatos_entre_tracks_distintos(doc: AnnotationDoc, pila: UndoStack) -> None:
    """
    Unir dos tracks afirma que son la misma persona y eso se confirma de a uno, por AcceptJoin.
    Colar una union en el lote saltearia esa confirmacion.
    """
    asignar(pila, 1, TrackRole.A, 0, 15)
    union = GapCandidate(from_track_id=1, to_track_id=2, last_frame=5, first_frame=9, iou=0.9)
    with pytest.raises(ValueError, match="tracks distintos"):
        pila.do(FillInternalGaps(candidatos=[union]))


# -- descartar publico ------------------------------------------------------


def test_ignora_los_tracks_chicos(doc: AnnotationDoc, pila: UndoStack) -> None:
    from boxtwin.core.identity_ops import IgnoreSmallTracks

    cmd = IgnoreSmallTracks(track_ids=[5, 6, 7], total_frames=1000, umbral=0.3)
    pila.do(cmd)
    assert cmd.aplicados == 3
    for a in doc.identity.assignments:
        assert a.role is TrackRole.IGNORE
        assert (a.start_frame, a.end_frame_excl) == (0, 1000)
    pila.undo()
    assert doc.identity.assignments == []


def test_nunca_pisa_un_track_que_ya_es_peleador(doc: AnnotationDoc, pila: UndoStack) -> None:
    """
    Un boxeador puede quedar chico en un plano abierto. Una operacion en lote que le borre el
    rol por eso es justo el error que este sistema no puede permitirse, porque no se ve: el
    esqueleto sigue estando sobre un cuerpo.
    """
    from boxtwin.core.identity_ops import IgnoreSmallTracks

    asignar(pila, 5, TrackRole.A, 0, 1000)
    cmd = IgnoreSmallTracks(track_ids=[5, 6], total_frames=1000)
    pila.do(cmd)
    assert cmd.aplicados == 1
    roles = {a.track_id: a.role for a in doc.identity.assignments}
    assert roles[5] is TrackRole.A
    assert roles[6] is TrackRole.IGNORE


def test_ignorar_dos_veces_no_duplica(doc: AnnotationDoc, pila: UndoStack) -> None:
    from boxtwin.core.identity_ops import IgnoreSmallTracks

    pila.do(IgnoreSmallTracks(track_ids=[5, 6], total_frames=1000))
    segundo = IgnoreSmallTracks(track_ids=[5, 6], total_frames=1000)
    pila.do(segundo)
    assert segundo.aplicados == 0
    assert len(doc.identity.assignments) == 2


def test_altura_maxima_usa_el_maximo_y_no_la_mediana() -> None:
    """
    Un peleador puede quedar chico durante un plano abierto. Con la mediana, un track que
    arranca de lejos y se acerca quedaria del lado equivocado.
    """
    from boxtwin.core.identity import altura_maxima_por_track

    # El track 1 esta chico casi siempre y grande una vez.
    por_frame = {f: [(1, (0.0, 0.0, 10.0, 10.0))] for f in range(9)}
    por_frame[9] = [(1, (0.0, 0.0, 10.0, 90.0))]
    alturas = altura_maxima_por_track(cache_con(por_frame, 10), 100)
    assert alturas[1] == pytest.approx(0.9)


# -- solapamientos de rol: no se truncan ------------------------------------


def test_asignar_no_le_quita_el_rol_al_anterior(doc: AnnotationDoc, pila: UndoStack) -> None:
    """
    Se probo truncar al ocupante anterior y es peor. Sobre el round anotado de Pacquiao vs
    Margarito elimina las 1165 colisiones, pero los cuadros sin pose dentro de eventos suben
    de 51 a 218: los solapamientos hacen de respaldo cuando el tracker fragmenta a un
    peleador y un track viejo vuelve a aparecer.
    """
    asignar(pila, 1, TrackRole.A, 0, 1000)
    asignar(pila, 2, TrackRole.A, 400, 1000)

    roles = [(a.track_id, a.start_frame, a.end_frame_excl) for a in doc.identity.assignments]
    assert (1, 0, 1000) in roles
    assert (2, 400, 1000) in roles


def test_ignorar_un_rango_no_toca_lo_de_afuera(doc: AnnotationDoc, pila: UndoStack) -> None:
    """El boton de plano acota a [ini, fin): fuera de ahi los track_id son otros."""
    from boxtwin.core.identity_ops import IgnoreSmallTracks

    pila.do(IgnoreSmallTracks(track_ids=[9], total_frames=600, start_frame=300))
    a = doc.identity.assignments[0]
    assert (a.start_frame, a.end_frame_excl) == (300, 600)


def test_ignorar_recorta_lo_que_ya_habia_del_mismo_track(
    doc: AnnotationDoc, pila: UndoStack
) -> None:
    """
    Sin recortar, un track con un ignore suelto previo queda con dos intervalos solapados y
    el resolver elige segun el orden de la lista. Sobre el round anotado de Pacquiao vs
    Margarito salieron 30 asi, todos ignore contra ignore.
    """
    from boxtwin.core.identity_ops import IgnoreSmallTracks
    from boxtwin.core.validation import validate_document

    asignar(pila, 9, TrackRole.IGNORE, 723, 1000)
    pila.do(IgnoreSmallTracks(track_ids=[9], total_frames=800, start_frame=718))

    tramos = sorted(
        (a.start_frame, a.end_frame_excl)
        for a in doc.identity.assignments
        if a.track_id == 9
    )
    assert tramos == [(718, 800), (800, 1000)]
    assert "ID_ASSIGNMENT_OVERLAP" not in {i.code for i in validate_document(doc)}
