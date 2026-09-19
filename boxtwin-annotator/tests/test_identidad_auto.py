"""
La asignacion automatica de identidad, sobre evidencia sintetica.

Lo que se fija aca son las invariantes que costaron una medicion cada una: que dos semillas
del mismo peleador no se elijan, que un piso de separacion bloquee en vez de avisar, y que
lo indecidible quede sin asignar en vez de adivinado.
"""

from __future__ import annotations

import pytest

from boxtwin.core.deteccion_guantes import ACROMATICO, distancia_color, perfil_de
from boxtwin.core.identidad_auto import (
    ConfigIdentidadAuto,
    EvidenciaTrack,
    cargar_evidencia,
    guardar_evidencia,
    proponer,
)
from boxtwin.core.types import TrackRole

ROJO = (175.0, 200.0, 120.0)
AZUL = (105.0, 200.0, 120.0)
FPS = 30.0


def _track(tid, color, frames, alto=0.9, recortes=None, primer=None, ultimo=None):
    """Un track con un color constante en los cuadros dados."""
    frames = list(frames)
    return EvidenciaTrack(
        track_id=tid,
        primer_frame=primer if primer is not None else (frames[0] if frames else 0),
        ultimo_frame=ultimo if ultimo is not None else (frames[-1] if frames else 0),
        alto_max=alto,
        recortes=recortes if recortes is not None else max(len(frames), 20),
        con_guante=max(len(frames), 20),
        colores=[(f, color) for f in frames],
        frames_con_guante=frames,
        xs=[(f, 100.0 if color is ROJO else 500.0) for f in frames],
    )


def _dos_peleadores(n=40, color_a=ROJO, color_b=AZUL):
    """Dos tracks que coexisten: son dos personas distintas por construccion."""
    frames = list(range(0, n * 5, 5))
    return {1: _track(1, color_a, frames), 2: _track(2, color_b, frames)}


# -- color ------------------------------------------------------------------


def test_el_tono_se_promedia_circularmente():
    # El rojo vive en los dos extremos de la escala. Una media aritmetica de 5 y 175 daria
    # 90, que es verde: el color que ninguno de los dos tiene.
    p = perfil_de([(5.0, 200.0, 120.0), (175.0, 200.0, 120.0)])
    assert p[0] < 10 or p[0] > 170


def test_dos_colores_iguales_distan_cero():
    assert distancia_color(ROJO, ROJO) == pytest.approx(0.0)


def test_un_acromatico_y_uno_con_color_no_se_confunden():
    negro = (ACROMATICO, 10.0, 20.0)
    assert distancia_color(negro, ROJO) > distancia_color(negro, (ACROMATICO, 12.0, 22.0))


# -- filtros ----------------------------------------------------------------


def test_un_track_bajo_se_descarta_por_altura():
    ev = _dos_peleadores()
    ev[9] = _track(9, ROJO, list(range(0, 100, 5)), alto=0.2)
    prop = proponer(ev, ConfigIdentidadAuto(), total_frames=500, fps=FPS)
    assert prop.roles[9] is TrackRole.IGNORE


def test_un_track_sin_guantes_se_descarta_aunque_sea_alto():
    # Es el arbitro: adentro del ring, del tamano de un peleador, sin guantes de boxeo.
    ev = _dos_peleadores()
    arbitro = _track(9, ROJO, [], alto=0.95, recortes=100)
    ev[9] = EvidenciaTrack(
        track_id=9, primer_frame=0, ultimo_frame=400, alto_max=0.95,
        recortes=100, con_guante=5,   # fraccion 0,05
    )
    prop = proponer(ev, ConfigIdentidadAuto(), total_frames=500, fps=FPS)
    assert prop.roles[9] is TrackRole.IGNORE


def test_un_track_con_poquisimos_recortes_no_decide_nada():
    ev = _dos_peleadores()
    ev[9] = _track(9, ROJO, [0, 5], recortes=2)
    prop = proponer(ev, ConfigIdentidadAuto(), total_frames=500, fps=FPS)
    assert prop.roles[9] is TrackRole.IGNORE


# -- siembra ----------------------------------------------------------------


def test_las_dos_semillas_coexisten_y_son_personas_distintas():
    ev = _dos_peleadores()
    prop = proponer(ev, ConfigIdentidadAuto(), total_frames=500, fps=FPS)
    assert set(prop.semillas.values()) == {1, 2}


def test_dos_fragmentos_de_la_misma_persona_no_se_eligen_como_semillas():
    # El error que costo tres fuentes al borde del azar: dos tracks con muchos guantes que
    # nunca aparecen a la vez son el MISMO peleador partido en dos ids.
    pares = list(range(0, 200, 10))
    impares = list(range(5, 205, 10))
    ev = {
        1: _track(1, ROJO, pares),      # fragmento uno
        2: _track(2, ROJO, impares),    # fragmento dos, nunca coinciden
        3: _track(3, AZUL, pares),      # el rival de verdad, coexiste con el 1
    }
    prop = proponer(ev, ConfigIdentidadAuto(), total_frames=500, fps=FPS)
    elegidas = set(prop.semillas.values())
    assert elegidas == {1, 3}, "tienen que ser dos tracks que coexisten"


def test_sin_coexistencia_suficiente_no_siembra():
    ev = {
        1: _track(1, ROJO, list(range(0, 200, 10))),
        2: _track(2, AZUL, list(range(5, 205, 10))),
    }
    cfg = ConfigIdentidadAuto(min_coexistencia=5)
    prop = proponer(ev, cfg, total_frames=500, fps=FPS)
    assert not prop.semillas
    assert prop.avisos


# -- piso de separacion -----------------------------------------------------


def test_con_guantes_del_mismo_color_la_geometria_igual_asigna():
    # Dos tracks que aparecen en el mismo cuadro, cada uno aislado, son dos personas
    # distintas. Eso no depende del color y por eso vale aunque los guantes sean iguales:
    # es la razon por la que la coexistencia va antes que la apariencia.
    ev = _dos_peleadores(color_a=ROJO, color_b=(174.0, 198.0, 119.0))
    prop = proponer(ev, ConfigIdentidadAuto(), total_frames=500, fps=FPS)
    assert prop.roles[1] is not prop.roles[2]


def test_pero_los_sueltos_no_se_deciden_por_un_color_que_no_distingue():
    # Un track que nunca coexiste no tiene restriccion geometrica, y con los dos perfiles
    # casi iguales el voto es una moneda. Medido: con separacion 0,449 el reparto salio 13
    # tracks a A contra 1 a B, una asignacion equivocada con cara de correcta.
    ev = _dos_peleadores(color_a=ROJO, color_b=(174.0, 198.0, 119.0))
    ev[9] = _track(9, ROJO, list(range(1000, 1200, 5)))   # solo, mucho despues
    prop = proponer(ev, ConfigIdentidadAuto(), total_frames=2000, fps=FPS)
    assert 9 in prop.sin_asignar
    assert any("mismo color" in a for a in prop.avisos)


def test_los_filtros_de_descarte_no_dependen_de_poder_decidir_a_y_b():
    ev = _dos_peleadores(color_a=ROJO, color_b=(174.0, 198.0, 119.0))
    ev[9] = _track(9, ROJO, list(range(0, 100, 5)), alto=0.2)
    prop = proponer(ev, ConfigIdentidadAuto(), total_frames=500, fps=FPS)
    assert prop.roles[9] is TrackRole.IGNORE


# -- asignacion -------------------------------------------------------------


def test_cada_peleador_se_lleva_sus_fragmentos():
    # Los fragmentos van DESPUES en el tiempo, que es como se fragmenta de verdad un track:
    # el id se pierde y aparece otro. Coexistir con el original seria imposible.
    ev = _dos_peleadores(n=40)
    tarde = list(range(1000, 1200, 5))
    ev[10] = _track(10, ROJO, tarde)
    ev[11] = _track(11, AZUL, tarde)
    prop = proponer(ev, ConfigIdentidadAuto(), total_frames=2000, fps=FPS)
    assert prop.roles[10] is prop.roles[1]
    assert prop.roles[11] is prop.roles[2]
    assert prop.roles[1] is not prop.roles[2]


def test_un_intercambio_mutuo_de_identidad_no_rompe_la_particion():
    # El caso que ninguna eleccion de semilla arreglaba: el tracker cruza los dos ids entre
    # las dos personas, asi que el color de cada track mezcla a los dos. Los dos siguen
    # coexistiendo y separados, y la geometria los sigue repartiendo bien.
    mitad = [(f, ROJO) for f in range(0, 100, 5)]
    otra = [(f, AZUL) for f in range(100, 200, 5)]
    frames = [f for f, _ in mitad + otra]
    ev = {
        1: EvidenciaTrack(1, 0, 200, 0.9, 40, 40, mitad + otra, frames,
                          [(f, 100.0) for f in frames]),
        2: EvidenciaTrack(2, 0, 200, 0.9, 40, 40,
                          [(f, AZUL) for f, _ in mitad] + [(f, ROJO) for f, _ in otra],
                          frames, [(f, 500.0) for f in frames]),
    }
    prop = proponer(ev, ConfigIdentidadAuto(), total_frames=500, fps=FPS)
    assert prop.roles[1] is not prop.roles[2], "coexisten: son dos personas, pase lo que pase"


def test_un_track_con_pocos_guantes_queda_sin_asignar_y_no_adivinado():
    ev = _dos_peleadores()
    ev[12] = _track(12, ROJO, [0, 5, 10], recortes=30)
    prop = proponer(ev, ConfigIdentidadAuto(min_guantes_voto=10),
                    total_frames=500, fps=FPS)
    assert 12 in prop.sin_asignar
    assert 12 not in prop.roles, "sin asignar no es lo mismo que ignore"


def test_los_rangos_cubren_la_vida_del_track():
    ev = _dos_peleadores()
    prop = proponer(ev, ConfigIdentidadAuto(), total_frames=500, fps=FPS)
    ini, fin = prop.rangos[1]
    assert ini == ev[1].primer_frame and fin == ev[1].ultimo_frame + 1


# -- persistencia -----------------------------------------------------------


def test_la_evidencia_sobrevive_la_ida_y_vuelta_a_disco(tmp_path):
    ev = _dos_peleadores()
    destino = tmp_path / "ev.json"
    guardar_evidencia(ev, destino)
    vuelta = cargar_evidencia(destino)
    assert set(vuelta) == set(ev)
    assert vuelta[1].colores == ev[1].colores
    assert vuelta[1].alto_max == ev[1].alto_max


def test_un_json_ajeno_se_rechaza(tmp_path):
    destino = tmp_path / "otro.json"
    destino.write_text('{"kind": "otra.cosa", "tracks": []}')
    with pytest.raises(ValueError, match="evidencia"):
        cargar_evidencia(destino)


# -- el comando -------------------------------------------------------------


def _propuesta_simple():
    ev = _dos_peleadores()
    return proponer(ev, ConfigIdentidadAuto(), total_frames=500, fps=FPS)


def test_el_comando_escribe_las_asignaciones(doc_min):
    from boxtwin.core.identidad_auto import AsignarIdentidadAuto

    cmd = AsignarIdentidadAuto(_propuesta_simple())
    cmd.do(doc_min)
    roles = {a.track_id: a.role for a in doc_min.identity.assignments}
    assert roles[1] is not roles[2]
    assert cmd.aplicados == len(doc_min.identity.assignments)


def test_un_solo_ctrl_z_revierte_la_operacion_entera(doc_min):
    from boxtwin.core.identidad_auto import AsignarIdentidadAuto

    antes = len(doc_min.identity.assignments)
    cmd = AsignarIdentidadAuto(_propuesta_simple())
    cmd.do(doc_min)
    assert len(doc_min.identity.assignments) != antes
    cmd.undo(doc_min)
    assert len(doc_min.identity.assignments) == antes


def test_se_niega_a_pisar_una_correccion_manual(doc_min):
    from datetime import datetime

    from boxtwin.core.identidad_auto import AsignarIdentidadAuto
    from boxtwin.core.schema import Assignment, Origin
    from boxtwin.core.types import AssignmentOp

    # Un swap corregido a mano y borrado en silencio reaparece recien en el export, con los
    # keypoints del peleador equivocado adentro.
    doc_min.identity.assignments = [
        Assignment(
            id="as_1", track_id=7, role=TrackRole.A, start_frame=0, end_frame_excl=100,
            origin=Origin(op=AssignmentOp.SWAP, op_id="op_1", at_frame=0,
                          annotator="lucas", created_at=datetime.now().astimezone()),
        )
    ]
    with pytest.raises(ValueError, match="manual"):
        AsignarIdentidadAuto(_propuesta_simple()).do(doc_min)


def test_con_forzar_si_la_pisa(doc_min):
    from datetime import datetime

    from boxtwin.core.identidad_auto import AsignarIdentidadAuto
    from boxtwin.core.schema import Assignment, Origin
    from boxtwin.core.types import AssignmentOp

    doc_min.identity.assignments = [
        Assignment(
            id="as_1", track_id=7, role=TrackRole.A, start_frame=0, end_frame_excl=100,
            origin=Origin(op=AssignmentOp.SWAP, op_id="op_1", at_frame=0,
                          annotator="lucas", created_at=datetime.now().astimezone()),
        )
    ]
    AsignarIdentidadAuto(_propuesta_simple(), forzar=True).do(doc_min)
    assert 7 not in {a.track_id for a in doc_min.identity.assignments}
