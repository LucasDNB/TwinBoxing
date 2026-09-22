"""
La propuesta de candidatos para la siembra.

La invariante que importa: los dos que se le muestran al usuario tienen que estar en el
MISMO cuadro y separados. Mostrarlos en cuadros distintos no le permite a nadie descartar
que sean la misma persona dos veces, que es justo el error que la siembra existe para
evitar.
"""

from __future__ import annotations

from boxtwin.core.identidad_auto import ConfigIdentidadAuto, EvidenciaTrack, proponer
from boxtwin.mvp.candidatos import elegir

ROJO = (175.0, 200.0, 120.0)
AZUL = (105.0, 200.0, 120.0)


def _track(tid, color, frames, alto=0.9, x=100.0):
    frames = list(frames)
    return EvidenciaTrack(
        track_id=tid,
        primer_frame=frames[0] if frames else 0,
        ultimo_frame=frames[-1] if frames else 0,
        alto_max=alto,
        recortes=max(len(frames), 20),
        con_guante=max(len(frames), 20),
        colores=[(f, color) for f in frames],
        frames_con_guante=frames,
        xs=[(f, x) for f in frames],
    )


def test_propone_el_par_que_mas_coexiste():
    ev = {
        1: _track(1, ROJO, range(0, 300, 5)),
        2: _track(2, AZUL, range(0, 300, 5), x=500.0),
        3: _track(3, ROJO, range(400, 430, 5), x=120.0),
    }
    pareja, cands = elegir(ev, [1, 2, 3])
    assert pareja == (1, 2)
    assert [c.track for c in cands[:2]] == [1, 2]


def test_los_dos_sugeridos_salen_del_mismo_cuadro():
    ev = {1: _track(1, ROJO, range(0, 300, 5)),
          2: _track(2, AZUL, range(0, 300, 5), x=500.0)}
    _, cands = elegir(ev, [1, 2])
    assert cands[0].cuadro == cands[1].cuadro
    assert cands[0].coexiste_con == 2 and cands[1].coexiste_con == 1


def test_el_cuadro_elegido_no_es_el_primero():
    # Al principio del video suele estar alguien entrando al plano, con medio cuerpo afuera.
    ev = {1: _track(1, ROJO, range(0, 300, 5)),
          2: _track(2, AZUL, range(0, 300, 5), x=500.0)}
    _, cands = elegir(ev, [1, 2])
    assert cands[0].cuadro > 0


def test_sin_coexistencia_no_se_propone_pareja_pero_si_candidatos():
    # El caso de "no pude separar dos: elegi vos" del flujo. Devolver una pareja igual
    # seria afirmar que son dos personas sin tener con que.
    ev = {1: _track(1, ROJO, range(0, 100, 5)),
          2: _track(2, AZUL, range(200, 300, 5), x=500.0)}
    pareja, cands = elegir(ev, [1, 2])
    assert pareja is None
    assert len(cands) == 2
    assert all(c.coexiste_con is None for c in cands)


def test_los_candidatos_se_ordenan_por_evidencia():
    ev = {
        1: _track(1, ROJO, range(0, 300, 5)),
        2: _track(2, AZUL, range(0, 300, 5), x=500.0),
        3: _track(3, ROJO, range(400, 420, 5)),
        4: _track(4, AZUL, range(400, 500, 5), x=500.0),
    }
    _, cands = elegir(ev, [1, 2, 3, 4])
    assert [c.track for c in cands] == [1, 2, 4, 3]


def test_no_se_muestran_mas_de_los_pedidos():
    ev = {i: _track(i, ROJO, range(0, 300, 5), x=100.0 * i) for i in range(1, 9)}
    _, cands = elegir(ev, list(ev), maximo=4)
    assert len(cands) == 4


def test_la_propuesta_es_determinista():
    ev = {
        1: _track(1, ROJO, range(0, 300, 5)),
        2: _track(2, AZUL, range(0, 300, 5), x=500.0),
        3: _track(3, ROJO, range(0, 300, 5), x=700.0),
    }
    primera = elegir(ev, [1, 2, 3])
    for _ in range(5):
        assert elegir(ev, [3, 1, 2])[0] == primera[0]


def test_los_candidatos_salen_del_mismo_nucleo_que_usa_la_identidad():
    # Una sola definicion de peleador. Si la propuesta usara la suya, el usuario podria
    # elegir como semilla un track que la asignacion despues descarta por no ser peleador.
    ev = {
        1: _track(1, ROJO, range(0, 300, 5)),
        2: _track(2, AZUL, range(0, 300, 5), x=500.0),
        9: _track(9, ROJO, range(0, 300, 5), alto=0.2, x=900.0),   # publico: muy bajo
    }
    prop = proponer(ev, ConfigIdentidadAuto(), total_frames=400, fps=30.0)
    nucleo = prop.diagnostico["nucleo_tracks"]
    assert 9 not in nucleo
    _, cands = elegir(ev, nucleo)
    assert 9 not in [c.track for c in cands]
