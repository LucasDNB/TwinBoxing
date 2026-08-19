"""
Reporte de acuerdo.

Kappa se verifica contra un caso de valor conocido, no contra lo que devuelva la
implementacion: un test que compara el codigo consigo mismo no detecta que la formula este
mal, y una formula mal puesta produce un numero plausible que despues va a la tesis.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from boxtwin.core.agreement import (
    DIMENSIONES,
    a_texto,
    cohen_kappa,
    comparar,
    matriz_confusion,
)
from boxtwin.core.reanno import ReannoTrial, TrialLabels, sortear
from boxtwin.core.schema import AnnotationDoc, Event
from boxtwin.core.types import (
    Completeness,
    FighterId,
    Guard,
    Landed,
    PunchType,
    Quality,
    Side,
    Target,
)

AR = timezone(timedelta(hours=-3))
T0 = datetime(2026, 8, 11, 14, 0, 0, tzinfo=AR)


# -- kappa -----------------------------------------------------------------


def test_kappa_caso_conocido() -> None:
    """
    Caso clasico de 2x2: [[20, 5], [10, 15]].
    po = 35/50 = 0.7 · pe = (25*30 + 25*20)/2500 = 0.5 · kappa = 0.4
    """
    po, pe, k, nota = cohen_kappa([[20, 5], [10, 15]])
    assert po == pytest.approx(0.7)
    assert pe == pytest.approx(0.5)
    assert k == pytest.approx(0.4)
    assert nota == ""


def test_kappa_acuerdo_perfecto() -> None:
    _, _, k, _ = cohen_kappa([[10, 0], [0, 10]])
    assert k == pytest.approx(1.0)


def test_kappa_cero_cuando_el_acuerdo_es_el_del_azar() -> None:
    """
    Es la razon de usar kappa y no porcentaje: aca hay 50% de acuerdo y no vale nada.
    """
    _, _, k, _ = cohen_kappa([[25, 25], [25, 25]])
    assert k == pytest.approx(0.0)


def test_kappa_negativa_con_desacuerdo_sistematico() -> None:
    _, _, k, _ = cohen_kappa([[0, 20], [20, 0]])
    assert k < 0


def test_kappa_no_definida_con_una_sola_categoria() -> None:
    """
    Si los dos usaron una sola categoria, la formula divide por cero. No es acuerdo perfecto
    ni nulo: la medida no aplica, y decir 1.0 seria mentir.
    """
    po, pe, k, nota = cohen_kappa([[30, 0], [0, 0]])
    assert po == pytest.approx(1.0)
    assert k is None
    assert "no definida" in nota


def test_kappa_sin_intentos() -> None:
    _, _, k, nota = cohen_kappa([[0, 0], [0, 0]])
    assert k is None
    assert "sin intentos" in nota


def test_matriz_de_confusion() -> None:
    a = ["x", "x", "y", "y", "y"]
    b = ["x", "y", "y", "y", "x"]
    m = matriz_confusion(a, b, ["x", "y"])
    assert m == [[1, 1], [1, 2]]  # filas: original, columnas: reanotacion


# -- comparacion completa --------------------------------------------------


def ev(eid: str, ini: int, **kw) -> Event:
    base = dict(
        id=eid, fighter=FighterId.A, start_frame=ini, end_frame=ini + 20,
        side=Side.LEFT, punch_type=PunchType.STRAIGHT, target=Target.HEAD,
        completeness=Completeness.FULL, landed=Landed.LANDED,
        guard=Guard.ORTHODOX, quality=Quality.CLEAN,
    )
    base.update(kw)
    return Event(**base)


def intento(eid: str, ini: int, fin: int, *, revealed: bool = False, **kw) -> ReannoTrial:
    labels = dict(
        start_frame=ini, end_frame=fin, side=Side.LEFT, punch_type=PunchType.STRAIGHT,
        target=Target.HEAD, completeness=Completeness.FULL,
    )
    labels.update(kw)
    return ReannoTrial(
        event_id=eid, fighter=FighterId.A, annotator="lucas", annotated_at=T0,
        revealed=revealed, punches=[TrialLabels(**labels)],
    )


@pytest.fixture
def par(doc_min: AnnotationDoc):
    doc_min.video.total_frames = 5000
    doc_min.events = [ev(f"ev_{i:04d}", 100 + i * 60) for i in range(1, 11)]
    re_doc = sortear(doc_min, fraction=1.0, seed=1)
    return doc_min, re_doc


def test_acuerdo_perfecto(par) -> None:
    doc, re_doc = par
    re_doc.trials = [
        intento(e.id, e.start_frame, e.end_frame) for e in doc.events
    ]
    r = comparar(doc, re_doc)
    assert r.n_ciegos == 10
    for dim in DIMENSIONES:
        ac = r.dimensiones[dim]
        assert ac.acuerdo_observado == 1.0
    assert r.fronteras["start"]["mae"] == 0.0
    assert r.fronteras["start"]["exactos"] == 10


def test_desacuerdo_en_una_dimension(par) -> None:
    doc, re_doc = par
    re_doc.trials = [
        intento(e.id, e.start_frame, e.end_frame,
                punch_type=PunchType.HOOK if i < 3 else PunchType.STRAIGHT)
        for i, e in enumerate(doc.events)
    ]
    r = comparar(doc, re_doc)
    assert r.dimensiones["punch_type"].acuerdo_observado == pytest.approx(0.7)
    assert r.dimensiones["side"].acuerdo_observado == 1.0  # las demas no se contaminan


def test_los_revelados_no_entran(par) -> None:
    """Un intento revelado mide si el anotador acepta lo que habia, no si llega solo."""
    doc, re_doc = par
    re_doc.trials = [
        intento(e.id, e.start_frame, e.end_frame, revealed=(i < 4))
        for i, e in enumerate(doc.events)
    ]
    r = comparar(doc, re_doc)
    assert r.n_intentos == 10
    assert r.n_ciegos == 6
    assert r.n_revelados == 4
    assert r.dimensiones["side"].n == 6
    assert any("no fueron ciegos" in a for a in r.avisos)


def test_incluir_revelados_a_pedido(par) -> None:
    doc, re_doc = par
    re_doc.trials = [
        intento(e.id, e.start_frame, e.end_frame, revealed=True) for e in doc.events
    ]
    assert comparar(doc, re_doc, solo_ciegos=False).dimensiones["side"].n == 10


def test_error_de_fronteras_con_sesgo(par) -> None:
    """
    El sesgo distingue una definicion dificil de aplicar de una entendida distinto: ruido
    simetrico da sesgo cerca de cero, interpretacion distinta da sesgo consistente.
    """
    doc, re_doc = par
    re_doc.trials = [
        intento(e.id, e.start_frame + 3, e.end_frame + 3) for e in doc.events
    ]
    r = comparar(doc, re_doc)
    assert r.fronteras["start"]["mae"] == 3.0
    assert r.fronteras["start"]["sesgo"] == 3.0  # todos tarde: sesgo, no ruido
    assert r.fronteras["duracion"]["mae"] == 0.0  # la duracion no cambio


def test_ruido_simetrico_no_deja_sesgo(par) -> None:
    doc, re_doc = par
    re_doc.trials = [
        intento(e.id, e.start_frame + (2 if i % 2 else -2), e.end_frame)
        for i, e in enumerate(doc.events)
    ]
    r = comparar(doc, re_doc)
    assert r.fronteras["start"]["mae"] == 2.0
    assert abs(r.fronteras["start"]["sesgo"]) < 0.01


def test_avisa_con_n_bajo(par) -> None:
    doc, re_doc = par
    e = doc.events[0]
    re_doc.trials = [intento(e.id, e.start_frame, e.end_frame)]
    r = comparar(doc, re_doc)
    assert any("inestable con n bajo" in a for a in r.avisos)


def test_avisa_de_acuerdo_bajo(par) -> None:
    doc, re_doc = par
    re_doc.trials = [
        intento(e.id, e.start_frame, e.end_frame,
                punch_type=PunchType.HOOK if i % 2 else PunchType.STRAIGHT)
        for i, e in enumerate(doc.events)
    ]
    r = comparar(doc, re_doc)
    assert any("revisar la definicion" in a for a in r.avisos)


def test_un_evento_borrado_no_rompe_el_reporte(par) -> None:
    doc, re_doc = par
    re_doc.trials = [intento(e.id, e.start_frame, e.end_frame) for e in doc.events]
    doc.events = doc.events[:5]
    r = comparar(doc, re_doc)
    assert r.dimensiones["side"].n == 5
    assert any("ya no existe" in a for a in r.avisos)


def test_sin_intentos(par) -> None:
    doc, re_doc = par
    r = comparar(doc, re_doc)
    assert r.dimensiones == {}
    assert any("no hay golpes emparejados" in a for a in r.avisos)


# -- salida ----------------------------------------------------------------


def test_el_texto_declara_el_n(par) -> None:
    """Es lo que se pega en el capitulo: un kappa sin su n no se puede interpretar."""
    doc, re_doc = par
    re_doc.trials = [intento(e.id, e.start_frame, e.end_frame) for e in doc.events]
    txt = a_texto(comparar(doc, re_doc))
    assert "intentos: 10" in txt
    assert "kappa" in txt
    assert "matrices de confusion" in txt
    assert "fronteras, en cuadros" in txt


def test_las_cuatro_dimensiones_del_enunciado() -> None:
    assert set(DIMENSIONES) == {"side", "punch_type", "target", "completeness"}


# -- emparejamiento --------------------------------------------------------


class Marca:
    """Un golpe cualquiera con fronteras, que es lo unico que mira el emparejador."""

    def __init__(self, ini: int, fin: int) -> None:
        self.start_frame, self.end_frame = ini, fin


def test_empareja_por_solapamiento() -> None:
    from boxtwin.core.agreement import emparejar

    parejas, faltan, sobran = emparejar([Marca(100, 110)], [Marca(101, 111)])
    assert parejas and parejas[0][:2] == (0, 0)
    assert not faltan and not sobran


def test_no_empareja_lo_que_no_se_toca() -> None:
    from boxtwin.core.agreement import emparejar

    parejas, faltan, sobran = emparejar([Marca(100, 110)], [Marca(200, 210)])
    assert parejas == []
    assert faltan == [0] and sobran == [0]


def test_una_combinacion_no_se_colapsa_en_una_pareja() -> None:
    """
    Uno a uno: sin esa restriccion un golpe reanotado largo se llevaria los dos de un 1-2 y
    el recall saldria inflado.
    """
    from boxtwin.core.agreement import emparejar

    jab, cross = Marca(100, 110), Marca(108, 120)
    largo = Marca(100, 120)
    parejas, faltan, sobran = emparejar([jab, cross], [largo])
    assert len(parejas) == 1
    assert len(faltan) == 1


def test_el_mejor_solapamiento_gana() -> None:
    from boxtwin.core.agreement import emparejar

    parejas, _, _ = emparejar([Marca(100, 110)], [Marca(130, 140), Marca(101, 111)])
    assert parejas[0][1] == 1


def test_deteccion_cuenta_omitidos_y_agregados(par) -> None:
    """Los dos numeros que la v1 no podia dar: si el golpe se encontro."""
    doc, re_doc = par
    e = doc.events[0]
    # Se marca el golpe correcto y ademas uno que no existe en la anotacion.
    t = intento(e.id, e.start_frame, e.end_frame)
    t.punches = [*t.punches, TrialLabels(
        start_frame=e.start_frame, end_frame=e.start_frame + 5,
        side=Side.LEFT, punch_type=PunchType.HOOK, target=Target.HEAD,
        completeness=Completeness.FULL,
    )]
    re_doc.trials = [t]
    r = comparar(doc, re_doc)
    assert r.deteccion["emparejados"] == 1
    assert r.deteccion["agregados"] == 1
    assert r.deteccion["precision"] == 0.5


def test_una_ventana_vacia_es_una_respuesta(par) -> None:
    """Cero golpes es "no vi ninguno", no un intento invalido."""
    doc, re_doc = par
    e = doc.events[0]
    t = intento(e.id, e.start_frame, e.end_frame)
    t.punches = []
    re_doc.trials = [t]
    r = comparar(doc, re_doc)
    assert r.deteccion["ventanas_sin_ningun_golpe"] == 1
    assert r.deteccion["omitidos"] >= 1
    assert r.deteccion["recall"] == 0.0
