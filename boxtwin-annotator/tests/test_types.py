"""Derivacion lead/rear. Las cuatro combinaciones, porque invertirla es un error silencioso."""

from __future__ import annotations

import pytest

from boxtwin.core.types import ArmRole, Guard, Side, arm_role


@pytest.mark.parametrize(
    "guard, side, esperado",
    [
        # El jab del ortodoxo sale de la izquierda adelantada.
        (Guard.ORTHODOX, Side.LEFT, ArmRole.LEAD),
        # El cross del ortodoxo sale de la derecha atrasada.
        (Guard.ORTHODOX, Side.RIGHT, ArmRole.REAR),
        # En zurdo se invierte: la izquierda pasa a ser la mano de poder.
        (Guard.SOUTHPAW, Side.LEFT, ArmRole.REAR),
        (Guard.SOUTHPAW, Side.RIGHT, ArmRole.LEAD),
    ],
)
def test_arm_role_cuatro_combinaciones(guard: Guard, side: Side, esperado: ArmRole) -> None:
    assert arm_role(guard, side) is esperado


def test_arm_role_es_biyectiva_por_guardia() -> None:
    """Fijada la guardia, los dos lados dan roles distintos. Si no, el mapeo esta roto."""
    for guard in Guard:
        roles = {arm_role(guard, side) for side in Side}
        assert roles == {ArmRole.LEAD, ArmRole.REAR}


def test_event_arm_role_usa_la_guardia_del_evento(doc_rich) -> None:
    """
    El evento lleva su propia guardia, no la del peleador.

    ev_0044 es de fighter_B, que es southpaw por defecto, pero cae en el tramo en que
    cambio a ortodoxo. Un uppercut derecho ahi es mano atrasada, no adelantada.
    """
    ev = doc_rich.event_by_id("ev_0044")
    assert ev is not None
    assert ev.guard is Guard.ORTHODOX
    assert ev.arm_role is ArmRole.REAR


def test_arm_role_no_se_serializa(doc_rich) -> None:
    """Es derivado. Si apareciera en el archivo habria dos fuentes de verdad."""
    payload = doc_rich.model_dump(mode="json")
    assert "arm_role" not in payload["events"][0]
