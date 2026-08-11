"""Mapa de teclas: superposicion sobre el default y rechazo de conflictos."""

from __future__ import annotations

from pathlib import Path

import pytest

from boxtwin.gui.keymap import DEFAULT_KEYMAP, Keymap, KeymapError


def escribir_config(tmp_path: Path, cuerpo: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(cuerpo, encoding="utf-8")
    return p


def test_default_sin_archivo(tmp_path: Path) -> None:
    km = Keymap.load(tmp_path / "no_existe.yaml")
    assert km.sequence("player.play_pause") == "Space"
    assert km.bindings == DEFAULT_KEYMAP


def test_el_default_no_tiene_conflictos() -> None:
    Keymap().validate()


def test_config_se_superpone_sin_reemplazar(tmp_path: Path) -> None:
    """
    Agregar una accion nueva en una version posterior no puede dejar sin teclas a quien ya
    tenia su config escrita.
    """
    cfg = escribir_config(tmp_path, "keymap:\n  player.play_pause: 'Ctrl+Space'\n")
    km = Keymap.load(cfg)
    assert km.sequence("player.play_pause") == "Ctrl+Space"
    assert km.sequence("player.step_forward") == "Right"
    assert set(km.bindings) == set(DEFAULT_KEYMAP)


def test_rechaza_accion_desconocida(tmp_path: Path) -> None:
    """Un typo en el nombre de la accion no puede quedar en silencio."""
    cfg = escribir_config(tmp_path, "keymap:\n  player.plaay_pause: 'Ctrl+Space'\n")
    with pytest.raises(KeymapError, match="desconocidas"):
        Keymap.load(cfg)


def test_rechaza_teclas_repetidas(tmp_path: Path) -> None:
    """
    Con dos acciones en la misma tecla gana una y la otra deja de responder sin avisar, que
    en medio de una sesion se siente como que la aplicacion se colgo.
    """
    cfg = escribir_config(tmp_path, "keymap:\n  player.step_forward: 'Space'\n")
    with pytest.raises(KeymapError, match="repetidas"):
        Keymap.load(cfg)


def test_tecla_vacia_desactiva_sin_chocar(tmp_path: Path) -> None:
    cfg = escribir_config(
        tmp_path, "keymap:\n  player.step_forward: ''\n  player.step_back: ''\n"
    )
    km = Keymap.load(cfg)
    assert km.sequence("player.step_forward") == ""


def test_config_sin_seccion_keymap(tmp_path: Path) -> None:
    cfg = escribir_config(tmp_path, "otra_cosa:\n  valor: 1\n")
    assert Keymap.load(cfg).bindings == DEFAULT_KEYMAP


def test_accion_inexistente_al_consultar() -> None:
    with pytest.raises(KeymapError, match="accion desconocida"):
        Keymap().sequence("no.existe")


def test_incluye_las_acciones_de_bloques_futuros() -> None:
    """
    Las acciones de anotacion e identidad se declaran ya, aunque sus manejadores lleguen
    despues: asi el archivo del usuario no cambia de forma y se remapea una sola vez.
    """
    for accion in ("event.mark_start", "event.mark_end", "edit.undo", "file.save"):
        assert accion in DEFAULT_KEYMAP
