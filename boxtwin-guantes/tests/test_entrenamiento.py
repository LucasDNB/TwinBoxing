import pytest

from boxtwin_guantes.entrenamiento import ConfigEntrenamiento, entrenar


def test_el_flip_vertical_va_apagado_por_defecto():
    # Un guante de boxeo dado vuelta no existe: ensenar esa orientacion gasta capacidad en
    # algo que el modelo nunca va a ver. Se fija explicito por ser decision de dominio.
    assert ConfigEntrenamiento().flipud == 0.0
    assert ConfigEntrenamiento().fliplr == 0.5


def test_la_resolucion_por_defecto_es_la_del_recorte():
    # Las entradas son recortes de persona, no cuadros completos.
    assert ConfigEntrenamiento().imgsz == 320


def test_sin_data_yaml_falla_antes_de_cargar_el_modelo(tmp_path):
    # Tiene que fallar por el dataset y no por torch: el mensaje dice que correr primero.
    pytest.importorskip("ultralytics")
    with pytest.raises(FileNotFoundError, match="recortes"):
        entrenar(tmp_path / "no-esta", tmp_path / "salida")
