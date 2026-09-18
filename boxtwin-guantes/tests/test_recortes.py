import pytest

from boxtwin_guantes.recortes import ConfigRecortes, _nombre_corto, recorte_de

CFG = ConfigRecortes()
# Persona de 400 px de alto en una imagen de 1000x1000, centrada.
PERSONA = (300.0, 300.0, 500.0, 700.0)
ANCHO = ALTO = 1000


def _guante(cx, cy, w=0.05, h=0.062, clase=0):
    """Por defecto, un guante de 62 px sobre una persona de 400: ratio 0,155, el anatomico."""
    return (clase, cx, cy, w, h)


# -- geometria --------------------------------------------------------------


def test_el_guante_queda_dentro_del_recorte():
    # La persona va de x=300 a x=500, asi que su centro son 400 px = 0,40 normalizado.
    # El guante se pone a 0,35 para que quede claramente a la izquierda.
    res = recorte_de(PERSONA, [_guante(0.35, 0.45)], ANCHO, ALTO, CFG)
    assert res is not None
    (x0, y0, x1, y1), cajas = res
    assert len(cajas) == 1
    _, cx, cy, w, h = cajas[0]
    assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
    # Estaba a la izquierda y arriba del centro de la persona, y sigue estandolo.
    assert cx < 0.5 and cy < 0.5


def test_el_recorte_agranda_la_caja_de_persona():
    res = recorte_de(PERSONA, [_guante(0.40, 0.45)], ANCHO, ALTO, CFG)
    (x0, y0, x1, y1), _ = res
    assert x0 < PERSONA[0] and y0 < PERSONA[1]
    assert x1 > PERSONA[2] and y1 > PERSONA[3]


def test_el_recorte_no_se_sale_de_la_imagen():
    # Persona pegada al borde: el margen no puede producir coordenadas negativas.
    res = recorte_de((0.0, 0.0, 200.0, 400.0), [_guante(0.10, 0.20)], ANCHO, ALTO, CFG)
    (x0, y0, x1, y1), _ = res
    assert x0 >= 0 and y0 >= 0 and x1 <= ANCHO and y1 <= ALTO


# -- filtros ----------------------------------------------------------------


def test_un_guante_de_otra_persona_no_entra():
    # Centro del guante fuera de la caja de persona: es del rival, no de esta.
    assert recorte_de(PERSONA, [_guante(0.90, 0.45)], ANCHO, ALTO, CFG) is None


def test_una_persona_sin_guantes_no_da_recorte():
    assert recorte_de(PERSONA, [], ANCHO, ALTO, CFG) is None


def test_una_proporcion_de_retrato_descarta_el_recorte():
    # Guante de 200 px sobre una persona de 400: ratio 0,5. Eso es alguien mostrando el
    # guante a camara, y ensena una proporcion que en un ring no existe.
    assert recorte_de(PERSONA, [_guante(0.40, 0.45, h=0.20)], ANCHO, ALTO, CFG) is None


def test_una_persona_muy_chica_no_da_recorte():
    chica = (300.0, 300.0, 330.0, 350.0)  # 50 px de alto, bajo el minimo de 64
    assert recorte_de(chica, [_guante(0.31, 0.32, h=0.008)], ANCHO, ALTO, CFG) is None


def test_la_banda_de_proporcion_es_configurable():
    cfg = ConfigRecortes(ratio_max=0.60)
    assert recorte_de(PERSONA, [_guante(0.40, 0.45, h=0.20)], ANCHO, ALTO, cfg) is not None


# -- varios guantes ---------------------------------------------------------


def test_los_dos_guantes_de_una_persona_entran_juntos():
    res = recorte_de(PERSONA, [_guante(0.36, 0.45), _guante(0.45, 0.45)], ANCHO, ALTO, CFG)
    assert res is not None and len(res[1]) == 2


def test_si_uno_de_los_guantes_rompe_la_banda_se_descarta_todo():
    # No se queda con el bueno: si hay un guante desproporcionado, el recorte entero es de
    # un encuadre que no queremos ensenar.
    res = recorte_de(PERSONA, [_guante(0.36, 0.45), _guante(0.45, 0.45, h=0.20)],
                     ANCHO, ALTO, CFG)
    assert res is None


# -- nombres ----------------------------------------------------------------


def test_un_nombre_corto_queda_igual():
    assert _nombre_corto("pelea_mp4-12") == "pelea_mp4-12"


def test_un_nombre_largo_se_acorta_y_sigue_siendo_unico():
    # El dataset trae nombres de scraping de hasta 200 caracteres y el sistema de archivos
    # los rechaza al sumarle el sufijo del recorte.
    a, b = "x" * 300 + "uno", "x" * 300 + "dos"
    ca, cb = _nombre_corto(a), _nombre_corto(b)
    assert len(ca) < 80 and ca != cb


def test_el_nombre_corto_es_determinista():
    largo = "y" * 250
    assert _nombre_corto(largo) == _nombre_corto(largo)
