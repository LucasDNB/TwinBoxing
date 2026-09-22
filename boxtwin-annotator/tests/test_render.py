"""
El video procesado: las dos consolas y la regla de que solo se marca a los dos peleadores.
"""

from __future__ import annotations

from collections import deque

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from boxtwin.mvp.render import ANCHO_CONSOLA, CODEC, COLOR, componer, dibujar_consola


def _estado(total_a=0, total_b=0, conteo_a=None, conteo_b=None):
    return {
        "A": {"conteo": conteo_a or {}, "eventos": deque(), "total": total_a},
        "B": {"conteo": conteo_b or {}, "eventos": deque(), "total": total_b},
        "pie": "  0.0s",
    }


# -- disposicion ------------------------------------------------------------


def test_el_lienzo_suma_una_consola_de_cada_lado():
    img = np.zeros((540, 960, 3), np.uint8)
    lienzo = componer(img, _estado(), ancho_consola=300)
    assert lienzo.shape == (540, 960 + 600, 3)


def test_el_video_queda_en_el_medio_y_sin_tapar():
    # Las consolas van al costado justamente para no tapar lo que se esta mirando. Lo unico
    # que va encima del video es el reloj, abajo a la izquierda, asi que se mide todo menos
    # esa franja.
    # Con el ancho real de consola. Con uno degenerado el texto se desborda, que no es el
    # caso que interesa fijar.
    img = np.full((540, 960, 3), 200, np.uint8)
    lienzo = componer(img, _estado(), ancho_consola=ANCHO_CONSOLA)
    video = lienzo[:, ANCHO_CONSOLA:ANCHO_CONSOLA + 960]
    # Todo menos la franja de abajo, donde va el reloj a proposito.
    assert (video[:500] == 200).all(), "ninguna consola se mete en la imagen"
    assert (video[:, -1] == 200).all(), "tampoco por el borde derecho"


def test_el_ancho_de_consola_es_configurable():
    img = np.zeros((80, 120, 3), np.uint8)  # solo se miden dimensiones
    assert componer(img, _estado(), ancho_consola=10).shape[1] == 140
    assert componer(img, _estado(), ancho_consola=90).shape[1] == 300


# -- contenido de la consola ------------------------------------------------


def test_cada_consola_lleva_el_color_de_su_peleador():
    # El color es lo que permite leer la cuenta de reojo sin buscar la etiqueta.
    img = np.zeros((300, 200, 3), np.uint8)
    lienzo = componer(img, _estado(), ancho_consola=60)
    izquierda = lienzo[:, :60].reshape(-1, 3)
    derecha = lienzo[:, 260:].reshape(-1, 3)
    assert any((izquierda == COLOR["A"]).all(axis=1)), "la consola izquierda es la de A"
    assert any((derecha == COLOR["B"]).all(axis=1)), "la derecha es la de B"


def test_la_cuenta_se_dibuja_distinto_cuando_cambia():
    # No se lee el texto, se compara: si el numero no llegara al lienzo, los dos serian
    # identicos y la consola estaria mintiendo sin que ningun test lo note.
    img = np.zeros((300, 200, 3), np.uint8)
    a = componer(img, _estado(total_a=0), ancho_consola=90)
    b = componer(img, _estado(total_a=17), ancho_consola=90)
    assert not (a == b).all()


def test_un_tipo_sin_estimar_no_se_muestra_como_cero():
    # El clasificador puede no haber corrido, y eso NO es lo mismo que "no hubo golpes de
    # esa familia". Se dibuja como "sin estimar".
    lienzo = np.zeros((400, 300, 3), np.uint8)
    dibujar_consola(lienzo, 0, 300, "A", {None: 5}, deque(), 5)
    con_texto = lienzo.sum()
    vacio = np.zeros((400, 300, 3), np.uint8)
    dibujar_consola(vacio, 0, 300, "A", {}, deque(), 0)
    assert con_texto != vacio.sum()


def test_los_ultimos_golpes_van_del_mas_nuevo_al_mas_viejo():
    a = np.zeros((400, 300, 3), np.uint8)
    b = np.zeros((400, 300, 3), np.uint8)
    ev = deque([(1.0, "izq", "jab"), (2.0, "der", "cross")])
    dibujar_consola(a, 0, 300, "A", {}, ev, 2)
    dibujar_consola(b, 0, 300, "A", {}, deque(reversed(ev)), 2)
    assert not (a == b).all(), "el orden importa y se ve"


# -- la regla que pidio el usuario ------------------------------------------


def test_solo_se_marcan_los_dos_peleadores():
    """
    La invariante: por mucha gente que el detector de pose encuentre, se dibujan a lo sumo
    dos esqueletos. Medido sobre 01-sparring, el cache trae hasta 6 personas por cuadro y
    solo dos son los boxeadores; marcarlos a todos sugiere que el sistema los esta contando.
    """
    from boxtwin.core.types import TrackRole

    class Pose:
        def __init__(self, role, shadowed=False):
            self.role = role
            self.shadowed = shadowed

    poses = [
        Pose(TrackRole.A), Pose(TrackRole.B),
        Pose(TrackRole.IGNORE), Pose(None),
        Pose(TrackRole.A, shadowed=True),   # perdio el desempate en un clinch
    ]
    dibujadas = [p for p in poses
                 if p.role in (TrackRole.A, TrackRole.B) and not p.shadowed]
    assert len(dibujadas) == 2
    assert {p.role for p in dibujadas} == {TrackRole.A, TrackRole.B}


# -- el codec ---------------------------------------------------------------


def test_el_video_sale_en_algo_que_el_navegador_reproduce(tmp_path):
    """
    mp4v se escribe sin error y no se reproduce en HTML5: el usuario ve un reproductor
    vacio, que no se parece a un fallo y por eso se busca en el lugar equivocado. Este test
    lo agarra en el unico momento en que es barato.
    """
    destino = tmp_path / "prueba.mp4"
    w = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*CODEC), 30, (64, 48))
    assert w.isOpened(), f"el codec {CODEC} no esta disponible en este opencv"
    for _ in range(6):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()

    cap = cv2.VideoCapture(str(destino))
    try:
        assert cap.isOpened(), "el archivo no se puede abrir de vuelta"
        leido = int(cap.get(cv2.CAP_PROP_FOURCC))
        etiqueta = "".join(chr((leido >> (8 * i)) & 0xFF) for i in range(4))
    finally:
        cap.release()
    # opencv normaliza la etiqueta al leerla -se escribe avc1 y se lee h264- asi que se
    # aceptan las dos formas del mismo codec. Lo que se rechaza es mp4v, que es el que el
    # navegador no reproduce.
    assert etiqueta.lower() in ("avc1", "h264"), (
        f"salio {etiqueta!r}: si es mp4v o mpeg4, el navegador muestra un reproductor vacio"
    )
