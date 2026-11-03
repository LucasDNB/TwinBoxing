"""
BoxTwin - Features por cuadro y por brazo.

POR QUE EXISTE
  El clasificador PoseConv3D no generalizo a una fuente nueva: los tres folds que dejan una
  fuente afuera quedaron en o por debajo de su linea de base. La sospecha razonable es que
  aprendio la camara y la escala, porque trabaja sobre volumenes de heatmaps en pixeles.

  Aca las features se construyen para que esa via este cerrada por diseno:

  - CENTRADAS en el punto medio de los hombros, asi la posicion en el cuadro no entra.
  - ESCALADAS por max(ancho de hombros, largo del torso), asi la distancia a camara no entra.

    NO por el ancho de hombros solo, que fue el primer intento y estaba roto. La guardia de
    boxeo es de perfil casi siempre, y de perfil los hombros se superponen en la proyeccion:
    su ancho medido va de 16 px en el percentil 1 a 142 en la mediana, un factor de nueve.
    Dividir por eso hace explotar todas las features. Medido sobre las tres fuentes, entre
    el 19% y el 37% de los cuadros daban una extension de muneca mayor a 2 anchos de hombro,
    que es fisicamente imposible, con maximos de 354, 59 y 680.

    El largo del torso no colapsa al girar sobre el eje vertical, que es el giro que hace un
    boxeador: su razon percentil 99 sobre percentil 1 es 3,0 contra 20,2 la del ancho de
    hombros. Tomando el maximo de los dos, una sola medicion degenerada no puede arruinar la
    normalizacion.

    Y no es solo estabilidad: la extension normalizada asi SEPARA MEJOR. AUC por cuadro de
    0,657 a 0,727 en sparring-3, 0,577 a 0,584 en Sparring y 0,639 a 0,744 en Pacquiao.
  - ESPEJADAS por brazo, asi un jab y un cross se ven iguales. Detectar no necesita saber
    que brazo es, y espejar duplica los datos efectivos.

  Es la pieza del sistema con mejor chance de sobrevivir al leave-one-source-out, porque la
  senal que busca es geometrica y normalizable. Lo que NO arregla es la familia del golpe:
  eso esta medido tres veces que no es separable en pose monocular, y el detector no lo
  intenta.

QUE HACE
  De (T, 17, 2) keypoints COCO y (T, 17) scores, saca (T, 20) features y una mascara de
  cuadros usables, para un brazo dado.

  Los angulos que dan la vuelta se codifican como (coseno, seno) y no como el angulo: a
  -pi y a +pi el cuerpo esta en la misma pose, y un escalar que salta de uno a otro le
  ensena al modelo un borde que no existe.

  El umbral de score es 0,3 para todas las articulaciones salvo que se pase `umbrales`: un
  numero (el mismo para todas) o un dict por articulacion (nariz, hombro, codo, muneca,
  cadera). Existe por E8b (docs/specs/2026-10-30-mejoras-post-e8/01): el 0,3 esta pensado
  para como puntua YOLOv8, y otro estimador reparte sus scores distinto. La nariz se acepta
  en el dict para que la calibracion sea de las cinco, pero no entra en la mascara, igual
  que hoy: con el umbral en 0,3 todo da lo mismo que antes, bit a bit.

  Con el guante detectado (E12, docs/specs/2026-10-30-mejoras-post-e8/04): si el esqueleto
  trae 19 keypoints, el 17 y el 18 son los centros de los guantes izquierdo y derecho, con la
  confianza del detector de guantes como score (0 = no hay guante). Se espejan como un par
  mas, asi que el brazo derecho no necesita una rama aparte. Suman 5 features por brazo
  (NOMBRES_GUANTE) y cambian la mascara: un cuadro con la muneca o el codo bajo umbral sirve
  si hay guante en esa mano, y entonces las features de muneca y codo van en 0 y la mano la
  lleva el guante. Con 17 keypoints todo es como antes.

USO
  f, ok = features_de(kp, sc, brazo="left")
  f, ok = features_de(kp, sc, brazo="left", umbrales={"nariz": .2, "hombro": .25, ...})
"""

from __future__ import annotations

import numpy as np

__all__ = ["NOMBRES", "N_FEATURES", "NOMBRES_GUANTE", "L_GUA", "R_GUA", "features_de", "espejar", "MIN_SCORE", "ARTICULACIONES",
           "resolver_umbrales"]

# COCO17
NARIZ = 0
L_HOM, R_HOM = 5, 6
L_COD, R_COD = 7, 8
L_MUN, R_MUN = 9, 10
L_CAD, R_CAD = 11, 12
# E12: el guante entra como dos keypoints mas
L_GUA, R_GUA = 17, 18

MIN_SCORE = 0.3
_EPS = 1e-6

# Las cinco que usan las features; izquierda y derecha comparten umbral.
ARTICULACIONES = ("nariz", "hombro", "codo", "muneca", "cadera")


def resolver_umbrales(umbrales=None) -> dict[str, float]:
    """None, un numero o un dict con las cinco articulaciones -> dict con las cinco."""
    if umbrales is None:
        return {a: MIN_SCORE for a in ARTICULACIONES}
    if isinstance(umbrales, (int, float)):
        return {a: float(umbrales) for a in ARTICULACIONES}
    if set(umbrales) != set(ARTICULACIONES):
        raise ValueError(f"umbrales tiene que traer exactamente {ARTICULACIONES}, trae {sorted(umbrales)}")
    return {a: float(umbrales[a]) for a in ARTICULACIONES}

# Un brazo humano estirado no llega a 2 desde el centro de hombros, ni con la escala mas
# favorable. Es una red de contencion: con la escala actual no descarta practicamente nada
# (el maximo medido es 2,4), pero deja que una pose rota entre en la mascara y no en la
# perdida, en vez de meter un valor de 300 en la entrada del modelo.
EXTENSION_MAX = 3.0

# Un espejo real no es solo negar la x: lo que era el hombro izquierdo de la persona aparece
# del otro lado y sigue siendo su hombro izquierdo, asi que los pares tambien se permutan.
# Sin la permutacion, el vector de hombros queda con el signo cambiado y el brazo derecho no
# se ve como un izquierdo, que es todo el objetivo.
PARES_LR = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (L_GUA, R_GUA)]


def espejar(kp: np.ndarray, sc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Refleja el esqueleto: niega la x y permuta los pares izquierda/derecha."""
    kp, sc = kp.copy(), sc.copy()
    kp[..., 0] *= -1
    for a, b in PARES_LR:
        if b >= kp.shape[1]:        # sin guantes el esqueleto tiene 17
            continue
        kp[:, [a, b]] = kp[:, [b, a]]
        sc[:, [a, b]] = sc[:, [b, a]]
    return kp, sc

NOMBRES = [
    "muneca_x", "muneca_y",
    "codo_x", "codo_y",
    "extension",
    "angulo_codo",
    "hombros_cos", "hombros_sin",
    "caderas_cos", "caderas_sin",
    "torsion_cos", "torsion_sin",
    "otra_muneca_x", "otra_muneca_y",
    "nariz_y",
    "d_muneca_x", "d_muneca_y", "d_extension", "d_angulo_codo",
    "dd_extension",
]
N_FEATURES = len(NOMBRES)

# Las 5 del guante, que van despues de las 20 cuando el esqueleto trae 19 keypoints.
NOMBRES_GUANTE = ["guante_x", "guante_y", "guante_ok", "d_guante_x", "d_guante_y"]

# Las que salen de la muneca o del codo del brazo que actua: con el guante llevando la mano
# van en 0, porque su keypoint esta bajo umbral.
_DE_MUNECA_Y_CODO = [NOMBRES.index(n) for n in (
    "muneca_x", "muneca_y", "codo_x", "codo_y", "extension", "angulo_codo",
    "d_muneca_x", "d_muneca_y", "d_extension", "d_angulo_codo", "dd_extension")]


def _angulo(v: np.ndarray) -> np.ndarray:
    """Angulo de un vector (T, 2), como (cos, sin). Sin saltos en +-pi."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, _EPS)


def features_de(
    kp: np.ndarray, sc: np.ndarray, brazo: str, umbrales=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Features del brazo `brazo` ("left" o "right") y mascara de cuadros usables.

    El brazo derecho se resuelve espejando el esqueleto entero y leyendo despues el
    izquierdo. Asi la invariancia queda por construccion y no por dos ramas de codigo que
    hay que mantener de acuerdo.

    Un cuadro es usable si los dos hombros, el codo y la muneca del brazo que actua tienen
    score suficiente y los hombros no estan colapsados. Los cuadros no usables van en cero,
    pero eso no los hace fondo: la mascara es lo que los saca de la perdida.

    Los cuadros vecinos a uno no usable tambien se marcan, porque las derivadas se calculan
    por diferencias y ahi cruzan el agujero.
    """
    if brazo not in ("left", "right"):
        raise ValueError(f"brazo invalido: {brazo!r}")
    kp = np.asarray(kp, np.float64)
    sc = np.asarray(sc, np.float64)
    u = resolver_umbrales(umbrales)
    if brazo == "right":
        kp, sc = espejar(kp, sc)
    T = len(kp)

    centro = (kp[:, L_HOM] + kp[:, R_HOM]) / 2
    ancho = np.linalg.norm(kp[:, L_HOM] - kp[:, R_HOM], axis=-1)
    torso = np.linalg.norm(centro - (kp[:, L_CAD] + kp[:, R_CAD]) / 2, axis=-1)
    caderas_ok = np.minimum(sc[:, L_CAD], sc[:, R_CAD]) >= u["cadera"]
    escala = np.where(caderas_ok, np.maximum(ancho, torso), ancho)

    cuerpo_ok = (
        (np.minimum(sc[:, L_HOM], sc[:, R_HOM]) >= u["hombro"])
        & caderas_ok
        & (escala > _EPS)
    )
    brazo_ok = (sc[:, L_COD] >= u["codo"]) & (sc[:, L_MUN] >= u["muneca"])

    rel = (kp - centro[:, None, :]) / np.maximum(escala, _EPS)[:, None, None]

    muneca = rel[:, L_MUN]
    codo = rel[:, L_COD]
    hombro = rel[:, L_HOM]
    extension = np.linalg.norm(muneca, axis=-1)

    # angulo del codo entre hombro->codo y codo->muneca, en [0, pi]. No da la vuelta, asi
    # que va como escalar y no como (cos, sin).
    a = codo - hombro
    b = muneca - codo
    cos = (a * b).sum(-1) / np.maximum(
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), _EPS
    )
    angulo_codo = np.arccos(np.clip(cos, -1.0, 1.0))

    hom = _angulo(rel[:, L_HOM] - rel[:, R_HOM])
    cad = _angulo(rel[:, L_CAD] - rel[:, R_CAD])
    # torsion = hombros menos caderas, compuesta como rotacion: (cos(a-b), sin(a-b))
    tor = np.stack(
        [
            hom[:, 0] * cad[:, 0] + hom[:, 1] * cad[:, 1],
            hom[:, 1] * cad[:, 0] - hom[:, 0] * cad[:, 1],
        ],
        axis=-1,
    )

    def d(x: np.ndarray) -> np.ndarray:
        return np.gradient(x) if T > 1 else np.zeros_like(x)

    f = np.stack(
        [
            muneca[:, 0], muneca[:, 1],
            codo[:, 0], codo[:, 1],
            extension,
            angulo_codo,
            hom[:, 0], hom[:, 1],
            cad[:, 0], cad[:, 1],
            tor[:, 0], tor[:, 1],
            rel[:, R_MUN, 0], rel[:, R_MUN, 1],
            rel[:, NARIZ, 1],
            d(muneca[:, 0]), d(muneca[:, 1]), d(extension), d(angulo_codo),
            d(d(extension)),
        ],
        axis=-1,
    )
    brazo_ok &= extension <= EXTENSION_MAX
    if kp.shape[1] > L_GUA:
        f, ok = _con_guante(f, rel, sc, cuerpo_ok, brazo_ok, d)
    else:
        ok = cuerpo_ok & brazo_ok
    f[~ok] = 0.0
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    # las derivadas cruzan los agujeros: el vecino de un cuadro no usable tampoco lo es
    usable = ok.copy()
    if T > 1:
        usable[:-1] &= ok[1:]
        usable[1:] &= ok[:-1]
    return f.astype(np.float32), usable


def _con_guante(f, rel, sc, cuerpo_ok, brazo_ok, d):
    """
    Las 5 features del guante y la mascara de E12 §3.

    La derivada del guante no cruza un cuadro sin guante, con la misma regla de vecinos que
    la mascara: si falta en t-1 o en t+1, va en 0 en t.
    """
    hay = sc[:, L_GUA] > 0
    g = rel[:, L_GUA]
    vecinos = hay.copy()
    if len(hay) > 1:
        vecinos[:-1] &= hay[1:]
        vecinos[1:] &= hay[:-1]
    dg = np.stack([d(g[:, 0]), d(g[:, 1])], -1)
    dg[~vecinos] = 0.0
    fg = np.concatenate([g, hay[:, None].astype(np.float64), dg], -1)
    fg[~hay] = 0.0
    usa_guante = ~brazo_ok & hay
    f = np.concatenate([f, fg], -1)
    f[np.ix_(usa_guante, _DE_MUNECA_Y_CODO)] = 0.0
    return f, cuerpo_ok & (brazo_ok | hay)
