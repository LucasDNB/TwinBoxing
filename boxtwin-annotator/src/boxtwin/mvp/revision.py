"""
BoxTwin - La revision de identidad: tramos, correcciones por rango y la cola de lo dudoso.

POR QUE EXISTE
  Hasta el 2-11 la revision de la web mostraba solo los tracks SIN ASIGNAR (`tramos.py`). Los
  asignados con el rol equivocado no aparecian nunca: en C1 eran 20 de 105 tracks, y la API
  ni siquiera aceptaba corregirlos. Un rol equivocado se dibuja sobre un cuerpo y se ve
  perfecto, que es justo la asimetria que el modulo de identidad dice cuidar. Ademas la
  unidad de correccion era el track entero, y 18 de esos 20 errores son tracks mixtos (el
  tracker paso de una persona a la otra): ni a mano se podian arreglar.

  Spec: docs/specs/2026-11-02-identidad-estable/00-README.md. Lo que se exige es que ningun
  cuadro con el rol equivocado llegue a la Fight-Card sin haber pasado por los ojos de la
  persona: lo que la maquina no sabe lo pregunta, lo que sospecha lo muestra.

QUE HACE
  La unidad pasa a ser el TRAMO: (track, desde, hasta) en cuadros, [desde, hasta).
  tramos_de_propuesta   la propuesta de `proponer` (con o sin cortes de E11) como tramos
  aplicar_correcciones  las correcciones de la persona por rango, encima de lo automatico
  conflictos            pares de tramos con el mismo rol en el mismo cuadro medido
  degradar_conflictos   les saca el rol a los automaticos de un conflicto (modo estricto)
  cola_de_revision      lo dudoso, en orden: conflicto, color que contradice, posible
                        mezcla, y despues los sin asignar de siempre
  a_propuesta           los tramos finales de vuelta a una Propuesta, para escribirlos con
                        AsignarIdentidadAuto sin tocarlo
  Ningun umbral es nuevo: todos son parametros de ConfigIdentidadAuto.

USO
  from boxtwin.mvp.revision import (tramos_de_propuesta, aplicar_correcciones,
      degradar_conflictos, cola_de_revision, a_propuesta)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from boxtwin.core import apariencia as _apariencia
from boxtwin.core.deteccion_guantes import distancia_color
from boxtwin.core.identidad_auto import (
    ConfigIdentidadAuto,
    EvidenciaTrack,
    Propuesta,
    _mejor_corte,
    _sub_evidencia,
)
from boxtwin.core.types import TrackRole
from boxtwin.mvp.tramos import DURACION_MINIMA_S, motivo_de

__all__ = [
    "MOTIVOS_SOSPECHA", "PRIORIDAD", "Correccion", "Tramo", "a_propuesta",
    "aplicar_correcciones", "cola_de_revision", "conflictos", "correcciones_de_sesion",
    "degradar_conflictos", "tramos_de_propuesta",
]

# Lo que la cola le dice a la web, ademas de los cuatro motivos de sin asignar de tramos.py.
MOTIVOS_SOSPECHA = ("conflicto", "color_contradice", "posible_mezcla",
                    "apariencia_contradice", "apariencia_mezcla")
PRIORIDAD = {"conflicto": 1, "color_contradice": 2, "apariencia_contradice": 2,
             "posible_mezcla": 3, "apariencia_mezcla": 3}   # el resto, 4

_ROL_DE = {TrackRole.A: "A", TrackRole.B: "B", TrackRole.IGNORE: "ninguno"}
_ROL_A = {"A": TrackRole.A, "B": TrackRole.B, "ninguno": TrackRole.IGNORE}


@dataclass(frozen=True)
class Tramo:
    """Un pedazo de track con un rol. `rol` vacio es sin asignar."""

    track: int
    desde: int
    hasta: int
    rol: str                      # "A", "B", "ninguno" o "" (sin asignar)
    origen: str = "auto"          # "auto" o "manual"
    motivo: str | None = None     # por que quedo sin asignar (solo auto y rol vacio)

    @property
    def cuadros(self) -> int:
        return self.hasta - self.desde


@dataclass(frozen=True)
class Correccion:
    """Lo que dijo la persona: en este track, en este rango, es este rol."""

    track: int
    desde: int
    hasta: int
    rol: str


# ---------------------------------------------------------------------------
# De la propuesta a tramos y de vuelta
# ---------------------------------------------------------------------------


def _ev_de(evidencia: dict[int, EvidenciaTrack], track: int, desde: int, hasta: int,
           clave: int) -> EvidenciaTrack | None:
    e = evidencia.get(track)
    if e is None:
        return None
    if desde <= e.primer_frame and hasta > e.ultimo_frame:
        return e
    return _sub_evidencia(e, clave, desde, hasta)


def tramos_de_propuesta(
    prop: Propuesta,
    evidencia: dict[int, EvidenciaTrack],
    cfg: ConfigIdentidadAuto,
) -> list[Tramo]:
    """
    La propuesta como tramos. Con cortes de E11 las claves de `prop` son segmentos y
    `prop.tracks` dice de que track es cada uno; sin cortes, la clave es el track.
    Los sin asignar llevan su motivo, calculado con la evidencia del tramo.
    """
    salida: list[Tramo] = []
    for k, rol in prop.roles.items():
        ini, fin = prop.rangos[k]
        salida.append(Tramo(prop.tracks.get(k, k), ini, fin, _ROL_DE[rol]))
    for k in dict.fromkeys(prop.sin_asignar):
        if k in prop.roles:                  # aplicar_manuales viejo pudo dejarlo en los dos
            continue
        ini, fin = prop.rangos[k]
        track = prop.tracks.get(k, k)
        e = _ev_de(evidencia, track, ini, fin, k)
        motivo = motivo_de(k, {k: e} if e is not None else {}, prop, cfg)
        salida.append(Tramo(track, ini, fin, "", motivo=motivo))
    salida.sort(key=lambda t: (t.track, t.desde))
    return salida


def ignorar_cortos(tramos: list[Tramo]) -> list[Tramo]:
    """
    Los sin asignar con motivo `corto` ("salio de cuadro") pasan a ignorados, sin preguntar
    (pedido de Lucas, 3-11). Sus golpes ya no se contaban; lo que cambia es que no ocupan la
    cola. Siguen siendo automaticos: una correccion de la persona les gana, y con `todos` la
    cola los muestra para poder rescatarlos.
    """
    return [replace(t, rol="ninguno") if t.rol == "" and t.motivo == "corto" else t
            for t in tramos]


def a_propuesta(tramos: list[Tramo], base: Propuesta) -> Propuesta:
    """
    Los tramos finales como Propuesta, para AsignarIdentidadAuto (que no cambia). Cada tramo
    con rol lleva una clave sintetica negativa y su track en `tracks`, igual que los segmentos
    de E11; los sin asignar no reciben assignment, que es la decision del modulo.
    """
    p = Propuesta(roles={}, rangos={}, perfiles=dict(base.perfiles), semillas=dict(base.semillas),
                  avisos=list(base.avisos), diagnostico=dict(base.diagnostico))
    for i, t in enumerate(sorted(tramos, key=lambda t: (t.track, t.desde))):
        if t.hasta <= t.desde:
            continue
        clave = -(10_000_000 + i)
        p.tracks[clave] = t.track
        p.rangos[clave] = (t.desde, t.hasta)
        if t.rol in _ROL_A:
            p.roles[clave] = _ROL_A[t.rol]
        else:
            p.sin_asignar.append(clave)
    return p


# ---------------------------------------------------------------------------
# Correcciones
# ---------------------------------------------------------------------------


def correcciones_de_sesion(identidad: dict, extension: dict[int, tuple[int, int]],
                           fps: float) -> list[Correccion]:
    """
    Lo que hay en sesion.json -> identidad, en el orden en que se aplica.

    `asignaciones_manuales` ({track: rol}, la B3 de antes) es una correccion del track entero.
    `correcciones` ([{track, desde, hasta, rol}] en segundos) es la nueva, por rango; va
    despues, asi que si las dos tocan el mismo cuadro manda la de rango. `extension` es
    track -> [primer, ultimo+1) de los tramos automaticos: un track que no esta no es de la
    sesion y se saltea.
    """
    salida: list[Correccion] = []
    for k, rol in (identidad.get("asignaciones_manuales") or {}).items():
        t = int(k)
        if t in extension:
            salida.append(Correccion(t, *extension[t], rol))
    for c in identidad.get("correcciones") or []:
        t = int(c["track"])
        if t not in extension:
            continue
        lo, hi = extension[t]
        desde = lo if c.get("desde") is None else int(round(float(c["desde"]) * fps))
        hasta = hi if c.get("hasta") is None else int(round(float(c["hasta"]) * fps))
        salida.append(Correccion(t, desde, hasta, c["rol"]))
    for c in salida:
        if c.rol not in _ROL_A:
            raise ValueError(f"rol desconocido para el track {c.track}: {c.rol!r}")
    return salida


def extension_de(tramos: list[Tramo]) -> dict[int, tuple[int, int]]:
    ext: dict[int, tuple[int, int]] = {}
    for t in tramos:
        lo, hi = ext.get(t.track, (t.desde, t.hasta))
        ext[t.track] = (min(lo, t.desde), max(hi, t.hasta))
    return ext


def aplicar_correcciones(tramos: list[Tramo], correcciones: list[Correccion]) -> list[Tramo]:
    """
    Cada correccion manda en su rango y en nada mas. Lo automatico que la toca se recorta y
    lo que sobra a los costados queda como estaba. No se usa como semilla para el resto del
    video: es la misma regla que tenia `aplicar_manuales` (una persona mira unos segundos).
    El rango se recorta a la extension del track; una correccion vacia no hace nada.
    """
    ext = extension_de(tramos)
    actual = list(tramos)
    for c in correcciones:
        if c.track not in ext:
            continue
        lo, hi = ext[c.track]
        d, h = max(c.desde, lo), min(c.hasta, hi)
        if h <= d:
            continue
        nuevo: list[Tramo] = []
        for t in actual:
            if t.track != c.track or t.hasta <= d or t.desde >= h:
                nuevo.append(t)
                continue
            if t.desde < d:
                nuevo.append(replace(t, hasta=d))
            if t.hasta > h:
                nuevo.append(replace(t, desde=h))
        nuevo.append(Tramo(c.track, d, h, c.rol, origen="manual"))
        actual = nuevo
    actual.sort(key=lambda t: (t.track, t.desde))
    return actual


# ---------------------------------------------------------------------------
# Sospechas
# ---------------------------------------------------------------------------


def _medidos(evidencia: dict[int, EvidenciaTrack], t: Tramo) -> set[int]:
    """Los cuadros donde el track se midio (un recorte cada `paso`), adentro del tramo."""
    e = evidencia.get(t.track)
    if e is None:
        return set()
    return {f for f, _ in e.xs if t.desde <= f < t.hasta}


def conflictos(tramos: list[Tramo], evidencia: dict[int, EvidenciaTrack],
               cfg: ConfigIdentidadAuto) -> list[tuple[int, int]]:
    """
    Pares de indices de `tramos` con el mismo rol (A o B), de tracks distintos, que se
    midieron en el mismo cuadro al menos `min_coexistencia` veces. Una persona no esta dos
    veces en el mismo cuadro: uno de los dos esta mal (o el tracker duplico a alguien, que
    tambien hay que mirar).
    """
    pares: list[tuple[int, int]] = []
    for rol in ("A", "B"):
        idx = [i for i, t in enumerate(tramos) if t.rol == rol]
        med = {i: _medidos(evidencia, tramos[i]) for i in idx}
        for a_i, i in enumerate(idx):
            if not med[i]:
                continue
            for j in idx[a_i + 1:]:
                if tramos[i].track == tramos[j].track:
                    continue
                if tramos[j].hasta <= tramos[i].desde or tramos[j].desde >= tramos[i].hasta:
                    continue
                if len(med[i] & med[j]) >= cfg.min_coexistencia:
                    pares.append((i, j))
    return pares


def degradar_conflictos(tramos: list[Tramo], evidencia: dict[int, EvidenciaTrack],
                        cfg: ConfigIdentidadAuto) -> tuple[list[Tramo], int]:
    """
    Modo estricto: los tramos AUTOMATICOS de un conflicto pierden el rol. Lo que decidio una
    persona no se toca. Devuelve los tramos y cuantos se degradaron.
    """
    marcados = {i for par in conflictos(tramos, evidencia, cfg) for i in par
                if tramos[i].origen == "auto"}
    salida = [replace(t, rol="", motivo="conflicto") if i in marcados else t
              for i, t in enumerate(tramos)]
    return salida, len(marcados)


def _serie_color(evidencia, t: Tramo, pa, pb):
    e = evidencia.get(t.track)
    if e is None:
        return [], []
    muestras = sorted((f, c) for f, c in e.colores if t.desde <= f < t.hasta)
    return ([f for f, _ in muestras],
            [distancia_color(c, pb) - distancia_color(c, pa) for _, c in muestras])


def _sospecha_de(t: Tramo, evidencia, perfiles: dict, cfg: ConfigIdentidadAuto,
                apariencia=None) -> str | None:
    """
    Color que contradice o posible mezcla, para un tramo automatico con rol A o B. Con el
    modelo de `apariencia` (IE7), si el color del guante no dijo nada, mira guante, remera y
    pantalon ponderados por lo que separa cada uno en este video.
    """
    if t.origen != "auto" or t.rol not in ("A", "B"):
        return None
    motivo = _sospecha_por_color(t, evidencia, perfiles, cfg)
    if motivo is None and apariencia is not None:
        motivo = _apariencia.sospecha(apariencia, evidencia.get(t.track), t.rol,
                                      t.desde, t.hasta, cfg)
    return motivo


def _sospecha_por_color(t: Tramo, evidencia, perfiles: dict,
                        cfg: ConfigIdentidadAuto) -> str | None:
    pa, pb = perfiles.get("fighter_A"), perfiles.get("fighter_B")
    if pa is None or pb is None or distancia_color(pa, pb) < cfg.separacion_minima:
        return None
    frames, s = _serie_color(evidencia, t, pa, pb)
    n = len(s)
    if n < cfg.min_guantes_voto:
        return None
    # s > 0 es "mas A". Contradice si la mayoria de los guantes esta del otro lado.
    a = sum(1 for v in s if v > 0)
    b = sum(1 for v in s if v < 0)
    if (t.rol == "A" and b > a) or (t.rol == "B" and a > b):
        return "color_contradice"
    if n >= 2 * cfg.min_guantes_voto and _mejor_corte(
            frames, s, cfg.min_guantes_voto, cfg.margen_orientacion / 2) is not None:
        return "posible_mezcla"
    return None


def _cuadro_central(evidencia, t: Tramo) -> int:
    """El cuadro medido mas cercano al centro del tramo: ahi el track seguro esta."""
    medio = (t.desde + t.hasta - 1) // 2
    fs = sorted(_medidos(evidencia, t))
    if not fs:
        return medio
    return min(fs, key=lambda f: (abs(f - medio), f))


def clave_de(t: Tramo, ext: dict[int, tuple[int, int]]) -> str:
    """'37' si el tramo es el track entero, '37-1200' si es un pedazo."""
    return str(t.track) if ext.get(t.track) == (t.desde, t.hasta) else f"{t.track}-{t.desde}"


def cola_de_revision(
    auto: list[Tramo],
    corregidos: list[Tramo],
    evidencia: dict[int, EvidenciaTrack],
    perfiles: dict,
    cfg: ConfigIdentidadAuto,
    fps: float,
    todos: bool = True,
    estricta: bool = False,
    apariencia=None,
) -> list[dict]:
    """
    Lo que la persona tiene que mirar, en orden.

    Los items salen de los tramos AUTOMATICOS (antes de las correcciones), para que la lista
    no se achique al decidir y una decision se pueda cambiar; cada item lleva al lado lo que
    ya decidio la persona en su rango (`asignado` si una sola decision lo cubre, `partes` si
    lo partio). Los conflictos se buscan en `corregidos` (despues de las correcciones y ANTES
    de degradar): una correccion puede crear uno o resolverlo. Con `estricta`, un tramo
    automatico en conflicto sale como sin asignar, que es como quedo en la Fight-Card.
    Con `apariencia` (un ModeloApariencia de boxtwin.core.apariencia, IE7), los asignados se
    miran tambien con guante, remera y pantalon: suma `apariencia_contradice` y
    `apariencia_mezcla`. Sin el, la cola es la de antes.
    Con `todos`, la lista suma los asignados sin sospecha (tipo "asignado") al final, para que
    la web pueda mostrar todo lo que se quiera revisar a mano.

    `revision_pendiente` de cada item es verdadero si es sospechoso o sin asignar y nadie
    decidio nada en su rango.
    """
    ext = extension_de(auto)
    manuales = [t for t in corregidos if t.origen == "manual"]
    # Un conflicto en los corregidos se marca en el tramo automatico que lo contiene. Si los
    # dos lados son decisiones de la persona, no se le discute.
    zonas_conflicto = [corregidos[i] for par in conflictos(corregidos, evidencia, cfg)
                       for i in par if corregidos[i].origen == "auto"]

    def toca(a: Tramo, b: Tramo) -> bool:
        return a.track == b.track and a.desde < b.hasta and b.desde < a.hasta

    items = []
    for t in auto:
        dur = (t.hasta - t.desde) / fps if fps else 0.0
        if t.rol in ("A", "B") and any(toca(t, z) for z in zonas_conflicto):
            motivo = "conflicto"
            tipo = "sin_asignar" if estricta else "sospechoso"
        elif t.rol == "":
            if dur <= DURACION_MINIMA_S:
                continue
            motivo, tipo = t.motivo or "sin_coexistencia", "sin_asignar"
        elif t.rol == "ninguno":
            # Solo los que ignoro `ignorar_cortos`, y solo en la lista completa.
            if t.motivo != "corto" or not todos or dur <= DURACION_MINIMA_S:
                continue
            motivo, tipo = "corto", "asignado"
        else:
            motivo = _sospecha_de(t, evidencia, perfiles, cfg, apariencia)
            tipo = "sospechoso" if motivo else "asignado"
            if motivo is None and (not todos or dur <= DURACION_MINIMA_S):
                continue
        partes = [m for m in manuales if toca(t, m)]
        cubre = (len(partes) == 1 and partes[0].desde <= t.desde and partes[0].hasta >= t.hasta)
        items.append({
            "clave": clave_de(t, ext),
            "track": int(t.track),
            "desde": int(t.desde),
            "hasta": int(t.hasta),
            "t_desde": round(t.desde / fps, 2) if fps else 0.0,
            "t_hasta": round(t.hasta / fps, 2) if fps else 0.0,
            "cuadro": int(_cuadro_central(evidencia, t)),
            "tipo": tipo,
            "motivo": motivo,
            "rol_actual": (None if tipo == "sin_asignar" else t.rol) or None,
            "asignado": partes[0].rol if cubre else None,
            "partes": [] if cubre else [
                {"t_desde": round(max(m.desde, t.desde) / fps, 2) if fps else 0.0,
                 "t_hasta": round(min(m.hasta, t.hasta) / fps, 2) if fps else 0.0,
                 "rol": m.rol} for m in partes],
            "revision_pendiente": tipo != "asignado" and not partes,
            "recorte": None,
        })
    items.sort(key=lambda x: (PRIORIDAD.get(x["motivo"], 4) if x["tipo"] != "asignado" else 9,
                              -(x["hasta"] - x["desde"]), x["track"], x["desde"]))
    return items
