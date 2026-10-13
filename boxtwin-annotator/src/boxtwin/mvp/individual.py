"""
BoxTwin - Identidad de una sesion de un solo peleador: bolsa y sombra.

POR QUE EXISTE
  La identidad del producto esta pensada para dos personas que se separan por coexistencia
  y por color de guante. En bolsa o sombra hay una sola, y la pregunta cambia: no es "cual de
  los dos" sino "cuales de estos tracks son el, y cuales son el que pasa por atras". Se
  responde con lo mismo que ya se mide, con una sola semilla.

  Esto NO esta medido y por eso la bandera `sesiones_individuales` esta apagada. Antes de
  encenderla, el spec fija el criterio: sobre dos videos propios de bolsa o sombra, la
  persona queda asignada en al menos el 95% de los cuadros en que esta en cuadro, mirado a
  ojo sobre el video procesado (prototipo/specs/05-api-y-datos.md, B8).

QUE HACE
  Con la semilla como perfil de color: un track que coexiste con uno ya asignado es otra
  persona (dos A al mismo tiempo no existen); uno que no coexiste y tiene guantes del color
  del perfil es el; el resto, ninguno. Los que no alcanzan para votar quedan sin asignar.

USO
  from boxtwin.mvp.individual import proponer_individual
  prop = proponer_individual(evidencia, cfg, total_frames, fps, semilla=12)
"""

from __future__ import annotations

from boxtwin.core.deteccion_guantes import distancia_color, perfil_de
from boxtwin.core.identidad_auto import Propuesta
from boxtwin.core.types import TrackRole

__all__ = ["proponer_individual"]


def proponer_individual(evidencia: dict, cfg, total_frames: int, fps: float, semilla: int) -> Propuesta:
    if semilla not in evidencia:
        raise ValueError(f"la semilla {semilla} no es un track del cache; hay {len(evidencia)} tracks medidos")
    prop = Propuesta(roles={}, rangos={})
    for tid, e in evidencia.items():
        prop.rangos[tid] = (e.primer_frame, min(e.ultimo_frame + 1, total_frames))

    colores_semilla = [c for _, c in evidencia[semilla].colores]
    perfil = perfil_de(colores_semilla) if colores_semilla else None
    alto_maximo = max(e.alto_max for e in evidencia.values())
    umbral_alto = cfg.fraccion_altura * alto_maximo

    # Cuadros en que ya hay alguien asignado: lo que coexiste con eso es otra persona.
    ocupados: set[int] = set(range(*prop.rangos[semilla]))
    prop.roles[semilla] = TrackRole.A
    distancias: dict[int, float] = {}
    # Del mas largo al mas corto: los tramos largos fijan la ocupacion antes que los fragmentos.
    orden = sorted((t for t in evidencia if t != semilla),
                   key=lambda t: -(prop.rangos[t][1] - prop.rangos[t][0]))
    for t in orden:
        e = evidencia[t]
        cuadros = set(range(*prop.rangos[t]))
        if len(cuadros & ocupados) >= cfg.min_coexistencia:
            prop.roles[t] = TrackRole.IGNORE          # en cuadro a la vez que el: es otro
            continue
        if e.alto_max < umbral_alto:
            prop.roles[t] = TrackRole.IGNORE
            continue
        propios = [c for _, c in e.colores]
        if perfil is None or len(propios) < cfg.min_guantes_voto:
            prop.sin_asignar.append(t)
            continue
        d = distancia_color(perfil_de(propios), perfil)
        distancias[t] = round(d, 3)
        if d < cfg.separacion_minima:
            prop.roles[t] = TrackRole.A
            ocupados |= cuadros
        else:
            prop.roles[t] = TrackRole.IGNORE

    prop.semillas = {"fighter_A": semilla}
    if perfil is not None:
        prop.perfiles = {"fighter_A": perfil}
    prop.diagnostico = {
        "tracks": len(evidencia),
        "siembra": "humana_individual",
        "asignados_A": sum(1 for r in prop.roles.values() if r is TrackRole.A),
        "asignados_B": 0,
        "sin_asignar": len(prop.sin_asignar),
        "distancias_al_perfil": distancias,
        "separacion_de_perfiles": None,
    }
    if perfil is None:
        prop.avisos.append(
            "la semilla no tiene cuadros con color de guante: solo se asigna el track marcado"
        )
    return prop
