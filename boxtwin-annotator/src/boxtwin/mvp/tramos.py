"""
BoxTwin - Los tramos que la identidad no pudo asignar, y la asignacion a mano.

POR QUE EXISTE
  La siembra no cubre todo el video. Medido sobre `amateur_estatico`, quedaron 327 de 453
  tracks sin asignar, y los golpes que caen en un tramo sin asignar no se cuentan: no hay a
  quien atribuirselos. Es el plan B de C1 que el Cap 6 comprometio como pieza del producto:
  si la identidad automatica no llega al 95%, una persona termina el trabajo en la web.

  Cada tramo viaja con el motivo por el que quedo sin asignar, porque es lo que le dice al
  entrenador donde mirar: "guante tapado o poca luz" se revisa distinto que "cruce o cuerpo
  a cuerpo". El motivo se deduce de la misma evidencia y la misma propuesta que tomaron la
  decision, sin volver a correr nada.

  Las asignaciones manuales son ademas evidencia para la tesis: cuantos tramos tuvo que
  asignar una persona mide cuanto le falta a C1 en uso real.

QUE HACE
  motivo_de       por que un track quedo sin asignar, en uno de cuatro valores
  listar_tramos   los tramos sin asignar de mas de 1 s, ordenados por duracion
  aplicar_manuales  fuerza las asignaciones de la persona encima de la propuesta

USO
  from boxtwin.mvp.tramos import aplicar_manuales, listar_tramos
"""

from __future__ import annotations

from boxtwin.core.types import TrackRole

__all__ = ["MOTIVOS", "ROLES_MANUALES", "aplicar_manuales", "listar_tramos", "motivo_de"]

# Los cuatro motivos que la web sabe decir (prototipo/specs/05-api-y-datos.md, B3).
MOTIVOS = ("color_insuficiente", "sin_coexistencia", "corto", "abstencion")
ROLES_MANUALES = {"A": TrackRole.A, "B": TrackRole.B, "ninguno": TrackRole.IGNORE}

DURACION_MINIMA_S = 1.0


def motivo_de(track: int, evidencia: dict, prop, cfg) -> str:
    """
    Por que este track quedo sin asignar. El orden es el de la decision en `proponer`.

      abstencion          el voto de color empato: no se adivina
      corto               no alcanzaron los recortes para medir nada
      color_insuficiente  los guantes de los dos lados se parecen, o el track tiene muy pocos
                          cuadros con color para votar
      sin_coexistencia    nada lo ata por geometria a ninguno de los dos lados
    """
    votos = (prop.diagnostico or {}).get("votos", {}) or {}
    v = votos.get(track) or votos.get(str(track))
    if v is not None and v[0] == v[1]:
        return "abstencion"
    e = evidencia.get(track)
    if e is None or e.recortes < cfg.min_recortes:
        return "corto"
    separacion = (prop.diagnostico or {}).get("separacion_de_perfiles")
    if separacion is not None and separacion < cfg.separacion_minima:
        return "color_insuficiente"
    if len(e.colores) < cfg.min_guantes_voto:
        return "color_insuficiente"
    return "sin_coexistencia"


def listar_tramos(
    evidencia: dict,
    prop,
    cfg,
    fps: float,
    sin_asignar: list[int] | None = None,
    manuales: dict[int, str] | None = None,
) -> list[dict]:
    """
    Los tramos para revisar: los sin asignar de mas de 1 s, del mas largo al mas corto.

    `sin_asignar` es la lista ANTES de aplicar las manuales, para que un tramo que ya asigno
    la persona siga en la lista con su asignacion a la vista y se pueda cambiar.
    """
    pendientes = sin_asignar if sin_asignar is not None else list(prop.sin_asignar)
    manuales = manuales or {}
    salida = []
    for t in dict.fromkeys(pendientes):          # sin repetidos, en orden
        primero, fin = prop.rangos.get(t, (0, 0))
        dur = (fin - primero) / fps if fps else 0.0
        if dur <= DURACION_MINIMA_S:
            continue
        salida.append({
            "track": int(t),
            "t_desde": round(primero / fps, 2),
            "t_hasta": round(fin / fps, 2),
            "cuadro": int((primero + fin - 1) // 2),
            "motivo": motivo_de(t, evidencia, prop, cfg),
            "asignado": manuales.get(int(t)),
            "recorte": None,
        })
    salida.sort(key=lambda x: (-(x["t_hasta"] - x["t_desde"]), x["track"]))
    return salida


def aplicar_manuales(prop, manuales: dict[int, str]) -> int:
    """
    Fuerza lo que dijo la persona encima de la propuesta. Devuelve cuantos quedaron en A o B.

    Va DESPUES de la asignacion por color y no antes: la persona mira un tramo puntual y su
    respuesta manda sobre ese tramo, pero no se usa como semilla nueva para el resto, que
    seria extender a todo el video una decision tomada mirando tres segundos.
    """
    asignados = 0
    for track, rol in manuales.items():
        if rol not in ROLES_MANUALES:
            raise ValueError(f"rol desconocido para el track {track}: {rol!r}")
        t = int(track)
        prop.roles[t] = ROLES_MANUALES[rol]
        if t in prop.sin_asignar:
            prop.sin_asignar.remove(t)
        if rol in ("A", "B"):
            asignados += 1
    return asignados
