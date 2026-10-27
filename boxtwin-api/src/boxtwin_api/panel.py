"""
BoxTwin API - El panel del gimnasio (prototipo/specs/05-api-y-datos.md, B5).

POR QUE EXISTE
  La pantalla 01 muestra cuatro indicadores y lo pendiente. Sin esto la web tendria que
  pedir cada Fight-Card del mes para sumar rounds y golpes, que en un celular por datos son
  decenas de pedidos para dibujar cuatro numeros.

  Ningun numero del panel sale de una constante (CA-01.2): rounds y golpes salen de las
  Fight-Cards, y el recall del detector sale del documento mas reciente, no de la web.

QUE HACE
  Cuenta peleadores activos y preliminares, sesiones del mes por estado, rounds del mes por
  contexto y golpes detectados del mes. Lee las Fight-Cards de disco con una cache por
  sesion que se invalida con la fecha de modificacion del archivo: el entrenador corrige y
  re-siembra, y un panel que no lo refleja miente.

USO
  from boxtwin_api.panel import construir_panel
  construir_panel(acceso, db, cfg.dir_sesion, cfg.perfil_preliminar_sesiones)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select

from boxtwin_api.modelos import Boxeador, Sesion

__all__ = ["EN_CURSO", "construir_panel", "resumen_fightcard"]

EN_CURSO = ("subida", "en_cola", "procesando", "completando")
DIAS_ACTIVO = 60

# ruta -> (mtime, resumen). Vive en el proceso: con un nodo y un worker que escribe atomico,
# la fecha de modificacion alcanza para saber si cambio.
_cache: dict[str, tuple[float, dict | None]] = {}


def resumen_fightcard(ruta: Path) -> dict | None:
    """Rounds, golpes detectados y recall de una Fight-Card. None si no hay o no se lee."""
    try:
        mtime = ruta.stat().st_mtime
    except OSError:
        return None
    clave = str(ruta)
    ya = _cache.get(clave)
    if ya and ya[0] == mtime:
        return ya[1]
    try:
        fc = json.loads(ruta.read_text())
        pel = fc.get("peleadores") or {}
        # Una sesion sin rounds declarados es un tramo continuo: cuenta como uno. Sin esto
        # un gimnasio que no declara rounds veria "0 rounds analizados" con sesiones listas.
        rondas = len((fc.get("video") or {}).get("rounds") or []) or 1
        r = {
            "rounds": rondas,
            "golpes": sum(int((p.get("total") or {}).get("total") or 0) for p in pel.values()),
            "recall": (fc.get("detector") or {}).get("recall_medido"),
        }
    except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
        r = None
    _cache[clave] = (mtime, r)
    return r


def _utc(d: datetime | None) -> datetime | None:
    if d is None:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def construir_panel(acceso, db, dir_sesion, preliminar: int, ahora: datetime | None = None) -> dict:
    """Lo de la cuenta (versiones 26-10, B19). En Gimnasio, de todos sus miembros."""
    ahora = ahora or datetime.now(timezone.utc)
    inicio_mes = ahora.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    desde_activo = ahora - timedelta(days=DIAS_ACTIVO)

    sesiones = db.scalars(
        select(Sesion).where(Sesion.cuenta_id == acceso.cuenta.id).order_by(Sesion.creada.desc())
    ).all()

    # -- peleadores ----------------------------------------------------------------
    cantidad: dict[str, int] = {}
    activos: set[str] = set()
    for s in sesiones:
        for bid in {s.boxeador_a_id, s.boxeador_b_id} - {None}:
            cantidad[bid] = cantidad.get(bid, 0) + 1
            if _utc(s.creada) >= desde_activo:
                activos.add(bid)
    propios = set(db.scalars(select(Boxeador.id).where(Boxeador.cuenta_id == acceso.cuenta.id)).all())
    activos &= propios

    # -- sesiones, rounds y golpes del mes ---------------------------------------------
    del_mes = [s for s in sesiones if _utc(s.creada) >= inicio_mes]
    por_estado = {"total": len(del_mes), "listo": 0, "en_curso": 0, "espera_siembra": 0, "fallo": 0}
    rounds = {"total": 0}
    golpes = 0
    for s in del_mes:
        if s.estado in EN_CURSO:
            por_estado["en_curso"] += 1
        elif s.estado in por_estado:
            por_estado[s.estado] += 1
        if s.estado != "listo":
            continue
        r = resumen_fightcard(Path(dir_sesion(s.id)) / "fightcard.json")
        if r is None:
            continue
        ctx = s.contexto or "sparring"
        rounds["total"] += r["rounds"]
        rounds[ctx] = rounds.get(ctx, 0) + r["rounds"]
        golpes += r["golpes"]

    # El recall es el del detector que midio la sesion lista mas reciente: si el detector
    # cambia, la frase del panel cambia con el.
    recall = None
    for s in sesiones:
        if s.estado == "listo":
            r = resumen_fightcard(Path(dir_sesion(s.id)) / "fightcard.json")
            if r and r["recall"] is not None:
                recall = r["recall"]
                break

    # B24: en Gimnasio cada pendiente dice quien lo subio; cualquier miembro lo resuelve.
    # Solo ahi: en las otras versiones hay un miembro y el pendiente queda como antes.
    compartido = acceso.plan.plantel_compartido
    nombres = _nombres(db, {s.usuario_id for s in sesiones}) if compartido else {}
    pendientes = [
        {"sesion_id": s.id, "estado": s.estado, "titulo": s.nombre,
         **({"subida_por": {"id": s.usuario_id, "nombre": nombres.get(s.usuario_id)}}
            if compartido else {})}
        for s in sesiones if s.estado == "espera_siembra"
    ]

    return {
        "peleadores_activos": len(activos),
        "peleadores_preliminares": sum(1 for b in activos if cantidad.get(b, 0) < preliminar),
        "sesiones_mes": por_estado,
        "rounds_mes": rounds,
        "golpes_detectados_mes": golpes,
        "recall_detector": recall,
        "pendientes": pendientes,
        # B24: solo el titular de un Gimnasio; nulo para todos los demas (CA-G01.1).
        "por_entrenador": (
            _por_entrenador(acceso, db, sesiones)
            if acceso.plan.plantel_compartido and acceso.rol == "titular" else None
        ),
    }


def _nombres(db, ids: set) -> dict:
    from boxtwin_api.modelos import Usuario

    if not ids:
        return {}
    return {u.id: (u.nombre or u.email)
            for u in db.scalars(select(Usuario).where(Usuario.id.in_(ids))).all()}


def _por_entrenador(acceso, db, sesiones) -> list[dict]:
    """
    Sesiones, horas y pendientes del mes por miembro activo, en el orden en que entraron.
    El mes es el del cupo, cortado en Buenos Aires, para que las horas sumen lo mismo que
    GET /cuenta.
    """
    from collections import defaultdict

    from boxtwin_api.cupo import inicio_de_mes
    from boxtwin_api.modelos import Miembro

    desde, hasta, _ = inicio_de_mes(datetime.now(timezone.utc))
    por_usuario: dict[str, list] = defaultdict(list)
    for s in sesiones:
        por_usuario[s.usuario_id].append(s)
    miembros = db.scalars(
        select(Miembro).where(Miembro.cuenta_id == acceso.cuenta.id, Miembro.baja_en.is_(None))
        .order_by(Miembro.desde)
    ).all()
    nombres = _nombres(db, {m.usuario_id for m in miembros})
    filas = []
    for m in miembros:
        suyas = por_usuario.get(m.usuario_id, [])
        del_mes = [s for s in suyas if desde <= _utc(s.creada) < hasta]
        segundos = sum(
            (s.duracion_s if s.duracion_s is not None else (s.duracion_declarada_s or 0.0))
            for s in del_mes if s.estado != "fallo"
        )
        filas.append({
            "usuario_id": m.usuario_id,
            "nombre": nombres.get(m.usuario_id),
            "rol": m.rol,
            "sesiones_mes": len(del_mes),
            "horas_mes": round(segundos / 3600, 2),
            "pendientes": sum(1 for s in suyas if s.estado == "espera_siembra"),
        })
    return filas
