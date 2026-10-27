"""
BoxTwin API - Las cuentas de la version de prueba (versiones 26-10, B29; 02 §11).

POR QUE EXISTE
  El pedido del 26-10: por ser una version de prueba, un boton por version en el ingreso y
  nada de email ni codigo. Los botones entran a cuentas compartidas, asi que hacen falta dos
  cosas que no puede hacer una persona a mano sin equivocarse: cargarlas con material que
  tenga permiso para mostrarse (DV-9) y volverlas a cero sin tocar NINGUNA cuenta real.

  La segunda es la que importa. La estacion tiene las sesiones de prueba de la semana y el
  material de los atletas: un reinicio que borrara de mas no tiene vuelta. Por eso reiniciar
  junta todo lo que va a borrar, verifica cada fila y cada carpeta contra las cuentas marcadas
  `demo`, y si algo no cierra no borra nada.

QUE HACE
  sembrar: crea las tres cuentas de prueba, los cuatro usuarios con una clave aleatoria que
  nadie conoce, los peleadores con su seudonimo, los miembros, el consentimiento de cada
  atleta del gimnasio y la copia de las sesiones reales que dice el yaml. Cada sesion copiada
  es una fila nueva con id nuevo y una carpeta nueva: el video original y la cache de pose van
  con enlace duro (no se escriben nunca despues de procesar), el resto se copia, y en cada
  JSON se reescribe la ruta de la carpeta original por la nueva. Sin eso, re-sembrar una
  sesion de prueba escribiria en la carpeta de la sesion real.

  reiniciar: borra filas y carpetas de las cuentas de prueba y vuelve a sembrar.

USO
  python -m boxtwin_api.demo sembrar   [--yaml ~/boxtwin/demo.yaml]
  python -m boxtwin_api.demo reiniciar [--yaml ~/boxtwin/demo.yaml]
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from boxtwin_api import config
from boxtwin_api.modelos import (
    AsignacionManual,
    Boxeador,
    Consentimiento,
    Correccion,
    Cuenta,
    Miembro,
    Sesion,
    Trabajo,
    Usuario,
)
from boxtwin_api.planes import PERFILES_DEMO, email_demo
from boxtwin_api.seguridad import hashear

__all__ = ["ErrorDemo", "cargar", "reiniciar", "sembrar"]

log = logging.getLogger(__name__)

YAML_POR_DEFECTO = "~/boxtwin/demo.yaml"
NOTA_SEMILLA = "cargado para la versión de prueba"
POR_SEMILLA = "semilla de la versión de prueba"
# Se enlazan solo los archivos que nadie reescribe despues de procesar. procesado.mp4 y
# poses.npz los vuelven a escribir render y completar: un enlace duro haria que esa escritura
# truncara tambien el archivo de la sesion real.
_ENLAZAR = ("videos", "cache")


class ErrorDemo(RuntimeError):
    """Algo del yaml o de la base no cierra. No se toco nada."""


def cargar(ruta: str | Path) -> dict:
    import yaml

    p = Path(ruta).expanduser()
    if not p.is_file():
        raise ErrorDemo(f"no esta {p}: copiar despliegue/demo.yaml.ejemplo y completarlo")
    datos = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    for clave in ("individual", "entrenador", "gimnasio"):
        if not isinstance(datos.get(clave), dict):
            raise ErrorDemo(f"falta la seccion '{clave}' en {p}")
    return datos


# ---------------------------------------------------------------------------
# Sembrar
# ---------------------------------------------------------------------------


def _usuario(db: Session, perfil: str, nombre: str) -> Usuario:
    # La clave se tira apenas se hashea: nadie la conoce y no hay forma de entrar con ella.
    u = Usuario(email=email_demo(perfil), hash_clave=hashear(secrets.token_urlsafe(32)),
                nombre=nombre[:80])
    db.add(u)
    db.flush()
    return u


def _peleador(db: Session, cuenta: Cuenta, autor: Usuario, d: dict, entrenador_id=None) -> Boxeador:
    if not d or not d.get("nombre"):
        raise ErrorDemo("cada peleador del yaml necesita nombre (el seudonimo, DV-9)")
    b = Boxeador(
        usuario_id=autor.id, cuenta_id=cuenta.id, nombre=str(d["nombre"])[:120],
        guardia=d.get("guardia") if d.get("guardia") in ("ortodoxa", "zurda") else None,
        altura_m=d.get("altura_m"), envergadura_m=d.get("envergadura_m"),
        categoria=d.get("categoria"), entrenador_id=entrenador_id,
    )
    db.add(b)
    db.flush()
    return b


def _reescribir(texto: str, origen: Path, destino: Path) -> str:
    return texto.replace(str(origen), str(destino))


def copiar_carpeta(origen: Path, destino: Path) -> dict:
    """Enlace duro para lo pesado que no cambia, copia para el resto, rutas reescritas."""
    hechos = {"enlazados": 0, "copiados": 0}
    for f in sorted(origen.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(origen)
        dst = destino / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix == ".json":
            dst.write_text(_reescribir(f.read_text(encoding="utf-8"), origen, destino),
                           encoding="utf-8")
            hechos["copiados"] += 1
        elif rel.parts[0] in _ENLAZAR:
            try:
                os.link(f, dst)
                hechos["enlazados"] += 1
            except OSError:   # otro filesystem: se copia
                shutil.copy2(f, dst)
                hechos["copiados"] += 1
        else:
            shutil.copy2(f, dst)
            hechos["copiados"] += 1
    return hechos


def _copiar_sesion(db: Session, origen: Sesion, cuenta: Cuenta, subida_por: Usuario,
                   a: Boxeador | None, b: Boxeador | None) -> Sesion:
    nueva = Sesion(
        usuario_id=subida_por.id, cuenta_id=cuenta.id, nombre=origen.nombre,
        estado=origen.estado, error=origen.error, video_nombre=origen.video_nombre,
        duracion_s=origen.duracion_s, fps=origen.fps,
        duracion_declarada_s=origen.duracion_declarada_s,
        round_s=origen.round_s, descanso_s=origen.descanso_s,
        semilla_a=origen.semilla_a, semilla_b=origen.semilla_b,
        boxeador_a_id=a.id if a else None, boxeador_b_id=b.id if b else None,
        contexto=origen.contexto, practica=origen.practica, rounds=origen.rounds,
        consentimiento_en=datetime.now(timezone.utc), consentimiento_por=POR_SEMILLA,
        # La fecha original, para que la evolucion del perfil quede en su orden.
        creada=origen.creada,
    )
    db.add(nueva)
    db.flush()
    return nueva


def _origen(db: Session, sid: str, demo_ids: set[str]) -> Sesion:
    s = db.get(Sesion, str(sid))
    if s is None:
        raise ErrorDemo(f"la sesion {sid} del yaml no existe en la base")
    if s.cuenta_id in demo_ids:
        raise ErrorDemo(f"la sesion {sid} ya es de una cuenta de prueba: el yaml lista reales")
    if not config.cfg.dir_sesion(s.id).is_dir():
        raise ErrorDemo(f"la sesion {sid} no tiene carpeta en {config.cfg.datos}")
    return s


def _cuentas_demo(db: Session) -> list[Cuenta]:
    return list(db.scalars(select(Cuenta).where(Cuenta.demo.is_(True))).all())


def sembrar(db: Session, datos: dict) -> dict:
    """Crea todo en un commit. Si ya hay cuentas de prueba, no hace nada y lo dice."""
    if _cuentas_demo(db) or db.scalar(
            select(Usuario).where(Usuario.email.in_([email_demo(p) for p in PERFILES_DEMO]))):
        return {"ya_estaban": True}
    copias: list[tuple[Path, Path]] = []
    try:
        res = _sembrar(db, datos, copias)
        db.flush()
    except Exception:
        db.rollback()
        raise
    # Las carpetas despues de que las filas estan, y el commit despues de las carpetas: si
    # una copia falla, la base no queda apuntando a carpetas que no existen.
    hechos = {"enlazados": 0, "copiados": 0}
    try:
        for src, dst in copias:
            for k, v in copiar_carpeta(src, dst).items():
                hechos[k] += v
    except Exception:
        db.rollback()
        for _, dst in copias:
            shutil.rmtree(dst, ignore_errors=True)
        raise
    db.commit()
    res.update(hechos)
    return res


def _sembrar(db: Session, datos: dict, copias: list) -> dict:
    ind, ent, gim = datos["individual"], datos["entrenador"], datos["gimnasio"]
    res = {"cuentas": 0, "usuarios": 0, "peleadores": 0, "sesiones": 0, "consentimientos": 0}

    def cuenta(nombre, version) -> Cuenta:
        c = Cuenta(nombre=str(nombre)[:120], version=version, demo=True)
        db.add(c)
        db.flush()
        res["cuentas"] += 1
        return c

    def miembro(c: Cuenta, u: Usuario, rol: str) -> None:
        db.add(Miembro(cuenta_id=c.id, usuario_id=u.id, rol=rol))
        res["usuarios"] += 1

    def copiar(origen: Sesion, c: Cuenta, autor: Usuario, a, b) -> None:
        nueva = _copiar_sesion(db, origen, c, autor, a, b)
        copias.append((config.cfg.dir_sesion(origen.id), config.cfg.dir_sesion(nueva.id)))
        res["sesiones"] += 1

    # Ninguna cuenta de prueba existe todavia (sembrar lo verifico), asi que no hay ids que
    # excluir; _origen igual rechaza una sesion que ya sea de prueba.
    demo_ids: set[str] = set()

    # -- Individual ----------------------------------------------------------------------
    c_ind = cuenta(ind.get("cuenta", "Individual de prueba"), "individual")
    pel = ind.get("peleador") or {"nombre": "Boxeador de prueba"}
    u_ind = _usuario(db, "individual", str(pel.get("nombre", "Boxeador de prueba")))
    miembro(c_ind, u_ind, "titular")
    propio = _peleador(db, c_ind, u_ind, pel)
    c_ind.boxeador_propio_id = propio.id
    res["peleadores"] += 1
    for sid in ind.get("sesiones") or []:
        copiar(_origen(db, sid, demo_ids), c_ind, u_ind, propio, None)

    # -- Entrenador ----------------------------------------------------------------------
    c_ent = cuenta(ent.get("cuenta", "Entrenador de prueba"), "entrenador")
    u_ent = _usuario(db, "entrenador", str(ent.get("titular", "Entrenador de prueba")))
    miembro(c_ent, u_ent, "titular")
    por_nombre = {}
    for d in ent.get("peleadores") or []:
        b = _peleador(db, c_ent, u_ent, d)
        por_nombre[b.nombre] = b
        res["peleadores"] += 1
    for d in ent.get("sesiones") or []:
        copiar(_origen(db, d.get("id"), demo_ids), c_ent, u_ent,
               _por_nombre(por_nombre, d.get("a")), _por_nombre(por_nombre, d.get("b")))

    # -- Gimnasio ------------------------------------------------------------------------
    c_gim = cuenta(gim.get("cuenta", "Gimnasio de prueba"), "gimnasio")
    nombres = gim.get("miembros") or {}
    usuarios = {}
    for perfil in ("gimnasio_titular", "gimnasio_entrenador"):
        u = _usuario(db, perfil, str(nombres.get(perfil, PERFILES_DEMO[perfil].nombre)))
        miembro(c_gim, u, PERFILES_DEMO[perfil].rol)
        usuarios[perfil] = u
    titular = usuarios["gimnasio_titular"]
    version_texto = str(gim.get("consentimiento_version", "semilla"))[:40]
    por_nombre = {}
    for d in gim.get("peleadores") or []:
        a_cargo = d.get("entrenador")
        if a_cargo is not None and a_cargo not in usuarios:
            raise ErrorDemo(f"entrenador '{a_cargo}' no es un perfil del gimnasio")
        b = _peleador(db, c_gim, titular, d, usuarios[a_cargo].id if a_cargo else None)
        por_nombre[b.nombre] = b
        res["peleadores"] += 1
        db.add(Consentimiento(cuenta_id=c_gim.id, boxeador_id=b.id, texto_version=version_texto,
                              mayor_de_edad=True, nota=NOTA_SEMILLA, registrado_por=titular.id,
                              semilla=True))
        res["consentimientos"] += 1
    for d in gim.get("sesiones") or []:
        autor = d.get("subida_por") or "gimnasio_titular"
        if autor not in usuarios:
            raise ErrorDemo(f"subida_por '{autor}' no es un perfil del gimnasio")
        copiar(_origen(db, d.get("id"), demo_ids), c_gim, usuarios[autor],
               _por_nombre(por_nombre, d.get("a")), _por_nombre(por_nombre, d.get("b")))
    return res


def _por_nombre(tabla: dict, nombre) -> Boxeador | None:
    if nombre is None:
        return None
    if nombre not in tabla:
        raise ErrorDemo(f"'{nombre}' no esta entre los peleadores de esa cuenta en el yaml")
    return tabla[nombre]


# ---------------------------------------------------------------------------
# Reiniciar
# ---------------------------------------------------------------------------


def _a_borrar(db: Session) -> dict:
    """Todo lo que pertenece a cuentas de prueba, verificado. ErrorDemo si algo no cierra."""
    cuentas = _cuentas_demo(db)
    cids = {c.id for c in cuentas}
    miembros = list(db.scalars(select(Miembro).where(Miembro.cuenta_id.in_(cids))).all())
    uids = {m.usuario_id for m in miembros}
    # Los usuarios de prueba por email, aunque hayan perdido su miembro: un reinicio los recrea.
    for u in db.scalars(select(Usuario).where(
            Usuario.email.in_([email_demo(p) for p in PERFILES_DEMO]))).all():
        uids.add(u.id)
    sesiones = list(db.scalars(select(Sesion).where(Sesion.cuenta_id.in_(cids))).all())
    boxeadores = list(db.scalars(select(Boxeador).where(Boxeador.cuenta_id.in_(cids))).all())
    sids = {s.id for s in sesiones}
    bids = {b.id for b in boxeadores}

    # Verificaciones: nada de una cuenta real depende de lo que se va a borrar.
    for s in sesiones:
        if s.cuenta_id not in cids:   # pragma: no cover - por construccion de la consulta
            raise ErrorDemo(f"la sesion {s.id} no es de una cuenta de prueba")
    ajenas = db.scalars(select(Sesion).where(
        Sesion.cuenta_id.not_in(cids) | Sesion.cuenta_id.is_(None),
        Sesion.usuario_id.in_(uids)
        | Sesion.boxeador_a_id.in_(bids) | Sesion.boxeador_b_id.in_(bids),
    )).all()
    if ajenas:
        raise ErrorDemo(
            f"{len(ajenas)} sesion(es) de cuentas reales dependen de usuarios o peleadores de "
            "prueba; no se borra nada"
        )
    otros_miembros = db.scalars(select(Miembro).where(
        Miembro.usuario_id.in_(uids), Miembro.cuenta_id.not_in(cids))).all()
    if otros_miembros:
        raise ErrorDemo("un usuario de prueba es miembro de una cuenta real; no se borra nada")
    datos = config.cfg.datos.resolve()
    carpetas = []
    for sid in sids:
        d = config.cfg.dir_sesion(sid).resolve()
        if d.parent != datos / "sesiones":
            raise ErrorDemo(f"la carpeta {d} no esta adentro de {datos / 'sesiones'}")
        carpetas.append(d)
    return {"cuentas": cuentas, "miembros": miembros, "uids": uids, "sesiones": sesiones,
            "sids": sids, "boxeadores": boxeadores, "bids": bids, "carpetas": carpetas}


def reiniciar(db: Session, datos: dict) -> dict:
    """Borra lo de prueba (filas, commit, carpetas) y vuelve a sembrar. Valida el yaml antes."""
    if not isinstance(datos, dict) or not all(
            isinstance(datos.get(k), dict) for k in ("individual", "entrenador", "gimnasio")):
        raise ErrorDemo("el yaml no tiene las tres secciones; no se borra nada")
    plan = _a_borrar(db)
    sids, bids, cids = plan["sids"], plan["bids"], {c.id for c in plan["cuentas"]}
    for modelo in (Trabajo, Correccion, AsignacionManual):
        for fila in db.scalars(select(modelo).where(modelo.sesion_id.in_(sids))).all():
            db.delete(fila)
    for c in db.scalars(select(Consentimiento).where(
            Consentimiento.cuenta_id.in_(cids) | Consentimiento.boxeador_id.in_(bids))).all():
        db.delete(c)
    db.flush()
    for s in plan["sesiones"]:
        db.delete(s)
    db.flush()
    for b in plan["boxeadores"]:
        db.delete(b)
    for m in plan["miembros"]:
        db.delete(m)
    db.flush()
    for c in plan["cuentas"]:
        db.delete(c)
    for u in db.scalars(select(Usuario).where(Usuario.id.in_(plan["uids"]))).all():
        db.delete(u)
    db.commit()
    for d in plan["carpetas"]:
        shutil.rmtree(d, ignore_errors=True)
    res = sembrar(db, datos)
    res["borradas"] = {"sesiones": len(sids), "carpetas": len(plan["carpetas"])}
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("USO")[0])
    p.add_argument("accion", choices=("sembrar", "reiniciar"))
    p.add_argument("--yaml", default=YAML_POR_DEFECTO, help="default: %(default)s")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from boxtwin_api.db import crear_tablas, hacer_sesion

    crear_tablas()
    try:
        datos = cargar(args.yaml)
        with hacer_sesion() as db:
            res = sembrar(db, datos) if args.accion == "sembrar" else reiniciar(db, datos)
    except ErrorDemo as e:
        log.error("%s", e)
        return 2
    if res.get("ya_estaban"):
        log.info("las cuentas de prueba ya estaban: no se hizo nada (para volver a cero, reiniciar)")
    else:
        log.info("hecho: %s", res)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
