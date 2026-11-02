"""
BoxTwin - Carga en A el combate de transmision del gemelo de un tercero (gemelo-tercero, 2-11).

POR QUE EXISTE
  El 2-11 Lucas decidio armar el gemelo digital (perfil tactico) de un boxeador que no es el,
  con Pacquiao-Margarito, en A (produccion) y con seudonimos. La unica entrada de video de A
  es POST /videos, que exige que quien sube declare el consentimiento de las personas
  filmadas (R-09, Ley 25.326). Para una transmision publica ese consentimiento no existe:
  marcar la casilla dejaria en la base una declaracion falsa con el email de Lucas.
  Cambiar la ruta para aceptar otro origen es cambiar A, que se congela el 3-11.

  Este script es la otra puerta, de operador y sin red: crea la sesion directo en la base
  de A, igual que la crea la ruta, pero con el origen del material escrito donde iria el
  consentimiento (`consentimiento_por`). Asi queda auditable de donde salio el video y que
  nadie consintio nada. No toca el codigo de A: corre con el paquete instalado en su
  entorno.

QUE HACE
  1. Busca al usuario por email y su cuenta (la misma regla que acceso_actual).
  2. Crea, si no existen en esa cuenta, los dos peleadores con seudonimo, guardia y una
     nota que dice que son material de terceros. La guardia importa: el tipo v2 saca jab y
     cross del brazo y la guardia, y sin guardia supone ortodoxa.
  3. Crea la sesion (contexto, rounds, duracion del round y descanso), con los dos
     peleadores ya asignados (A y B por la siembra que se haga despues), copia o enlaza el
     video a su directorio y encola `procesar` en el mismo commit, como la ruta.
  4. Reanudable: si la cuenta ya tiene una sesion con ese nombre, no crea otra y dice cual es.
  Con --ensayo imprime lo que haria y no escribe nada.

USO
  # en la estacion, con el entorno de A (twinboxing_env) y su /etc/boxtwin.env cargado
  set -a; source /etc/boxtwin.env; set +a
  python boxtwin-api/scripts/cargar_tercero.py --email <email de Lucas> \
      --video ~/Proyectos/TwinBoxing/anotacion-pacquiao/videos/<video>.mp4 --ensayo
  (sin --ensayo, carga; los defaults son los de la spec docs/specs/2026-11-02-gemelo-tercero)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ORIGEN = ("material de terceros: transmision publica, uso academico (RE-06); "
          "sin consentimiento de las personas filmadas")
NOTA = ("Gemelo de un tercero (2-11-2026): perfil armado con material de transmision, "
        "con seudonimo. No es un atleta de la cuenta. docs/specs/2026-11-02-gemelo-tercero")
GUARDIAS = ("ortodoxa", "zurda")


# ---------------------------------------------------------------------------
# Partes puras (con pruebas)
# ---------------------------------------------------------------------------

def validar(args) -> list[str]:
    """Los errores de los argumentos, antes de abrir la base. Vacio si todo esta bien."""
    from boxtwin_api.esquemas import CONTEXTOS

    errores = []
    if args.contexto not in CONTEXTOS:
        errores.append(f"contexto tiene que ser uno de {', '.join(CONTEXTOS)}")
    for lado in ("a", "b"):
        if getattr(args, f"guardia_{lado}") not in GUARDIAS:
            errores.append(f"guardia_{lado} tiene que ser ortodoxa o zurda")
        if not getattr(args, f"seudonimo_{lado}").strip():
            errores.append(f"seudonimo_{lado} vacio")
    if args.seudonimo_a.strip() == args.seudonimo_b.strip():
        errores.append("los dos seudonimos son iguales")
    if args.rounds is not None and args.rounds < 1:
        errores.append("rounds tiene que ser 1 o mas")
    if args.round_s is not None and args.round_s <= 0:
        errores.append("round_s tiene que ser mayor que cero")
    if args.descanso_s < 0:
        errores.append("descanso_s no puede ser negativo")
    if len(args.origen) > 320:
        errores.append("origen supera los 320 caracteres de consentimiento_por")
    return errores


def _nombre_de_video(ruta: Path) -> str:
    """El mismo saneado que usa la ruta de subida (app._nombre_seguro)."""
    import re

    limpio = re.sub(r"[^A-Za-z0-9._-]", "_", ruta.name).lstrip(".")
    return limpio[:120] or "video.mp4"


def _poner_video(origen: Path, destino: Path, enlazar: bool) -> str:
    """Enlace duro si se puede (el video ya esta en el disco de la estacion), si no copia."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    if enlazar:
        try:
            os.link(origen, destino)
            return "enlace"
        except OSError:
            pass
    shutil.copy2(origen, destino)
    return "copia"


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def cargar(db, cfg, args, ahora: datetime | None = None) -> dict:
    """
    Hace la carga sobre una sesion de base abierta. Commitea al final, una vez. Devuelve lo
    que hizo (o lo que haria, con args.ensayo).
    """
    from sqlalchemy import select

    from boxtwin_api.acceso import acceso_de_usuario
    from boxtwin_api.cola import encolar
    from boxtwin_api.modelos import Boxeador, Sesion, Usuario

    ahora = ahora or datetime.now(timezone.utc)
    u = db.scalar(select(Usuario).where(Usuario.email == args.email))
    if u is None:
        raise SystemExit(f"no hay usuario con email {args.email}")
    cuenta = acceso_de_usuario(u, db).cuenta
    video = Path(args.video).expanduser()
    if not video.is_file():
        raise SystemExit(f"no existe el video {video}")

    out = {"cuenta": cuenta.id, "ensayo": bool(args.ensayo), "origen": args.origen,
           "peleadores": {}, "sesion": None}

    ya = db.scalar(select(Sesion).where(Sesion.cuenta_id == cuenta.id,
                                        Sesion.nombre == args.nombre))
    if ya is not None:
        out["sesion"] = {"id": ya.id, "estado": ya.estado, "nueva": False}
        return out

    ids = {}
    for lado in ("a", "b"):
        nombre = getattr(args, f"seudonimo_{lado}").strip()
        guardia = getattr(args, f"guardia_{lado}")
        b = db.scalar(select(Boxeador).where(Boxeador.cuenta_id == cuenta.id,
                                             Boxeador.nombre == nombre))
        nuevo = b is None
        if nuevo and not args.ensayo:
            b = Boxeador(usuario_id=u.id, cuenta_id=cuenta.id, nombre=nombre,
                         guardia=guardia, notas=NOTA, estilo=args.estilo or None)
            db.add(b)
            db.flush()
        out["peleadores"][lado.upper()] = {
            "nombre": nombre, "guardia": guardia if nuevo else b.guardia, "nuevo": nuevo,
            "id": None if b is None else b.id}
        ids[lado] = None if b is None else b.id

    if args.ensayo:
        out["sesion"] = {"nombre": args.nombre, "contexto": args.contexto,
                         "rounds": args.rounds, "round_s": args.round_s,
                         "descanso_s": args.descanso_s, "video": str(video), "nueva": True}
        return out

    ses = Sesion(
        usuario_id=u.id, cuenta_id=cuenta.id, nombre=args.nombre[:255],
        video_nombre=_nombre_de_video(video), round_s=args.round_s,
        descanso_s=args.descanso_s, estado="en_cola", contexto=args.contexto,
        rounds=args.rounds, boxeador_a_id=ids["a"], boxeador_b_id=ids["b"],
        consentimiento_en=ahora, consentimiento_por=args.origen,
    )
    db.add(ses)
    db.flush()
    modo = _poner_video(video, cfg.dir_sesion(ses.id) / "videos" / ses.video_nombre,
                        enlazar=not args.copiar)
    encolar(db, ses.id, "procesar", round_s=args.round_s, descanso_s=args.descanso_s)
    db.commit()
    out["sesion"] = {"id": ses.id, "estado": ses.estado, "nueva": True, "video": modo}
    return out


def armar_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--email", required=True, help="usuario de A dueno de la sesion")
    p.add_argument("--video", required=True)
    p.add_argument("--nombre", default="Combate de transmision · gemelo de tercero")
    p.add_argument("--contexto", default="combate")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--round-s", type=float, default=180.0)
    p.add_argument("--descanso-s", type=float, default=0.0,
                   help="0 si el video tiene los rounds netos, sin descansos")
    p.add_argument("--seudonimo-a", default="Tercero Z")
    p.add_argument("--guardia-a", default="zurda")
    p.add_argument("--seudonimo-b", default="Tercero O")
    p.add_argument("--guardia-b", default="ortodoxa")
    p.add_argument("--estilo", default="")
    p.add_argument("--origen", default=ORIGEN)
    p.add_argument("--copiar", action="store_true", help="copia en vez de enlace duro")
    p.add_argument("--ensayo", action="store_true", help="no escribe nada")
    return p


def main(argv: list[str] | None = None) -> int:
    args = armar_parser().parse_args(argv)
    errores = validar(args)
    if errores:
        for e in errores:
            print("error:", e, file=sys.stderr)
        return 2
    from boxtwin_api.config import cfg
    from boxtwin_api.db import crear_tablas, hacer_sesion

    crear_tablas()
    with hacer_sesion() as db:
        out = cargar(db, cfg, args)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
