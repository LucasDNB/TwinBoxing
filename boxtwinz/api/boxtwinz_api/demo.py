"""
BoxTwinZ - Datos de demo (Z18): una cuenta free y una premium con sesiones de las tres
modalidades, para recorrer la app en menos de 5 minutos sin esperar a la GPU.

POR QUE EXISTE
  La presentacion no puede depender de procesar video en vivo. Las sesiones de demo tienen
  golpes SINTETICOS (generados con semilla fija) y se marcan asi en el nombre del video, para
  que nadie confunda un numero de demo con una medicion. Los videos propios de Lucas (spec 15)
  se pasan con --videos y reemplazan al video de prueba; los golpes siguen siendo sinteticos
  hasta que esos videos se procesen de verdad por la app.

QUE HACE
  sembrar: crea demo-free@boxtwinz.ar y demo-premium@boxtwinz.ar (amigos entre si, perfiles
  publicos), 3 sombras, 3 bolsas y 2 sparrings por cuenta, valoradas; en premium un cajon
  "Sparring de demo" que recibe los sparrings. Idempotente: si las cuentas existen, no hace nada.
  reiniciar: borra las dos cuentas y vuelve a sembrar.

USO
  python -m boxtwinz_api.demo sembrar --clave una-clave-de-demo [--videos ~/videos-demo]
  python -m boxtwinz_api.demo reiniciar --clave una-clave-de-demo
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

from sqlalchemy import select

from boxtwinz_api.config import cfg
from boxtwinz_api.db import crear_tablas, hacer_sesion
from boxtwinz_api.models import Amistad, Golpe, Perfil, Sesion, Usuario, Valoracion, ahora
from boxtwinz_api.seguridad import hashear
from boxtwinz_api.services.borrado import borrar_sesiones
from boxtwinz_api.services.ingest import es_apta_ranking, recalcular_metricas
from boxtwinz_api.services.profiles import crear_ranuras

CUENTAS = {"demo-free@boxtwinz.ar": ("Demo Free", "free", "ortodoxa"),
           "demo-premium@boxtwinz.ar": ("Demo Premium", "premium", "zurda")}
DURACION = {"sombra": 200.0, "bolsa": 240.0, "sparring": 240.0}


def golpes_sinteticos(rng: random.Random, duracion: float, por_min: float, sosten: float, pct_izq: float,
                      quien: str = "usuario") -> list[tuple[str, float, str]]:
    """Rafagas de 1 a 3 golpes con el volumen cayendo despues del primer minuto de cada round."""
    out, t = [], rng.uniform(0.5, 2.0)
    while t < duracion:
        en_round = t % 180
        ritmo = por_min * (1.0 if en_round < 60 else sosten)
        for k in range(rng.choice((1, 1, 2, 2, 3))):
            if t + k * 0.35 < duracion:
                out.append((quien, round(t + k * 0.35, 3), "izquierda" if rng.random() < pct_izq else "derecha"))
        t += rng.expovariate(ritmo / 60 / 1.9) + 0.6
    return out


def _video(destino: Path, modalidad: str, videos: Path | None) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    propio = videos / f"{modalidad}.mp4" if videos else None
    if not (propio and propio.is_file()):
        # Un video de prueba por modalidad, generado una vez y copiado a cada sesion.
        propio = destino.parent / f"demo-plantilla-{modalidad}.mp4"
        if not propio.exists():
            subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                            f"testsrc=duration={int(DURACION[modalidad])}:size=320x180:rate=15",
                            "-pix_fmt", "yuv420p", str(propio)], check=True)
    shutil.copyfile(propio, destino)
    return destino


def sembrar(clave: str, videos: Path | None = None) -> bool:
    crear_tablas()
    with hacer_sesion() as db:
        if db.scalar(select(Usuario).where(Usuario.email.in_(list(CUENTAS)))):
            return False
        rng = random.Random(20261117)
        usuarios = {}
        for email, (nombre, plan, guardia) in CUENTAS.items():
            u = Usuario(email=email, hash=hashear(clave), nombre=nombre, plan=plan, rol="usuario",
                        consentimiento_uso_at=ahora(), consentimiento_uso_version=cfg.version_consentimiento_uso,
                        consentimiento_reent_at=ahora(), consentimiento_reent_version=cfg.version_consentimiento_reent)
            db.add(u)
            db.flush()
            ps = crear_ranuras(db, u, guardia)
            ps[0].publico = True
            if plan == "premium":
                ps[1].customizado, ps[1].nombre, ps[1].estilo, ps[1].arquetipo_id = True, "Sparring de demo", "fajador", None
            usuarios[email] = (u, ps)
        a, b = (usuarios[e][0] for e in CUENTAS)
        db.add(Amistad(solicitante_id=a.id, destinatario_id=b.id, estado="aceptada"))

        base = ahora() - timedelta(days=12)
        for email, (u, ps) in usuarios.items():
            fuerza = 1.0 if u.plan == "free" else 1.15
            dia = 0
            for modalidad, n in (("sombra", 3), ("bolsa", 3), ("sparring", 2)):
                for i in range(n):
                    dia += 1
                    dur = DURACION[modalidad]
                    s = Sesion(usuario_id=u.id, perfil_id=ps[0].id, modalidad=modalidad, estado="lista",
                               video_ruta="", video_duracion_s=dur, video_fps=15.0, video_ancho=320, video_alto=180,
                               calidad_pose=round(rng.uniform(0.82, 0.97), 3), segundos_gpu=dur * rng.uniform(1.2, 2.6),
                               rival_perfil_id=ps[1].id if (modalidad == "sparring" and u.plan == "premium") else None,
                               vista_resultado=True, creado=base + timedelta(days=dia))
                    db.add(s)
                    db.flush()
                    s.video_ruta = str(_video(cfg.dir_videos() / f"demo-sintetico-{s.id}.mp4", modalidad, videos))
                    vol = {"sombra": 30, "bolsa": 36, "sparring": 22}[modalidad] * fuerza * (1 + 0.04 * i)
                    gs = golpes_sinteticos(rng, dur, vol, rng.uniform(0.72, 0.95), 0.62 if u.plan == "free" else 0.4)
                    if modalidad == "sparring":
                        gs += golpes_sinteticos(rng, dur, 18, 0.85, 0.3, quien="rival")
                    for quien, t, mano in gs:
                        db.add(Golpe(sesion_id=s.id, quien=quien, mano=mano, t_inicio=t, t_fin=t + 0.25,
                                     confianza=round(rng.uniform(0.8, 0.99), 3),
                                     altura_estimada="cabeza" if rng.random() < 0.6 else "cuerpo"))
                    db.flush()
                    recalcular_metricas(db, s)
                    s.apta_ranking = es_apta_ranking(s, ps[0], u)
                    db.add(Valoracion(sesion_id=s.id, estrellas=rng.choice((3, 4, 4, 5)), modalidad=modalidad))
        db.commit()
    return True


def borrar(db) -> None:
    for u in db.scalars(select(Usuario).where(Usuario.email.in_(list(CUENTAS)))):
        ses = list(db.scalars(select(Sesion).where(Sesion.usuario_id == u.id)))
        for s in ses:
            Path(s.video_ruta).unlink(missing_ok=True)
        borrar_sesiones(db, [s.id for s in ses])
        for a in db.scalars(select(Amistad).where((Amistad.solicitante_id == u.id) | (Amistad.destinatario_id == u.id))):
            db.delete(a)
        for p in db.scalars(select(Perfil).where(Perfil.usuario_id == u.id)):
            db.delete(p)
        db.flush()
        db.delete(u)
    db.commit()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("USO")[0])
    p.add_argument("accion", choices=("sembrar", "reiniciar"))
    p.add_argument("--clave", required=True, help="clave de las dos cuentas de demo (8 o mas caracteres)")
    p.add_argument("--videos", type=Path, default=None, help="carpeta con sombra.mp4, bolsa.mp4 y sparring.mp4")
    a = p.parse_args(argv)
    if len(a.clave) < 8:
        p.error("la clave tiene que tener 8 caracteres o mas")
    if a.accion == "reiniciar":
        crear_tablas()
        with hacer_sesion() as db:
            borrar(db)
    hecho = sembrar(a.clave, a.videos)
    print("cuentas de demo sembradas" if hecho else "las cuentas de demo ya existian: nada que hacer")
    for email, (_, plan, _) in CUENTAS.items():
        print(f"  {email} ({plan})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
