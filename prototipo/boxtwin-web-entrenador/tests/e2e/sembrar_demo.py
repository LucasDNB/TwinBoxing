"""
POR QUE EXISTE
  Los recorridos E1 a E3 de las versiones (specs/versiones-26-10, 05 §4) entran por los
  botones de la version de prueba, y esas cuentas se cargan copiando sesiones reales. La
  base de los e2e arranca vacia: esto le pone una cuenta real con tres sesiones listas
  (bolsa, sombra y sparring, con los archivos del anotador falso) y despues siembra las
  cuentas de prueba con demo.py, como en la estacion. E3 verifica que la real no aparece.

QUE HACE
  Crea las tablas, la cuenta real y sus sesiones, escribe un demo.yaml y corre
  `demo.sembrar`. Deja en <salida>/real.json los ids reales para que E3 los busque.

USO
  python -I sembrar_demo.py <directorio para real.json>   (con BOXTWIN_DB y BOXTWIN_DATOS)
"""
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

AQUI = Path(__file__).resolve().parent
FIXTURES = AQUI.parents[0] / "fixtures" / "sesion-real"


def main(salida: str) -> None:
    from boxtwin_api.config import cfg
    from boxtwin_api.db import crear_tablas, hacer_sesion
    from boxtwin_api.demo import sembrar
    from boxtwin_api.modelos import Cuenta, Miembro, Sesion, Usuario
    from boxtwin_api.seguridad import hashear

    crear_tablas()
    video = AQUI / ".e2e" / "sparring.mp4"
    with hacer_sesion() as db:
        u = Usuario(email="real@usal.edu.ar", hash_clave=hashear("sparring2026"), nombre="Real")
        db.add(u)
        db.flush()
        c = Cuenta(nombre="real@usal.edu.ar", version="entrenador")
        db.add(c)
        db.flush()
        db.add(Miembro(cuenta_id=c.id, usuario_id=u.id, rol="titular"))
        ids = {}
        for i, ctx in enumerate(("bolsa", "sombra", "sparring")):
            s = Sesion(usuario_id=u.id, cuenta_id=c.id, nombre=f"Real {ctx}", estado="listo",
                       video_nombre="sparring.mp4", duracion_s=180.0, fps=30.0, contexto=ctx,
                       semilla_a=41, semilla_b=None if ctx != "sparring" else 64,
                       consentimiento_en=datetime.now(timezone.utc), consentimiento_por=u.email,
                       creada=datetime.now(timezone.utc) - timedelta(days=3 - i))
            db.add(s)
            db.flush()
            d = cfg.dir_sesion(s.id)
            (d / "videos").mkdir(parents=True)
            shutil.copy(video, d / "videos" / "sparring.mp4")
            fc = json.loads((FIXTURES / "fightcard-0.2.json").read_text())
            fc["identidad"]["sin_asignar"] = 0.0
            if ctx != "sparring":
                fc["peleadores"].pop("B", None)
            (d / "fightcard.json").write_text(json.dumps(fc))
            (d / "sesion.json").write_text(json.dumps({
                "estado": "listo", "candidatos": [], "identidad": {"tramos": []},
                "video": {"ruta": str(d / "videos" / "sparring.mp4"), "duracion_s": 180.0, "fps": 30.0},
            }))
            ids[ctx] = s.id
        db.commit()

    yaml = {
        "individual": {"cuenta": "Individual de prueba",
                       "peleador": {"nombre": "Boxeador de prueba", "guardia": "ortodoxa"},
                       "sesiones": [ids["bolsa"], ids["sombra"]]},
        "entrenador": {"cuenta": "Entrenador de prueba",
                       "peleadores": [{"nombre": "Atleta 1"}, {"nombre": "Atleta 2"}],
                       "sesiones": [{"id": ids["sparring"], "a": "Atleta 1", "b": "Atleta 2"}]},
        "gimnasio": {"cuenta": "Gimnasio de prueba",
                     "miembros": {"gimnasio_titular": "Titular de prueba",
                                  "gimnasio_entrenador": "Entrenador de prueba"},
                     "peleadores": [{"nombre": "Atleta 3", "entrenador": "gimnasio_entrenador"}],
                     "sesiones": []},
    }
    with hacer_sesion() as db:
        print("sembrar:", sembrar(db, yaml), file=sys.stderr)
    Path(salida).mkdir(parents=True, exist_ok=True)
    (Path(salida) / "real.json").write_text(json.dumps(ids))


if __name__ == "__main__":
    main(sys.argv[1])
