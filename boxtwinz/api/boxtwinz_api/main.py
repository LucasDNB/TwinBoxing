"""
BoxTwinZ API - La aplicacion FastAPI.

POR QUE EXISTE
  Un solo origen, como A: la API sirve la web exportada (BOXTWINZ_WEB) y todo lo de la API va
  bajo /api, asi no hay CORS y alcanza con un tunel.

QUE HACE
  Monta los routers bajo /api, /api/salud con commit, arranque y estado de la cola, y la web
  estatica con fallback a la pagina 404 de la exportacion.

USO
  uvicorn boxtwinz_api.main:app --host 127.0.0.1 --port 8003
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from boxtwinz_api import __version__
from boxtwinz_api.config import cfg
from boxtwinz_api.db import crear_tablas, obtener_sesion
from boxtwinz_api.models import Trabajo, ahora
from boxtwinz_api.routers import admin, cuentas, perfiles, planes, rival, sesiones, social

ARRANQUE = ahora()


def _commit(desde: Path | None = None) -> str | None:
    """El commit del checkout, leido de .git sin subprocesos (mismo criterio que A, T1)."""
    d = (desde or Path(__file__)).resolve()
    for padre in [d, *d.parents]:
        git = padre / ".git"
        if git.is_dir():
            gitdir, comun = git, git
            break
        if git.is_file():
            gitdir = Path(git.read_text().split("gitdir:", 1)[1].strip())
            gitdir = gitdir if gitdir.is_absolute() else (padre / gitdir).resolve()
            c = gitdir / "commondir"
            comun = (gitdir / c.read_text().strip()).resolve() if c.is_file() else gitdir
            break
    else:
        return None
    try:
        head = (gitdir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head[:12]
        ref = head.split(" ", 1)[1]
        for base in (gitdir, comun):
            if (base / ref).is_file():
                return (base / ref).read_text().strip()[:12]
        for linea in (comun / "packed-refs").read_text().splitlines():
            if linea.endswith(" " + ref):
                return linea.split(" ", 1)[0][:12]
    except OSError:
        return None
    return None


COMMIT = _commit()


@asynccontextmanager
async def _vida(_app: FastAPI):
    crear_tablas()
    yield


app = FastAPI(title="BoxTwinZ", version=__version__, lifespan=_vida, docs_url=None, redoc_url=None,
              openapi_url=None)
for r in (cuentas.router, perfiles.router, sesiones.router, planes.router, social.router, rival.router, admin.router):
    app.include_router(r, prefix="/api")


@app.get("/api/salud")
def salud(db: Session = Depends(obtener_sesion)) -> dict:
    cola = dict(db.execute(select(Trabajo.estado, func.count()).group_by(Trabajo.estado)).all())
    ultimo = db.scalar(select(func.max(Trabajo.tomado_en)))
    tomados = cola.get("tomado", 0)
    if ultimo is not None and ultimo.tzinfo is None:
        from datetime import timezone

        ultimo = ultimo.replace(tzinfo=timezone.utc)
    return {
        "ok": True, "servicio": "boxtwinz", "version": __version__, "commit": COMMIT,
        "arranque": ARRANQUE.isoformat(), "secreto_efimero": cfg.secreto_efimero,
        "cola": {"en_cola": cola.get("en_cola", 0), "tomados": tomados,
                 "fallidos": cola.get("fallo", 0), "listos": cola.get("listo", 0)},
        # Sin latido propio: el worker se ve vivo si tomo algo hace poco o tiene algo tomado.
        "worker": {"ultimo_trabajo": ultimo.isoformat() if ultimo else None,
                   "activo_reciente": bool(tomados) or bool(ultimo and ahora() - ultimo < timedelta(hours=1))},
    }


def _montar_web(web: Path) -> None:
    @app.get("/{ruta:path}", include_in_schema=False)
    def estatico(ruta: str):
        if ruta.startswith("api/"):
            return FileResponse(web / "404.html", status_code=404)
        base = web.resolve()
        for cand in (base / ruta, base / ruta / "index.html", base / f"{ruta}.html"):
            try:
                c = cand.resolve()
            except OSError:
                continue
            # Que la ruta resuelta quede adentro de la web: "../" no sale del directorio.
            if c.is_file() and base in c.parents:
                return FileResponse(c)
        return FileResponse(base / "404.html", status_code=404)


if cfg.web and cfg.web.is_dir():
    _montar_web(cfg.web)
