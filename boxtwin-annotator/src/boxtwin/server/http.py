"""
BoxTwin - Transporte HTTP.

POR QUE EXISTE
  Se usa el servidor de la biblioteca estandar y no un framework. La API son seis rutas y
  el unico cliente es el navegador de la maquina de al lado: una dependencia mas seria algo
  que puede romperse en seis meses a cambio de ahorrar cincuenta lineas de despacho.
  ThreadingHTTPServer y no el de un solo hilo porque el navegador abre varias conexiones en
  paralelo para traer imagenes; con un solo hilo, una peticion lenta congela el resto y la
  reproduccion se corta.

  Los cuadros se sirven con cache inmutable. Un cuadro dado de un video dado no cambia
  nunca, asi que el cache del navegador termina cumpliendo el papel de buffer del lado del
  cliente: retroceder sobre lo ya visto no genera ni una peticion.

QUE HACE
  Despacha rutas, sirve los archivos estaticos del cliente y traduce las respuestas de
  api.py a HTTP.

USO
  serve(Path("videos/spar.mp4"), host="0.0.0.0", port=8000)
"""

from __future__ import annotations

import json
import re
import socket
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from boxtwin.server.api import Api
from boxtwin.server.frames import FrameRenderer
from boxtwin.server.session import Session

__all__ = ["serve", "build_handler", "WEB_ROOT"]

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"

TIPOS = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
}

RUTA_FRAME = re.compile(r"^/api/frame/(\d+)\.jpg$")
RUTA_HIRES = re.compile(r"^/api/hires/(\d+)\.jpg$")


class _Handler(BaseHTTPRequestHandler):
    server_version = "boxtwin"
    protocol_version = "HTTP/1.1"

    api: Api  # inyectado por build_handler

    # -- utilidades --------------------------------------------------------

    def _responder(
        self, codigo: int, cuerpo: bytes, tipo: str, *, cache: str | None = None
    ) -> None:
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        if cache:
            self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(cuerpo)

    def _json(self, data: Any, codigo: int = 200) -> None:
        cuerpo = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._responder(codigo, cuerpo, "application/json; charset=utf-8", cache="no-store")

    def _error(self, codigo: int, mensaje: str) -> None:
        self._json({"error": mensaje}, codigo)

    def log_message(self, formato: str, *args) -> None:  # noqa: A003
        """Silencio por defecto: una linea por cuadro haria ilegible la consola."""
        return

    # -- despacho ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        ruta = url.path
        query = parse_qs(url.query)

        try:
            if ruta in ("/", "/index.html"):
                return self._estatico("index.html")
            if ruta.startswith("/static/"):
                return self._estatico(ruta[len("/static/") :])

            m = RUTA_FRAME.match(ruta)
            if m:
                return self._cuadro(int(m.group(1)), query, hires=False)
            m = RUTA_HIRES.match(ruta)
            if m:
                return self._cuadro(int(m.group(1)), query, hires=True)

            if ruta == "/api/meta":
                return self._json(self.api.meta())
            if ruta == "/api/poses":
                desde = int(query.get("from", ["0"])[0])
                hasta = int(query.get("to", [str(desde + 1)])[0])
                return self._json(self.api.poses(desde, hasta))
            if ruta == "/api/annotation":
                return self._json(self.api.annotation())
            if ruta == "/api/issues":
                return self._json(self.api.issues())
            if ruta == "/api/stats":
                return self._json(self.api.stats())

            self._error(404, f"ruta desconocida: {ruta}")
        except ValueError as exc:
            self._error(400, str(exc))
        except BrokenPipeError:
            # El navegador cancela peticiones de imagen al saltar de cuadro. Es normal.
            return
        except Exception as exc:  # noqa: BLE001
            self._error(500, f"{type(exc).__name__}: {exc}")

    # -- recursos ----------------------------------------------------------

    def _estatico(self, relativo: str) -> None:
        destino = (WEB_ROOT / relativo).resolve()
        if not destino.is_file() or WEB_ROOT not in destino.parents:
            return self._error(404, f"no existe {relativo}")
        tipo = TIPOS.get(destino.suffix, "application/octet-stream")
        # Sin cache: son los archivos que se editan mientras se desarrolla.
        self._responder(200, destino.read_bytes(), tipo, cache="no-cache")

    def _cuadro(self, n: int, query: dict[str, list[str]], *, hires: bool) -> None:
        calidad = int(query.get("q", ["0"])[0]) or None
        if hires:
            datos = self.api.renderer.hires_jpeg(n)
            if datos is None:
                return self._error(404, "no hay video original disponible")
        else:
            datos = self.api.renderer.jpeg(n, quality=calidad)
            if datos is None:
                return self._error(404, f"cuadro {n} fuera del video")
        # Un cuadro dado de un video dado no cambia nunca: el cache del navegador hace de
        # buffer del lado del cliente y retroceder no genera peticiones.
        self._responder(200, datos, "image/jpeg", cache="public, max-age=31536000, immutable")


def build_handler(api: Api) -> type[BaseHTTPRequestHandler]:
    return type("BoxTwinHandler", (_Handler,), {"api": api})


def _ip_local() -> str:
    """IP con la que se llega a esta maquina desde la red, para imprimirla al arrancar."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 80))  # no manda nada, solo elige la interfaz de salida
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def serve(
    video: Path,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    buffer_mb: int = 512,
    quality: int = 85,
    stamp: bool = False,
) -> int:
    session = Session.open(video, buffer_mb=buffer_mb)
    if not session.source.verify_exact_seek():
        session.close()
        raise RuntimeError(
            f"{session.source.path.name} no permite navegacion exacta por numero de cuadro. "
            "No se puede anotar sobre este archivo: las fronteras saldrian corridas."
        )

    api = Api(session, FrameRenderer(session, quality=quality, stamp=stamp))
    servidor = ThreadingHTTPServer((host, port), build_handler(api))
    servidor.daemon_threads = True

    ip = _ip_local()
    print(f"anotador de {session.paths.video.name}: {session.total_frames} cuadros a {session.fps:.3f} fps")
    print(f"  fuente     : {'proxy' if session.using_proxy else 'original'}")
    print(f"  anotacion  : {session.paths.annot}")
    if stamp:
        print("  MODO DIAGNOSTICO: cada cuadro lleva su numero escrito encima")
    print()
    print(f"  abrir en el navegador:  http://{ip}:{port}/")
    if host in ("0.0.0.0", "::"):
        print(f"  (desde esta maquina:    http://127.0.0.1:{port}/ )")
    else:
        print(f"  escuchando solo en {host}: hace falta tunel ssh -L {port}:localhost:{port}")
    print(flush=True)

    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\ncerrando")
    finally:
        servidor.server_close()
        session.close()
    return 0
