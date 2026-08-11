"""
BoxTwin - Logica de los endpoints, sin HTTP.

POR QUE EXISTE
  Separar que responde cada endpoint de como se transporta permite testear la API entera
  sin levantar un socket ni un navegador. Los handlers son funciones puras sobre la sesion:
  entran parametros, sale un cuerpo. El modulo http.py solo traduce.

  La carga de poses va por rangos y no por cuadro. Pedirlas de a una duplicaria la cantidad
  de peticiones y el navegador tiene un limite de conexiones concurrentes por origen que se
  gastaria en eso en vez de en traer imagenes. Las coordenadas se redondean a un decimal:
  el subpixel no aporta nada al dibujo y recortar decimales baja el JSON a la mitad.

QUE HACE
  Arma el meta de la sesion, sirve poses por rango y devuelve el documento de anotacion.

USO
  from boxtwin.server.api import Api
  api = Api(session, renderer)
  api.poses(100, 220)
"""

from __future__ import annotations

from typing import Any

from boxtwin.core.constants import (
    COCO17_EDGES,
    COCO17_NAMES,
    GLOVE_EDGES,
    PUNCH_COLORS,
    ROLE_COLOR_A,
    ROLE_COLOR_B,
    ROLE_COLOR_IGNORE,
    ROLE_COLOR_UNASSIGNED,
)
from boxtwin.core.validation import validate_document
from boxtwin.server.frames import FrameRenderer
from boxtwin.server.session import Session

__all__ = ["Api", "MAX_POSE_RANGE"]

# Tope de cuadros por peticion de poses. Con 240 el JSON ronda los 300 KB, que llega en un
# viaje y cubre ocho segundos de anotacion a fps nativo.
MAX_POSE_RANGE = 240


class Api:
    def __init__(self, session: Session, renderer: FrameRenderer) -> None:
        self.session = session
        self.renderer = renderer

    # -- metadatos ---------------------------------------------------------

    def meta(self) -> dict[str, Any]:
        """
        Todo lo que el cliente necesita una sola vez al arrancar.

        Incluye la paleta y el esqueleto porque el overlay se dibuja en el navegador: si el
        cliente tuviera su propia copia de los indices COCO, tarde o temprano dejaria de
        coincidir con la del servidor y el esqueleto saldria cruzado.
        """
        s = self.session
        doc = s.doc
        return {
            "video": {
                "name": s.paths.video.name,
                "width": s.video_size[0],
                "height": s.video_size[1],
                "fps": s.fps,
                "fps_source": doc.video.fps_source.value,
                "total_frames": s.total_frames,
                "duration_s": doc.video.duration_s,
                "codec": doc.video.codec,
            },
            "source": {
                "using_proxy": s.using_proxy,
                "scale": s.source.scale,
                "hires_available": s.hires_available,
            },
            "seams": s.seams,
            "fighters": {
                fid.value: {"label": fd.label, "guard": fd.guard.value}
                for fid, fd in doc.fighters.items()
            },
            "settings": doc.settings_snapshot.model_dump(mode="json"),
            "skeleton": {
                "names": list(COCO17_NAMES) + ["glove_left", "glove_right"],
                "edges": [list(e) for e in COCO17_EDGES],
                "glove_edges": [list(e) for e in GLOVE_EDGES],
            },
            "colors": {
                "fighter_A": list(ROLE_COLOR_A),
                "fighter_B": list(ROLE_COLOR_B),
                "ignore": list(ROLE_COLOR_IGNORE),
                "unassigned": list(ROLE_COLOR_UNASSIGNED),
                "punch": {k: list(v) for k, v in PUNCH_COLORS.items()},
            },
            "annot_path": str(s.paths.annot),
            "migrated": s.migrated,
        }

    # -- poses -------------------------------------------------------------

    def poses(self, desde: int, hasta: int) -> dict[str, Any]:
        """
        Poses resueltas de [desde, hasta), con el rol ya asignado.

        Se manda el rol y no el track_id crudo porque el color del overlay va por rol: un
        track_id cambia en cada oclusion y en cada reanudacion del preproceso, y si el color
        lo siguiera, el mismo peleador cambiaria de color solo.
        """
        total = self.session.total_frames
        desde = max(0, min(desde, total))
        hasta = max(desde, min(hasta, total, desde + MAX_POSE_RANGE))

        cuadros = []
        for f in range(desde, hasta):
            detecciones = []
            for p in self.session.resolver.resolve_frame(f):
                detecciones.append(
                    {
                        "track_id": p.track_id,
                        "role": p.role.value if p.role else None,
                        "conf": round(p.det_conf, 3),
                        "bbox": [round(float(v), 1) for v in p.bbox],
                        # (x, y, score) aplanado: un array de 51 numeros pesa bastante menos
                        # que 17 objetos con claves.
                        "kp": [
                            round(float(v), 1)
                            for i in range(17)
                            for v in (p.keypoints[i][0], p.keypoints[i][1])
                        ],
                        "ks": [round(float(v), 2) for v in p.kp_score],
                        "reliable": p.reliable,
                        "shadowed": p.shadowed,
                        "manual": p.manual,
                    }
                )
            cuadros.append(detecciones)

        return {"from": desde, "to": hasta, "frames": cuadros}

    # -- anotacion ---------------------------------------------------------

    def annotation(self) -> dict[str, Any]:
        return self.session.doc.model_dump(mode="json")

    def issues(self) -> dict[str, Any]:
        """Validaciones a nivel documento, para mostrarlas en el panel lateral."""
        return {
            "issues": [
                {
                    "level": i.level.value,
                    "code": i.code,
                    "message": i.message,
                    "ref": i.ref,
                    "frames": list(i.frames) if i.frames else None,
                }
                for i in validate_document(self.session.doc)
            ]
        }

    def stats(self) -> dict[str, Any]:
        return self.renderer.stats()
