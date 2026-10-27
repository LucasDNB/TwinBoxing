"""
BoxTwin - Rehace la evidencia de guante de una sesion y la compara con la guardada.

POR QUE EXISTE
  El 20-10 se cambio la lectura del video en `analizar`: de un salto con CAP_PROP_POS_FRAMES
  cada 5 cuadros a lectura en orden con grab(). Era la mitad del tiempo de procesar (C5). Esa
  evidencia alimenta la identidad, que ya se midio (C1) con la version vieja, asi que el
  cambio solo vale si la evidencia sale igual. Si el salto caia en cuadros equivocados, no
  sale igual, y eso tambien hay que saberlo.

QUE HACE
  Por cada directorio de sesion: corre `analizar` con el codigo importado (correr con
  PYTHONPATH apuntando a la version que se quiere probar), cronometra, y compara track por
  track contra el `evidencia.json` de la sesion: conjunto de tracks, recortes, cuadros con
  guante, alto, posiciones y colores (con tolerancia, porque la GPU no es determinista al
  ultimo decimal). Escribe la evidencia nueva y un resumen en --salida; no toca la sesion.

USO
  PYTHONPATH=src python scripts/comparar_evidencia.py ~/boxtwin/datos/sesiones/<id> \
      --modelo ~/Proyectos/TwinBoxing/boxtwin-guantes/modelos/guantes-v2.pt -o salida/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

TOL = 1e-3


def comparar(vieja: dict, nueva: dict, tol: float = TOL) -> dict:
    """
    Compara dos evidencias en el formato de guardar_evidencia (lista de tracks). Devuelve
    cuantos tracks son iguales y, para los distintos, en que campo difieren.
    """
    a = {t["track_id"]: t for t in vieja["tracks"]}
    b = {t["track_id"]: t for t in nueva["tracks"]}
    distintos: dict[int, list[str]] = {}

    def cerca(x, y) -> bool:
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(cerca(u, v) for u, v in zip(x, y))
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return abs(float(x) - float(y)) <= tol
        return x == y

    for t in sorted(set(a) & set(b)):
        campos = [k for k in a[t] if k != "track_id" and not cerca(a[t][k], b[t].get(k))]
        if campos:
            distintos[t] = campos
    return {
        "tracks_vieja": len(a), "tracks_nueva": len(b),
        "solo_en_vieja": sorted(set(a) - set(b)), "solo_en_nueva": sorted(set(b) - set(a)),
        "iguales": len(set(a) & set(b)) - len(distintos),
        "distintos": {str(k): v for k, v in distintos.items()},
        "identicas": not distintos and set(a) == set(b),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sesiones", nargs="+")
    ap.add_argument("--modelo", required=True)
    ap.add_argument("-o", "--salida", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import boxtwin
    from boxtwin.core.annotations import load as cargar_doc
    from boxtwin.core.deteccion_guantes import DetectorGuantes
    from boxtwin.core.identidad_auto import ConfigIdentidadAuto, analizar, guardar_evidencia
    from boxtwin.core.posecache import PoseCache
    from boxtwin.core.project import project_paths

    print(f"codigo: {boxtwin.__file__}")
    salida = Path(args.salida).expanduser()
    salida.mkdir(parents=True, exist_ok=True)
    det = DetectorGuantes(str(Path(args.modelo).expanduser()), device=args.device)
    resumen = []
    for s in args.sesiones:
        d = Path(s).expanduser()
        meta = json.loads((d / "sesion.json").read_text())
        video = Path(meta["video"]["ruta"])
        paths = project_paths(video)
        doc, _ = cargar_doc(paths.annot)
        cache = PoseCache.open(paths.npz)
        t0 = time.perf_counter()
        ev = analizar(cache, video, det, ConfigIdentidadAuto(), alto_imagen=doc.video.height,
                      fps=doc.video.fps, total_frames=doc.video.total_frames)
        seg = round(time.perf_counter() - t0, 1)
        destino = salida / f"{d.name}.evidencia.json"
        guardar_evidencia(ev, destino)
        cmp = comparar(json.loads((d / "evidencia.json").read_text()),
                       json.loads(destino.read_text()))
        antes = next((e["segundos"] for e in meta.get("etapas", []) if e["etapa"] == "evidencia"), None)
        fila = {"sesion": d.name, "duracion_s": meta["video"]["duracion_s"],
                "segundos_antes": antes, "segundos_ahora": seg, **cmp}
        resumen.append(fila)
        print(f"{d.name[:8]}  video {fila['duracion_s']:.0f} s  evidencia {antes} s -> {seg} s"
              f"  tracks {cmp['tracks_vieja']}/{cmp['tracks_nueva']}  iguales {cmp['iguales']}"
              f"  identicas {cmp['identicas']}  distintos {list(cmp['distintos'].items())[:5]}",
              flush=True)
    (salida / "resumen.json").write_text(json.dumps(resumen, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
