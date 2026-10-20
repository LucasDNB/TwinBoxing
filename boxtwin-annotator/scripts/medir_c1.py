"""
BoxTwin - Medicion de C1: identidad con siembra humana sobre las seis fuentes de gimnasio.

POR QUE EXISTE
  C1 es el criterio pre-registrado de la identidad (docs/specs/MVP.md §7): tracks bien
  asignados con siembra humana sobre las seis fuentes de gimnasio, umbral >= 95%. Si no
  pasa, el plan B es la correccion manual por track en la web, que ya existe. Estaba
  comprometido para el 6-10 y despues para el 20-10, y nunca se corrio porque faltaba
  armar la siembra simulada. Las reglas de abajo se fijan en
  docs/specs/2026-10-20-c1-preregistro.md antes de correr.

QUE HACE
  Por cada fuente, con el cache de pose del proyecto de anotacion (los track_id tienen que
  ser los mismos que los de la anotacion):
  1. Evidencia de guante: `analizar`, o la guardada si ya existe (reanudable).
  2. Propuesta automatica y candidatos, como `procesar` en el producto.
  3. Siembra simulada, con la regla fija del pre-registro: si el par sugerido son dos
     peleadores distintos segun la anotacion, la persona lo confirma; si no, elige entre
     los candidatos que muestra la pantalla, en orden, el primero que es A y el primero que
     es B. Si no hay uno de cada lado, la fuente queda "sin siembra posible".
  4. Propuesta con esas semillas, como `completar` en el producto.
  5. Puntuacion contra la anotacion manual, sin probar la correspondencia cruzada: con
     semillas, A es A. La anotacion asigna por rango de cuadros; la verdad de un track es su
     rol mayoritario por cuadros entre A y B.
  Escribe un JSON por fuente y un resumen con el veredicto.

  Primaria (pre-registrada): aciertos sobre tracks evaluables (peleadores de la anotacion
  que la propuesta decidio como A o B), agregado sobre las seis fuentes. Es la misma
  definicion que el 82,4% sin siembra y el 99,1% de color, asi que se puede poner al lado.
  Al lado, sin umbral: cobertura (aciertos sobre todos los tracks de peleador de la
  anotacion, lo no decidido cuenta como error) y la cobertura pesada por cuadros de
  peleador, que cuenta cada tramo de un track mixto con su propio rol.

USO
  python scripts/medir_c1.py medir --raiz ~/Proyectos/TwinBoxing \
      --modelo ~/boxtwin/modelos/guantes-v2.pt -o ~/boxtwin/mediciones/c1
  python scripts/medir_c1.py resumen ~/boxtwin/mediciones/c1
  (con --video se reemplaza la lista de fuentes; con --device cpu corre sin GPU)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

UMBRAL = 0.95

# Las seis fuentes de gimnasio, relativas a la raiz del repo. Pacquiao-Margarito es
# transmision y no entra en C1.
FUENTES = [
    "videos/Sparring.mp4",
    "anotacion-spar-01/videos/01-sparring.mp4",
    "anotacion-spar-02/videos/02-sparring.mp4",
    "anotacion-spar-03/videos/03-sparring.mp4",
    "anotacion-spar-04/videos/04-sparring.mp4",
    "anotacion-sparring-3/videos/sparring-3-rounds.mp4",
]

A, B = "fighter_A", "fighter_B"


# ---------------------------------------------------------------------------
# Partes puras (con pruebas)
# ---------------------------------------------------------------------------


def verdad_de(asignaciones, frames_de=None) -> tuple[dict[int, str], dict[int, dict]]:
    """
    La verdad por track, desde las asignaciones de la anotacion, que son por rango de cuadros.

    Un track puede ser A en un tramo y B en otro (el tracker cambio de persona), o peleador en
    un tramo e ignore en otro. Devuelve el rol mayoritario por cuadros entre A y B, que es lo
    que una asignacion de track entero puede acertar, y los cuadros de cada rol, que es lo que
    usa la cobertura pesada. `frames_de(track)` da los cuadros reales del track en el cache;
    sin el, se cuenta el largo del rango.
    """
    cuadros: dict[int, dict] = {}
    for a in asignaciones:
        rol = getattr(a.role, "value", a.role)
        if rol not in (A, B):
            continue
        t = int(a.track_id)
        ini, fin = int(a.start_frame), int(a.end_frame_excl)
        if frames_de is None:
            n = max(0, fin - ini)
        else:
            f = frames_de(t)
            n = int(((f >= ini) & (f < fin)).sum())
        d = cuadros.setdefault(t, {A: 0, B: 0})
        d[rol] += n
    # Un track sin cuadros adentro de sus rangos no se puede acertar ni errar: queda afuera.
    cuadros = {t: d for t, d in cuadros.items() if d[A] + d[B] > 0}
    verdad = {t: (A if d[A] >= d[B] else B) for t, d in cuadros.items()}
    return verdad, cuadros


def sembrar(pareja, candidatos: list[int], verdad: dict[int, str]) -> dict:
    """
    La regla fija de la siembra simulada. Devuelve semillas (a, b) y por que.

    La persona ve el par sugerido y los candidatos en el orden de la pantalla. Si el par son
    dos peleadores distintos, solo dice cual es cual. Si no, busca en la lista.
    """
    if pareja is not None:
        r0, r1 = verdad.get(pareja[0]), verdad.get(pareja[1])
        if {r0, r1} == {A, B}:
            a, b = (pareja[0], pareja[1]) if r0 == A else (pareja[1], pareja[0])
            return {"semillas": (a, b), "caso": "par_sugerido"}
    primero = {}
    for t in candidatos:
        r = verdad.get(t)
        if r in (A, B) and r not in primero:
            primero[r] = t
    if A in primero and B in primero:
        caso = "lista" if pareja is not None else "lista_sin_par_sugerido"
        return {"semillas": (primero[A], primero[B]), "caso": caso}
    return {"semillas": None, "caso": "sin_siembra_posible"}


def puntuar(roles: dict[int, str], verdad: dict[int, str],
            cuadros: dict[int, dict] | None = None) -> dict:
    """
    Aciertos sin correspondencia cruzada. `roles` es track -> rol propuesto (valor de texto).
    `cuadros` es track -> {rol: cuadros}, de verdad_de: un track mixto suma bien solo los
    cuadros del rol que se le asigno.
    """
    evaluables = [t for t in verdad if roles.get(t) in (A, B)]
    aciertos = [t for t in evaluables if roles[t] == verdad[t]]
    salida = {
        "peleadores_en_la_verdad": len(verdad),
        "evaluables": len(evaluables),
        "aciertos": len(aciertos),
        "invertidos": len(evaluables) - len(aciertos),
        "descartados": sum(1 for t in verdad if roles.get(t) == "ignore"),
        "sin_decidir": sum(1 for t in verdad if roles.get(t) not in (A, B, "ignore")),
    }
    if cuadros is not None:
        salida["cuadros_peleador"] = sum(d[A] + d[B] for d in cuadros.values())
        salida["cuadros_bien"] = sum(cuadros[t][roles[t]] for t in evaluables if t in cuadros)
        salida["tracks_mixtos"] = sum(1 for d in cuadros.values() if d[A] and d[B])
    return salida


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - m, 4), round(c + m, 4))


def resumir(por_fuente: list[dict]) -> dict:
    """Agrega las fuentes. Una sin siembra posible suma sus peleadores como no evaluados."""
    k = sum(f["puntaje"]["aciertos"] for f in por_fuente if f.get("puntaje"))
    n = sum(f["puntaje"]["evaluables"] for f in por_fuente if f.get("puntaje"))
    total = sum(f["peleadores_en_la_verdad"] for f in por_fuente)
    cb = sum(f["puntaje"].get("cuadros_bien", 0) for f in por_fuente if f.get("puntaje"))
    ct = sum(f.get("cuadros_peleador", 0) for f in por_fuente)
    primaria = round(k / n, 4) if n else None
    return {
        "fuentes": len(por_fuente),
        "completo": len(por_fuente) == len(FUENTES),
        "sin_siembra_posible": [f["fuente"] for f in por_fuente
                                if f["siembra"]["caso"] == "sin_siembra_posible"],
        "aciertos": k,
        "evaluables": n,
        "primaria": primaria,
        "wilson95": wilson(k, n),
        "umbral": UMBRAL,
        "pasa": (primaria is not None and primaria >= UMBRAL
                 and not any(f["siembra"]["caso"] == "sin_siembra_posible" for f in por_fuente)),
        "cobertura_tracks": round(k / total, 4) if total else None,
        "cobertura_cuadros": round(cb / ct, 4) if ct else None,
    }


# ---------------------------------------------------------------------------
# Medicion (necesita el cache de pose, el video y el detector de guantes)
# ---------------------------------------------------------------------------


def medir_fuente(video: Path, modelo: Path, salida: Path, device: str | None) -> dict:
    from boxtwin.core.annotations import load as cargar_doc
    from boxtwin.core.identidad_auto import (
        ConfigIdentidadAuto, analizar, cargar_evidencia, guardar_evidencia, proponer,
    )
    from boxtwin.core.posecache import PoseCache
    from boxtwin.core.project import project_paths
    from boxtwin.mvp.candidatos import elegir

    paths = project_paths(video)
    for p in (paths.npz, paths.annot):
        if not p.is_file():
            raise FileNotFoundError(f"falta {p}")
    doc, _ = cargar_doc(paths.annot)      # solo lectura: la anotacion no se escribe nunca
    cache = PoseCache.open(paths.npz)
    cfg = ConfigIdentidadAuto()
    nombre = video.stem

    ruta_ev = salida / f"{nombre}.evidencia.json"
    t0 = time.perf_counter()
    if ruta_ev.is_file():
        ev = cargar_evidencia(ruta_ev)
    else:
        from boxtwin.core.deteccion_guantes import DetectorGuantes

        det = DetectorGuantes(str(modelo), device=device)
        ev = analizar(cache, paths.video, det, cfg, alto_imagen=doc.video.height,
                      fps=doc.video.fps, total_frames=doc.video.total_frames,
                      progreso=lambda f, fin: None)
        guardar_evidencia(ev, ruta_ev)
    t_ev = round(time.perf_counter() - t0, 1)

    auto = proponer(ev, cfg, doc.video.total_frames, doc.video.fps)
    pareja, cands = elegir(ev, auto.diagnostico.get("nucleo_tracks", []),
                           min_coexistencia=cfg.min_coexistencia, fps=doc.video.fps,
                           separacion_minima=cfg.separacion_minima)
    verdad, cuadros = verdad_de(doc.identity.assignments, cache.frames_of_track)
    siembra = sembrar(pareja, [c.track for c in cands], verdad)

    res = {
        "fuente": nombre,
        "video": str(video),
        "segundos_evidencia": t_ev,
        "peleadores_en_la_verdad": len(verdad),
        "cuadros_peleador": sum(d[A] + d[B] for d in cuadros.values()),
        "tracks_mixtos": sum(1 for d in cuadros.values() if d[A] and d[B]),
        "par_sugerido": list(pareja) if pareja else None,
        "candidatos": [c.track for c in cands],
        "siembra": {"caso": siembra["caso"],
                    "semillas": list(siembra["semillas"]) if siembra["semillas"] else None},
        "sin_siembra": puntuar({t: r.value for t, r in auto.roles.items()}, verdad, cuadros),
        "puntaje": None,
        "avisos": [],
    }
    if siembra["semillas"]:
        prop = proponer(ev, cfg, doc.video.total_frames, doc.video.fps,
                        semillas=siembra["semillas"])
        res["avisos"] = list(prop.avisos)
        if prop.diagnostico.get("semillas_en_conflicto"):
            res["siembra"]["caso"] = "semillas_en_conflicto"
        res["puntaje"] = puntuar({t: r.value for t, r in prop.roles.items()}, verdad, cuadros)
    return res


def _cmd_medir(args) -> int:
    salida = Path(args.salida).expanduser()
    salida.mkdir(parents=True, exist_ok=True)
    raiz = Path(args.raiz).expanduser()
    videos = [Path(v).expanduser() for v in args.video] if args.video else \
             [raiz / f for f in FUENTES]
    for v in videos:
        destino = salida / f"{v.stem}.c1.json"
        if destino.is_file() and not args.rehacer:
            print(f"{v.stem}: ya medido, se saltea")
            continue
        print(f"{v.stem}: midiendo...", flush=True)
        res = medir_fuente(v, Path(args.modelo).expanduser(), salida, args.device)
        destino.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        p = res["puntaje"]
        print(f"  siembra: {res['siembra']['caso']}"
              + (f"  aciertos {p['aciertos']}/{p['evaluables']}" if p else ""))
    return _cmd_resumen(argparse.Namespace(directorio=str(salida)))


def _cmd_resumen(args) -> int:
    d = Path(args.directorio).expanduser()
    fuentes = [json.loads(p.read_text()) for p in sorted(d.glob("*.c1.json"))]
    if not fuentes:
        print("no hay mediciones", file=sys.stderr)
        return 1
    r = resumir(fuentes)
    (d / "resumen.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
    print(f"\n{'fuente':<22}{'siembra':<24}{'aciertos':>10}{'evaluables':>12}"
          f"{'peleadores':>12}{'sin siembra':>13}")
    for f in fuentes:
        p, s = f["puntaje"] or {}, f["sin_siembra"]
        sin = f"{s['aciertos']}/{s['evaluables']}" if s["evaluables"] else "-"
        print(f"{f['fuente']:<22}{f['siembra']['caso']:<24}{p.get('aciertos', '-'):>10}"
              f"{p.get('evaluables', '-'):>12}{f['peleadores_en_la_verdad']:>12}{sin:>13}")
    print(f"\nC1 (primaria): {r['aciertos']}/{r['evaluables']} = {r['primaria']}"
          f"  Wilson 95% {r['wilson95']}  umbral {UMBRAL}  -> "
          + ("PASA" if r["pasa"] else "NO PASA")
          + ("" if r["completo"] else f"  [incompleto: {r['fuentes']} de {len(FUENTES)} fuentes]"))
    print(f"sin umbral: cobertura de tracks {r['cobertura_tracks']}, "
          f"de cuadros {r['cobertura_cuadros']}")
    if r["sin_siembra_posible"]:
        print(f"sin siembra posible: {', '.join(r['sin_siembra_posible'])}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Medicion de C1: identidad con siembra humana.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("medir", help="mide las fuentes y escribe el resumen")
    m.add_argument("--raiz", default=".", help="raiz del repo en la estacion")
    m.add_argument("--video", action="append", help="reemplaza la lista de fuentes")
    m.add_argument("--modelo", required=True, help="pesos del detector de guantes")
    m.add_argument("--device", default=None, help="0, cpu, etc. Por defecto el de ultralytics")
    m.add_argument("-o", "--salida", required=True)
    m.add_argument("--rehacer", action="store_true", help="vuelve a puntuar lo ya medido")
    m.set_defaults(fn=_cmd_medir)
    r = sub.add_parser("resumen", help="rehace el resumen desde los JSON por fuente")
    r.add_argument("directorio")
    r.set_defaults(fn=_cmd_resumen)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
