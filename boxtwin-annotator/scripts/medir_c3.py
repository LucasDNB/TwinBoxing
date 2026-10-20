"""
BoxTwin - Medicion de C3: el indicador de guardia contra 100 golpes marcados a mano.

POR QUE EXISTE
  C3 es el criterio pre-registrado del indicador de guardia (docs/specs/MVP.md §7): acuerdo
  >= 80% en cada uno de los dos indicadores (volvio a la guardia dentro de 1 s, mano opuesta
  caida) contra 100 golpes de sparring-3 marcados por Lucas. Si no pasa, F5 sale del MVP. Es
  lo unico del MVP que no venia medido de antes, y el 22-09 sobre el amateur el indicador con
  radio 0,6 marcaba la mano caida en 40 de 41 golpes. Las reglas de abajo estan fijadas en
  docs/specs/2026-10-20-c3-guardia-preregistro.md §4 antes de sortear.

QUE HACE
  muestra  sortea 100 de los golpes anotados completos (semilla fija) y escribe muestra.json.
  clips    un mp4 por golpe, de 0,5 s antes del inicio a 1,5 s despues del fin anotado, con
           una caja sobre el peleador, el brazo escrito y un contador desde el fin del golpe.
           Nada que salga del indicador: ni esqueleto ni zona.
  marcar   un servidor local con una pagina para marcar los clips. Cada respuesta se escribe
           en marcas.json al momento (reanudable). No carga el indicador.
  medir    con los 100 marcados, corre el indicador por el mismo camino que el producto
           (series_por_peleador con la identidad anotada, guardia.medir) y escribe el acuerdo,
           Wilson, kappa y las matrices de confusion en resultado.json.
  Lee la anotacion y la cache; no escribe nada en ellas.

USO
  python scripts/medir_c3.py muestra --raiz ~/Proyectos/TwinBoxing -d ~/boxtwin/mediciones/c3
  python scripts/medir_c3.py clips   --raiz ~/Proyectos/TwinBoxing -d ~/boxtwin/mediciones/c3
  python scripts/medir_c3.py marcar  -d ~/boxtwin/mediciones/c3      (abre http://127.0.0.1:8765)
  python scripts/medir_c3.py medir   --raiz ~/Proyectos/TwinBoxing -d ~/boxtwin/mediciones/c3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

UMBRAL = 0.80
SEMILLA = 20102026
N_MUESTRA = 100
ANTES_S, DESPUES_S, LIMITE_S = 0.5, 1.5, 1.0
FUENTE = "anotacion-sparring-3"
ANNOT = f"{FUENTE}/annotations/sparring-3-rounds.annot.json"
RESPUESTAS = ("si", "no", "no_se_ve")
INDICADORES = ("volvio", "caida")


# ---------------------------------------------------------------------------
# Partes puras (con pruebas)
# ---------------------------------------------------------------------------


def candidatos_de(eventos: list[dict]) -> list[dict]:
    """Los golpes completos, ordenados por id: la poblacion del sorteo (pre-registro §4)."""
    return sorted((e for e in eventos if e.get("completeness") == "full"), key=lambda e: e["id"])


def sortear(eventos: list[dict], n: int = N_MUESTRA, semilla: int = SEMILLA) -> list[dict]:
    """Muestreo aleatorio simple sin reposicion. Devuelve en el orden del sorteo."""
    pob = candidatos_de(eventos)
    if len(pob) < n:
        raise ValueError(f"hay {len(pob)} golpes completos y se piden {n}")
    return random.Random(semilla).sample(pob, n)


def golpe_de(e: dict, fps: float) -> SimpleNamespace:
    """Un evento anotado con la forma de golpe que espera guardia.medir (fin inclusivo)."""
    return SimpleNamespace(
        id=e["id"], peleador=e["fighter"][-1], brazo=e["side"],
        inicio=int(e["start_frame"]), fin=int(e["end_frame"]),
        t_inicio=int(e["start_frame"]) / fps,
    )


def lectura_indicador(ev) -> dict[str, str | None]:
    """
    Las dos respuestas del indicador para un golpe, en el mismo vocabulario que las marcas.
    None es no medible y no entra al acuerdo.
    """
    if not ev.medible:
        volvio = None
    else:
        volvio = "si" if ev.retorno_ms is not None else "no"
    caida = None if ev.mano_opuesta_caida is None else ("si" if ev.mano_opuesta_caida else "no")
    return {"volvio": volvio, "caida": caida}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - m, 4), round(c + m, 4))


def kappa(pares: list[tuple[str, str]]) -> float | None:
    """Kappa de Cohen para dos clases. None si el acuerdo esperado es 1 (una sola clase)."""
    n = len(pares)
    if n == 0:
        return None
    po = sum(a == b for a, b in pares) / n
    ca, cb = Counter(a for a, _ in pares), Counter(b for _, b in pares)
    pe = sum(ca[c] * cb[c] for c in ("si", "no")) / (n * n)
    if pe >= 1.0:
        return None
    return round((po - pe) / (1 - pe), 4)


def acuerdo(lucas: list[str], indicador: list[str | None]) -> dict:
    """
    Acuerdo de un indicador sobre los golpes con si/no de los dos lados. Los no_se_ve de
    Lucas y los no medibles del indicador se cuentan aparte.
    """
    if len(lucas) != len(indicador):
        raise ValueError("las dos listas tienen que tener un elemento por golpe")
    pares = [(l, i) for l, i in zip(lucas, indicador) if l in ("si", "no") and i in ("si", "no")]
    k = sum(l == i for l, i in pares)
    n = len(pares)
    conf = {f"lucas_{l}_indicador_{i}": sum(1 for p in pares if p == (l, i))
            for l in ("si", "no") for i in ("si", "no")}
    return {
        "n": n,
        "aciertos": k,
        "acuerdo": round(k / n, 4) if n else None,
        "wilson95": wilson(k, n),
        "kappa": kappa(pares),
        "confusion": conf,
        "lucas_no_se_ve": sum(1 for l in lucas if l == "no_se_ve"),
        "indicador_no_medible": sum(1 for i in indicador if i is None),
        "pasa": bool(n) and k / n >= UMBRAL,
    }


def faltan(muestra: list[dict], marcas: dict) -> list[str]:
    """Los ids de la muestra que no tienen las dos respuestas validas."""
    return [g["id"] for g in muestra
            if not all(marcas.get(g["id"], {}).get(k) in RESPUESTAS for k in INDICADORES)]


# ---------------------------------------------------------------------------
# Entrada y salida
# ---------------------------------------------------------------------------


def _leer_json(p: Path, defecto=None):
    return json.loads(p.read_text()) if p.is_file() else defecto


def _escribir_json(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, p)       # atomico: un corte a mitad no deja el archivo roto


def _cmd_muestra(args) -> int:
    d = Path(args.dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    destino = d / "muestra.json"
    if destino.is_file():
        print(f"ya existe {destino}; no se vuelve a sortear")
        return 0
    annot = json.loads((Path(args.raiz).expanduser() / ANNOT).read_text())
    fps = float(annot["video"]["fps"])
    elegidos = sortear(annot["events"])
    golpes = [{"id": e["id"], "fighter": e["fighter"], "side": e["side"],
               "start_frame": e["start_frame"], "end_frame": e["end_frame"]} for e in elegidos]
    dist = Counter(f"{g['fighter'][-1]}-{g['side']}" for g in golpes)
    _escribir_json(destino, {"semilla": SEMILLA, "poblacion": len(candidatos_de(annot["events"])),
                             "fps": fps, "distribucion": dict(sorted(dist.items())),
                             "golpes": golpes})
    print(f"{len(golpes)} golpes de {len(candidatos_de(annot['events']))}: {dict(sorted(dist.items()))}")
    return 0


def _bbox(kp, sc, ancho, alto):
    import numpy as np

    ok = sc >= 0.3
    if ok.sum() < 3:
        return None
    xs, ys = kp[ok, 0], kp[ok, 1]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mx, my = 0.15 * (x1 - x0) + 10, 0.10 * (y1 - y0) + 10
    return (int(max(0, x0 - mx)), int(max(0, y0 - my)),
            int(min(ancho - 1, x1 + mx)), int(min(alto - 1, y1 + my)))


def _cmd_clips(args) -> int:
    import cv2

    from boxtwin.core.annotations import load as cargar_doc
    from boxtwin.core.identity import IdentityResolver
    from boxtwin.core.posecache import PoseCache
    from boxtwin.core.types import FighterId

    raiz = Path(args.raiz).expanduser()
    d = Path(args.dir).expanduser()
    muestra = json.loads((d / "muestra.json").read_text())
    clips = d / "clips"
    clips.mkdir(exist_ok=True)
    doc, _ = cargar_doc(raiz / ANNOT)
    cache = PoseCache.open(raiz / FUENTE / "cache" / "sparring-3-rounds.pose.npz")
    resolver = IdentityResolver(doc, cache)
    fps = float(muestra["fps"])
    video = raiz / FUENTE / "videos" / "sparring-3-rounds.mp4"
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    escala = 1280 / W
    brazo_txt = {"left": "BRAZO IZQUIERDO", "right": "BRAZO DERECHO"}

    for n, g in enumerate(muestra["golpes"], 1):
        destino = clips / f"{g['id']}.mp4"
        if destino.is_file():
            continue
        fid = FighterId.A if g["fighter"].endswith("A") else FighterId.B
        ini = max(0, g["start_frame"] - round(ANTES_S * fps))
        fin = min(int(doc.video.total_frames) - 1, g["end_frame"] + round(DESPUES_S * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, ini)
        ff = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"1280x{round(H * escala)}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
             str(destino) + ".tmp.mp4"],
            stdin=subprocess.PIPE)
        caja = None
        for f in range(ini, fin + 1):
            ok, img = cap.read()
            if not ok:
                break
            pose = resolver.by_fighter(f).get(fid)
            if pose is not None:
                caja = _bbox(pose.keypoints[:17], pose.kp_score[:17], W, H) or caja
            img = cv2.resize(img, (1280, round(H * escala)))
            if caja is not None:
                x0, y0, x1, y1 = (round(v * escala) for v in caja)
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)
            if f < g["start_frame"]:
                fase, color = "ANTES DEL GOLPE", (200, 200, 200)
            elif f <= g["end_frame"]:
                fase, color = "GOLPE", (0, 200, 255)
            else:
                t = (f - g["end_frame"]) / fps
                if t <= LIMITE_S:
                    fase, color = f"+{t:.2f} s desde el fin", (0, 220, 0)
                else:
                    fase, color = f"+{t:.2f} s  FUERA DEL LIMITE", (0, 0, 255)
            cv2.rectangle(img, (0, 0), (1280, 44), (0, 0, 0), -1)
            cv2.putText(img, f"{g['fighter'][-1]}  {brazo_txt[g['side']]}   {fase}", (12, 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            ff.stdin.write(img.tobytes())
        ff.stdin.close()
        if ff.wait() != 0:
            raise RuntimeError(f"ffmpeg fallo en {g['id']}")
        os.replace(str(destino) + ".tmp.mp4", destino)
        print(f"{n}/{len(muestra['golpes'])} {g['id']}", flush=True)
    return 0


PAGINA = r"""<!doctype html><html lang="es"><meta charset="utf-8">
<title>C3 marcado</title>
<style>
body{background:#111;color:#eee;font:16px system-ui,sans-serif;margin:0;padding:16px}
video{width:100%;max-width:1280px;display:block;background:#000}
.fila{display:flex;gap:24px;flex-wrap:wrap;margin:12px 0}
.preg b{display:block;margin-bottom:6px}
button{font-size:16px;padding:8px 14px;margin-right:6px;background:#333;color:#eee;border:1px solid #666;border-radius:6px;cursor:pointer}
button.on{background:#2a7;border-color:#2a7;color:#000}
#estado{color:#aaa}.ayuda{color:#888;font-size:14px}
</style>
<div id="estado"></div>
<video id="v" loop muted playsinline></video>
<div class="fila">
 <div class="preg"><b>Volvio a la guardia dentro de 1,0 s del fin</b>
  <button data-k="volvio" data-r="si">1 si</button><button data-k="volvio" data-r="no">2 no</button><button data-k="volvio" data-r="no_se_ve">3 no se ve</button></div>
 <div class="preg"><b>Mano opuesta caida durante el golpe</b>
  <button data-k="caida" data-r="si">Q si</button><button data-k="caida" data-r="no">W no</button><button data-k="caida" data-r="no_se_ve">E no se ve</button></div>
</div>
<div class="fila"><button id="ant">&larr; anterior</button><button id="sig">siguiente &rarr;</button>
<button id="sigf">proximo sin marcar</button></div>
<p class="ayuda">Espacio pausa. , y . un cuadro. Z X C velocidad 0,25 / 0,5 / 1. Enter siguiente.
Cada respuesta se guarda al momento.</p>
<script>
let golpes=[],marcas={},i=0;const v=document.getElementById('v');
async function cargar(){const r=await fetch('/estado');const d=await r.json();golpes=d.golpes;marcas=d.marcas;
 i=Math.max(0,golpes.findIndex(g=>!completo(g.id)));if(i<0)i=0;mostrar();}
function completo(id){const m=marcas[id]||{};return m.volvio&&m.caida;}
function mostrar(){const g=golpes[i];v.src='/clips/'+g.id+'.mp4';v.playbackRate=0.5;v.play();
 const hechos=golpes.filter(x=>completo(x.id)).length;
 document.getElementById('estado').textContent=`Golpe ${i+1} de ${golpes.length} (${g.id}) - marcados ${hechos}`;
 const m=marcas[g.id]||{};document.querySelectorAll('button[data-k]').forEach(b=>b.classList.toggle('on',m[b.dataset.k]===b.dataset.r));}
async function marcar(k,r){const id=golpes[i].id;marcas[id]=Object.assign({},marcas[id],{[k]:r});
 await fetch('/marca',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id,k,r})});mostrar();}
function ir(d){i=Math.min(golpes.length-1,Math.max(0,i+d));mostrar();}
document.querySelectorAll('button[data-k]').forEach(b=>b.onclick=()=>marcar(b.dataset.k,b.dataset.r));
document.getElementById('ant').onclick=()=>ir(-1);document.getElementById('sig').onclick=()=>ir(1);
document.getElementById('sigf').onclick=()=>{const j=golpes.findIndex(g=>!completo(g.id));if(j>=0){i=j;mostrar();}};
const teclas={'1':['volvio','si'],'2':['volvio','no'],'3':['volvio','no_se_ve'],'q':['caida','si'],'w':['caida','no'],'e':['caida','no_se_ve']};
document.onkeydown=e=>{const t=e.key.toLowerCase();
 if(teclas[t])marcar(...teclas[t]);else if(t===' '){e.preventDefault();v.paused?v.play():v.pause();}
 else if(t===','){v.pause();v.currentTime-=1/30;}else if(t==='.'){v.pause();v.currentTime+=1/30;}
 else if(t==='z')v.playbackRate=0.25;else if(t==='x')v.playbackRate=0.5;else if(t==='c')v.playbackRate=1;
 else if(t==='enter')ir(1);else if(t==='arrowleft')ir(-1);else if(t==='arrowright')ir(1);};
cargar();
</script></html>"""


def _cmd_marcar(args) -> int:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    d = Path(args.dir).expanduser()
    muestra = json.loads((d / "muestra.json").read_text())
    ruta_marcas = d / "marcas.json"
    ids = {g["id"] for g in muestra["golpes"]}
    # Lo que ve la pagina: id, peleador y brazo. Nada del indicador.
    publico = [{"id": g["id"], "fighter": g["fighter"], "side": g["side"]}
               for g in muestra["golpes"]]

    class Manejador(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _enviar(self, codigo, cuerpo: bytes, tipo: str):
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._enviar(200, PAGINA.encode(), "text/html; charset=utf-8")
            if self.path == "/estado":
                cuerpo = {"golpes": publico, "marcas": _leer_json(ruta_marcas, {})}
                return self._enviar(200, json.dumps(cuerpo).encode(), "application/json")
            if self.path.startswith("/clips/"):
                nombre = Path(self.path).name
                p = d / "clips" / nombre
                if nombre.removesuffix(".mp4") in ids and p.is_file():
                    return self._enviar(200, p.read_bytes(), "video/mp4")
            self._enviar(404, b"no", "text/plain")

        def do_POST(self):
            if self.path != "/marca":
                return self._enviar(404, b"no", "text/plain")
            datos = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if datos.get("id") not in ids or datos.get("k") not in INDICADORES \
                    or datos.get("r") not in RESPUESTAS:
                return self._enviar(400, b"marca invalida", "text/plain")
            marcas = _leer_json(ruta_marcas, {})
            marcas.setdefault(datos["id"], {})[datos["k"]] = datos["r"]
            _escribir_json(ruta_marcas, marcas)
            self._enviar(200, b"ok", "text/plain")

    srv = ThreadingHTTPServer(("127.0.0.1", args.puerto), Manejador)
    print(f"marcado en http://127.0.0.1:{args.puerto}  (Ctrl+C para cortar; lo marcado queda)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def _cmd_medir(args) -> int:
    from boxtwin.core.annotations import load as cargar_doc
    from boxtwin.core.posecache import PoseCache
    from boxtwin.mvp.guardia import ConfigGuardia, medir
    from boxtwin.mvp.orquesta import series_por_peleador

    raiz = Path(args.raiz).expanduser()
    d = Path(args.dir).expanduser()
    muestra = json.loads((d / "muestra.json").read_text())
    marcas = _leer_json(d / "marcas.json", {})
    pendientes = faltan(muestra["golpes"], marcas)
    if pendientes:
        print(f"faltan marcar {len(pendientes)} golpes; el acuerdo no se calcula antes "
              f"(pre-registro §4). Primero: {pendientes[:5]}")
        return 1

    annot = json.loads((raiz / ANNOT).read_text())
    doc, _ = cargar_doc(raiz / ANNOT)
    cache = PoseCache.open(raiz / FUENTE / "cache" / "sparring-3-rounds.pose.npz")
    kp, sc, _, _ = series_por_peleador(doc, cache)
    fps = float(doc.video.fps)
    # Todos los golpes anotados entran, para que el tope del proximo golpe del mismo brazo
    # sea el real; despues se leen solo los de la muestra.
    todos = [golpe_de(e, fps) for e in annot["events"]]
    eventos = {ev.golpe: ev for ev in medir(kp, sc, todos, fps=fps, cfg=ConfigGuardia())}

    por_golpe, lecturas = [], {k: ([], []) for k in INDICADORES}
    for g in muestra["golpes"]:
        ev = eventos[g["id"]]
        lec = lectura_indicador(ev)
        for k in INDICADORES:
            lecturas[k][0].append(marcas[g["id"]][k])
            lecturas[k][1].append(lec[k])
        por_golpe.append({"id": g["id"], "lucas": marcas[g["id"]], "indicador": lec,
                          "retorno_ms": ev.retorno_ms,
                          "fraccion_opuesta_afuera": ev.fraccion_opuesta_afuera,
                          "motivo": ev.motivo})
    res = {k: acuerdo(*lecturas[k]) for k in INDICADORES}
    res["pasa"] = all(res[k]["pasa"] for k in INDICADORES)
    res["umbral"] = UMBRAL
    res["config"] = vars(ConfigGuardia())
    _escribir_json(d / "resultado.json", {"resumen": res, "golpes": por_golpe})
    for k, nombre in (("volvio", "volvio a la guardia"), ("caida", "mano opuesta caida")):
        r = res[k]
        print(f"{nombre:<22} {r['aciertos']}/{r['n']} = {r['acuerdo']}  Wilson {r['wilson95']}"
              f"  kappa {r['kappa']}  no se ve {r['lucas_no_se_ve']}"
              f"  no medible {r['indicador_no_medible']}  -> {'PASA' if r['pasa'] else 'NO PASA'}")
    print(f"C3: {'PASA' if res['pasa'] else 'NO PASA'} (umbral {UMBRAL} en cada uno)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nombre, fn in (("muestra", _cmd_muestra), ("clips", _cmd_clips),
                       ("marcar", _cmd_marcar), ("medir", _cmd_medir)):
        p = sub.add_parser(nombre)
        p.add_argument("-d", "--dir", default="~/boxtwin/mediciones/c3")
        if nombre != "marcar":
            p.add_argument("--raiz", default="~/Proyectos/TwinBoxing")
        else:
            p.add_argument("--puerto", type=int, default=8765)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
