#!/usr/bin/env python3
"""
BoxingVI - Herramienta de reanotacion manual.

POR QUE EXISTE
  Verificacion manual sobre muestras de V1, V2 y V3 encontro tasas de acierto de
  etiqueta cercanas al azar, y la hipotesis inicial fue que V2 y V3 conservaban
  ventanas temporales correctas (el clip contiene un golpe completo y bien
  encuadrado) y fallaban solo en la clase, o sea recuperables reclasificando sin
  resegmentar.

  La reanotacion completa de V2 desmintio eso para V2: 98 de 232 clips (42%) no
  contienen ningun golpe, asi que la segmentacion temporal tambien esta rota, y
  entre los 134 usables la etiqueta original acierta 17 veces (12.7%), por debajo
  del azar de 6 clases (16.7%). Para V3 la hipotesis sigue SIN VERIFICAR: salio
  de mirar 4 clips. Conviene mirar la tasa de "sin golpe" en los primeros ~50
  clips de su reanotacion; si se acerca a la de V2, reclasificar no alcanza.

DISENO
  - Anotacion CIEGA por defecto: no muestra la etiqueta original, para no anclar
    el juicio. Se puede revelar con la tecla E, pero conviene decidir antes.
  - Reanudable: escribe el CSV despues de cada clip. Se corta y se retoma.
  - Camara lenta: los clips son de ~0.4 s; a velocidad normal no se distingue un
    hook de un uppercut. Arranca a 0.25x.
  - Registra tiempo de decision por clip: los clips que llevan mucho son los
    ambiguos, y esa distribucion es material de Cap 4.

TECLAS
  1..6  clase        0  sin golpe / descartar        D  dudoso (revisar despues)
  Space replay       ,/.  mas lento / mas rapido     U  deshacer ultimo
  E     revelar etiqueta original

Uso:
  python boxingvi_annot.py --csv ./clips/manifest_filtrado.csv --videos V3 V2 \
      --out ./clips/reanotado.csv --port 8000

  # Si estas por SSH, desde tu maquina local:
  #   ssh -L 8000:localhost:8000 lucasb@desarrollo-lucas
  # y abris http://localhost:8000
"""

import argparse
import csv
import json
import socketserver
import threading
import time
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

CLASSES = ["Jab", "Cross", "Lead Hook", "Rear Hook", "Lead Uppercut", "Rear Uppercut"]

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Reanotacion BoxingVI</title>
<style>
 :root{--bg:#12151a;--fg:#e8eaed;--dim:#8b949e;--acc:#4a9eff;--ok:#3fb950;--warn:#d29922}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;
      display:flex;height:100vh;overflow:hidden}
 #main{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:16px}
 video{max-width:100%;max-height:64vh;background:#000;border-radius:6px}
 #side{width:290px;border-left:1px solid #262c36;padding:18px;overflow-y:auto;flex-shrink:0}
 h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);
    margin:0 0 10px;font-weight:600}
 .k{display:flex;align-items:center;gap:10px;padding:7px 9px;border-radius:5px;margin-bottom:3px}
 .k:hover{background:#1c2128;cursor:pointer}
 .key{display:inline-flex;align-items:center;justify-content:center;min-width:24px;height:24px;
      background:#262c36;border-radius:4px;font:600 12px ui-monospace,monospace;color:var(--acc)}
 #bar{width:100%;max-width:640px;height:4px;background:#262c36;border-radius:2px;margin:14px 0 6px}
 #fill{height:100%;background:var(--acc);border-radius:2px;transition:width .2s}
 #meta{font-size:13px;color:var(--dim);display:flex;gap:18px;flex-wrap:wrap;justify-content:center}
 #orig{color:var(--warn);font-size:13px;min-height:20px;margin-top:6px}
 #flash{position:fixed;top:26px;left:50%;transform:translateX(-50%);padding:9px 20px;
        border-radius:6px;background:var(--ok);color:#0d1117;font-weight:600;opacity:0;
        transition:opacity .25s;pointer-events:none}
 #done{text-align:center}
 .tal{display:flex;justify-content:space-between;font-size:13px;padding:3px 0;color:var(--dim)}
 .tal b{color:var(--fg);font-variant-numeric:tabular-nums}
</style></head><body>
<div id="main">
  <video id="v" autoplay loop muted playsinline></video>
  <div id="bar"><div id="fill"></div></div>
  <div id="meta">
    <span id="prog"></span><span id="spd">0.25x</span><span id="rate"></span>
  </div>
  <div id="orig"></div>
</div>
<div id="side">
  <h2>Clases</h2><div id="keys"></div>
  <h2 style="margin-top:20px">Control</h2>
  <div class="k"><span class="key">0</span> sin golpe</div>
  <div class="k"><span class="key">D</span> dudoso</div>
  <div class="k"><span class="key">U</span> deshacer</div>
  <div class="k"><span class="key">Spc</span> replay</div>
  <div class="k"><span class="key">, .</span> velocidad</div>
  <div class="k"><span class="key">E</span> ver original</div>
  <h2 style="margin-top:20px">Conteo</h2><div id="tally"></div>
</div>
<div id="flash"></div>
<script>
const CLASSES=%%CLASSES%%;
let q=[],i=0,speed=0.25,t0=0,tally={},hist=[];
const v=document.getElementById('v');
CLASSES.forEach((c,n)=>{document.getElementById('keys').innerHTML+=
  `<div class="k" onclick="mark('${c}')"><span class="key">${n+1}</span>${c}</div>`;});
function flash(t,col){const f=document.getElementById('flash');
  f.textContent=t;f.style.background=col||'#3fb950';f.style.opacity=1;
  setTimeout(()=>f.style.opacity=0,420);}
function drawTally(){let h='';for(const k of [...CLASSES,'sin golpe','dudoso'])
  if(tally[k])h+=`<div class="tal"><span>${k}</span><b>${tally[k]}</b></div>`;
  document.getElementById('tally').innerHTML=h;}
function show(){
  if(i>=q.length){document.getElementById('main').innerHTML=
    '<div id="done"><h1>Listo</h1><p style="color:#8b949e">'+q.length+
    ' clips anotados. Cerra la ventana y frena el server con Ctrl+C.</p></div>';return;}
  const c=q[i];v.src='/clip?p='+encodeURIComponent(c.clip);v.playbackRate=speed;
  document.getElementById('prog').textContent=(i+1)+' / '+q.length;
  document.getElementById('fill').style.width=(i/q.length*100)+'%';
  document.getElementById('orig').textContent='';
  const el=Object.values(tally).reduce((a,b)=>a+b,0);
  t0=Date.now();}
function mark(cls){
  if(i>=q.length)return;
  const dt=(Date.now()-t0)/1000;
  hist.push(i);
  fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({...q[i],nueva_cls:cls,segundos:Math.round(dt*10)/10})});
  tally[cls]=(tally[cls]||0)+1;drawTally();
  flash(cls,cls==='sin golpe'?'#d29922':cls==='dudoso'?'#8b949e':'#3fb950');
  i++;show();}
document.addEventListener('keydown',e=>{
  const k=e.key.toLowerCase();
  if(k===' '){e.preventDefault();v.currentTime=0;v.play();return;}
  if(k>='1'&&k<='6'){mark(CLASSES[+k-1]);return;}
  if(k==='0'){mark('sin golpe');return;}
  if(k==='d'){mark('dudoso');return;}
  if(k==='u'){if(hist.length){i=hist.pop();show();flash('deshecho','#8b949e');}return;}
  if(k===','){speed=Math.max(0.1,speed-0.05);}
  if(k==='.'){speed=Math.min(2,speed+0.05);}
  if(k===','||k==='.'){v.playbackRate=speed;
    document.getElementById('spd').textContent=speed.toFixed(2)+'x';return;}
  if(k==='e'){document.getElementById('orig').textContent='etiqueta original: '+q[i].cls;}
});
fetch('/queue').then(r=>r.json()).then(d=>{q=d;drawTally();show();});
</script></body></html>"""


class Handler(SimpleHTTPRequestHandler):
    queue = []
    out_path = None
    lock = threading.Lock()
    done = 0

    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            html = PAGE.replace("%%CLASSES%%", json.dumps(CLASSES))
            self._send(200, "text/html; charset=utf-8", html.encode())
        elif u.path == "/queue":
            self._send(200, "application/json", json.dumps(Handler.queue).encode())
        elif u.path == "/clip":
            from urllib.parse import parse_qs
            p = Path(parse_qs(u.query).get("p", [""])[0])
            if not p.exists():
                self._send(404, "text/plain", b"no existe")
                return
            data = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "none")
            self.end_headers()
            self.wfile.write(data)
        else:
            self._send(404, "text/plain", b"nada")

    def do_POST(self):
        if urlparse(self.path).path != "/save":
            self._send(404, "text/plain", b"nada")
            return
        n = int(self.headers.get("Content-Length", 0))
        rec = json.loads(self.rfile.read(n))
        with Handler.lock:
            new = not Handler.out_path.exists()
            with open(Handler.out_path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=[
                    "clip", "video_key", "start_frame", "end_frame",
                    "cls_original", "nueva_cls", "segundos"])
                if new:
                    w.writeheader()
                w.writerow({"clip": rec["clip"], "video_key": rec["video_key"],
                            "start_frame": rec["start_frame"], "end_frame": rec["end_frame"],
                            "cls_original": rec["cls"], "nueva_cls": rec["nueva_cls"],
                            "segundos": rec["segundos"]})
            Handler.done += 1
            if Handler.done % 25 == 0:
                print(f"  {Handler.done} anotados -> {Handler.out_path}")
        self._send(200, "application/json", b'{"ok":1}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="manifest_filtrado.csv")
    ap.add_argument("--videos", nargs="+", required=True, help="video_keys a reanotar, ej: V3 V2")
    ap.add_argument("--out", default="./clips/reanotado.csv")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--shuffle", action="store_true",
                    help="mezclar el orden: evita sesgo de secuencia (rachas de una misma clase)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["video_key"].isin(args.videos)].copy()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    hechos = set()
    if out.exists():
        hechos = set(pd.read_csv(out)["clip"])
        print(f"reanudando: {len(hechos)} clips ya anotados")

    df = df[~df["clip"].isin(hechos)]
    if args.shuffle:
        df = df.sample(frac=1, random_state=args.seed)

    Handler.queue = df[["clip", "video_key", "start_frame", "end_frame", "cls"]].to_dict("records")
    Handler.out_path = out

    if not Handler.queue:
        print("no queda nada por anotar")
        return

    print(f"\npendientes: {len(Handler.queue)} clips de {args.videos}")
    print(f"salida: {out}")
    print(f"\n  abri  http://localhost:{args.port}")
    print(f"  si estas por SSH, desde tu maquina local primero:")
    print(f"    ssh -L {args.port}:localhost:{args.port} lucasb@desarrollo-lucas\n")
    print("Ctrl+C para frenar. El progreso queda guardado.\n")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), Handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print(f"\nfrenado. {Handler.done} anotados en esta sesion.")


if __name__ == "__main__":
    main()