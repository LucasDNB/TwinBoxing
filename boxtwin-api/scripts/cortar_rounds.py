"""
BoxTwin - Corta los rounds netos de una transmision (gemelo-tercero, 2-11).

POR QUE EXISTE
  A arma las ventanas de round contando desde el segundo 0 con round y descanso fijos
  (boxtwin.mvp.sesion.ventanas_de_round). Una transmision no cumple eso: arranca con la
  presentacion, los descansos duran lo que decida el canal y en ellos pasan repeticiones en
  camara lenta y desde otro angulo. El detector ve golpes en las repeticiones y A los suma al
  round que acaba de terminar, asi que el volumen saldria inflado con golpes repetidos. Con un
  solo minuto de desfase al principio, ademas, todos los rounds quedan corridos.

  La salida no es detectar el gong (componente nuevo, sin validar) sino que una persona marque
  donde empieza cada round, una vez, y cortar: queda un video de rounds netos que A procesa con
  descanso 0, como el material anotado de Pacquiao.

QUE HACE
  Lee los inicios de round (mm:ss o hh:mm:ss, uno por round), valida que esten en orden y no se
  pisen, corta cada round de --round-s segundos (el ultimo puede ser mas corto con --fin-ultimo,
  para un nocaut) reencodeando para que el corte caiga en el cuadro exacto y no en el keyframe,
  y los une en un solo archivo. Escribe al lado un JSON con los inicios, la duracion de cada
  round y el sha256 del original. No toca el original. Reanudable: un round ya cortado no se
  vuelve a cortar.

USO
  python boxtwin-api/scripts/cortar_rounds.py ~/Proyectos/twinboxing-finales/pacman3x1.mp4 \
      --inicios 0:41 4:43 8:45 ... -o ~/Proyectos/twinboxing-finales/pacman-netos.mp4
  (--ensayo imprime los comandos de ffmpeg y no corre nada)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Partes puras (con pruebas)
# ---------------------------------------------------------------------------

def a_segundos(texto: str) -> float:
    """'4:43' -> 283.0, '1:02:05.5' -> 3725.5, '41' -> 41.0."""
    partes = texto.strip().split(":")
    if not 1 <= len(partes) <= 3 or any(p == "" for p in partes):
        raise ValueError(f"tiempo mal escrito: {texto!r}")
    total = 0.0
    for p in partes:
        total = total * 60 + float(p)
    if total < 0:
        raise ValueError(f"tiempo negativo: {texto!r}")
    return total


def planear(inicios: list[float], round_s: float, duracion_s: float,
            fin_ultimo: float | None = None) -> list[dict]:
    """
    Los cortes, uno por round. Falla si los inicios no estan en orden, si un round pisa al
    siguiente o si se pasa del video.
    """
    if not inicios:
        raise ValueError("hace falta al menos un inicio")
    if round_s <= 0:
        raise ValueError("round_s tiene que ser mayor que cero")
    cortes = []
    for i, ini in enumerate(inicios):
        ultimo = i == len(inicios) - 1
        fin = fin_ultimo if (ultimo and fin_ultimo is not None) else ini + round_s
        if fin <= ini:
            raise ValueError(f"round {i + 1}: el fin no es posterior al inicio")
        # El ultimo se recorta a lo que dure el video, como ventanas_de_round; uno del medio
        # que se pasa es un inicio mal escrito.
        if fin > duracion_s + 0.5 and (not ultimo or ini >= duracion_s):
            raise ValueError(f"round {i + 1}: termina en {fin:.1f} s y el video dura "
                             f"{duracion_s:.1f} s")
        if not ultimo and fin > inicios[i + 1]:
            raise ValueError(f"round {i + 1} termina en {fin:.1f} s, despues del inicio del "
                             f"round {i + 2} ({inicios[i + 1]:.1f} s)")
        cortes.append({"round": i + 1, "inicio_s": round(ini, 3),
                       "dur_s": round(min(fin, duracion_s) - ini, 3)})
    return cortes


def comando_corte(video: Path, corte: dict, destino: Path, crf: int = 18) -> list[str]:
    """ffmpeg con -ss antes de -i (rapido) y reencode (corte en el cuadro exacto)."""
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{corte['inicio_s']:.3f}", "-i", str(video), "-t", f"{corte['dur_s']:.3f}",
            "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            str(destino)]


def comando_union(lista: Path, salida: Path) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
            "-i", str(lista), "-c", "copy", "-movflags", "+faststart", str(salida)]


# ---------------------------------------------------------------------------
# Lo que toca disco
# ---------------------------------------------------------------------------

def duracion(video: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(video)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        while bloque := f.read(1 << 20):
            h.update(bloque)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("video", type=Path)
    p.add_argument("--inicios", nargs="+", required=True, help="inicio de cada round")
    p.add_argument("--round-s", type=float, default=180.0)
    p.add_argument("--fin-ultimo", default=None, help="fin del ultimo round, si no fue completo")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("-o", "--salida", type=Path, required=True)
    p.add_argument("--ensayo", action="store_true")
    a = p.parse_args(argv)

    video = a.video.expanduser()
    if a.salida.expanduser().resolve() == video.resolve():
        print("error: la salida no puede ser el original", file=sys.stderr)
        return 2
    inicios = [a_segundos(t) for t in a.inicios]
    fin_ultimo = a_segundos(a.fin_ultimo) if a.fin_ultimo else None
    cortes = planear(inicios, a.round_s, duracion(video), fin_ultimo)

    salida = a.salida.expanduser()
    tmp = salida.with_suffix(".rounds")
    tmp.mkdir(parents=True, exist_ok=True)
    lista = tmp / "lista.txt"
    partes = []
    for c in cortes:
        destino = tmp / f"round{c['round']:02d}.mp4"
        partes.append(destino)
        cmd = comando_corte(video, c, destino, a.crf)
        if a.ensayo:
            print(" ".join(cmd))
        elif not destino.exists():
            subprocess.run(cmd, check=True)
            print(f"round {c['round']}: listo", file=sys.stderr)
    lista.write_text("".join(f"file '{x.resolve()}'\n" for x in partes))
    union = comando_union(lista, salida)
    if a.ensayo:
        print(" ".join(union))
        print(json.dumps(cortes, indent=2))
        return 0
    subprocess.run(union, check=True)
    meta = {"original": str(video), "sha256_original": sha256(video), "round_s": a.round_s,
            "cortes": cortes, "duracion_salida_s": round(duracion(salida), 3)}
    salida.with_suffix(".cortes.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
