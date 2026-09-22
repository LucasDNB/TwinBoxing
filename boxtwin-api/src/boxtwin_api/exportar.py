"""
BoxTwin API - Exportar la Fight-Card. F7.

POR QUE EL PDF SE HACE IMPRIMIENDO
  Generar PDF pide reportlab o un navegador headless: una dependencia mas en la imagen, o
  Chromium adentro del contenedor del worker. Para un documento de una carilla que el
  entrenador va a mandar por WhatsApp, la impresion del navegador da el mismo resultado y
  cuesta cero. Se entrega una pagina con estilos de impresion y se dice en la pagina como
  guardarla.

QUE HACE
  CSV de los golpes -una fila por evento, que es lo que se puede abrir en una planilla- y
  HTML listo para imprimir.

  Las dos salidas llevan el recall y la precision arriba de todo. Es el mismo criterio que
  en la pantalla (RF6): un conteo sin su margen, impreso y fuera de contexto, es peor que
  en la pantalla, porque circula solo.
"""

from __future__ import annotations

import csv
import html
import io

__all__ = ["a_csv", "a_html"]


def a_csv(fc: dict) -> str:
    salida = io.StringIO()
    w = csv.writer(salida)
    w.writerow([
        "peleador", "brazo", "t_inicio_s", "t_fin_s", "cuadro_inicio", "cuadro_fin",
        "score_deteccion", "tipo_estimado", "confianza_tipo", "corregido_a", "round",
    ])
    rondas = fc.get("video", {}).get("rounds", [])
    for p, datos in sorted(fc["peleadores"].items()):
        for g in datos["golpes"]:
            r = next(
                (v["round"] for v in rondas
                 if v["inicio_s"] <= g["t_inicio"] < v["fin_s"]), ""
            )
            w.writerow([
                p, g["brazo"], g["t_inicio"], g["t_fin"], g["cuadro_inicio"],
                g["cuadro_fin"], g["score"], g.get("tipo") or "",
                g.get("confianza_tipo") if g.get("confianza_tipo") is not None else "",
                (g.get("corregido") or {}).get("tipo", ""), r,
            ])
    return salida.getvalue()


def a_html(fc: dict) -> str:
    """Una carilla imprimible. Sin javascript ni recursos externos: se guarda y se manda."""
    v = fc.get("video", {})
    det = fc.get("detector", {})
    ident = fc.get("identidad", {})
    e = html.escape

    filas_round = []
    for p in ("A", "B"):
        for r in fc["peleadores"][p].get("por_round", []):
            filas_round.append(
                f"<tr><td>{p}</td><td>{r['round']}</td><td>{r['total']}</td>"
                f"<td>{r['izq']}</td><td>{r['der']}</td><td>{r['por_minuto']}</td></tr>"
            )

    totales = []
    for p in ("A", "B"):
        t = fc["peleadores"][p]["total"]
        totales.append(
            f"<tr><td>Peleador {p}</td><td class='n'>{t['total']}</td>"
            f"<td class='n'>{t['izq']}</td><td class='n'>{t['der']}</td></tr>"
        )

    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Fight-Card · {e(str(v.get('nombre', '')))}</title>
<style>
  body {{ font: 14px/1.5 system-ui, sans-serif; max-width: 760px; margin: 2rem auto;
         padding: 0 1rem; color: #111; }}
  h1 {{ font-size: 1.4rem; margin-bottom: .2rem; }}
  .sub {{ color: #555; margin-top: 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border-bottom: 1px solid #ddd; padding: .4rem .6rem; text-align: left; }}
  td.n, th.n {{ text-align: right; }}
  .margen {{ background: #fff8e1; border-left: 3px solid #f0b429; padding: .7rem 1rem;
             margin: 1rem 0; }}
  .no {{ color: #555; font-size: .9rem; }}
  @media print {{ body {{ margin: 0; }} .noprint {{ display: none; }} }}
</style></head><body>
<h1>Fight-Card</h1>
<p class="sub">{e(str(v.get('nombre', '')))} · {v.get('duracion_s', 0):.0f} s ·
   {v.get('fps', 0):.2f} fps</p>

<div class="margen">
  <strong>Golpes detectados, no golpes lanzados.</strong>
  El detector tiene precision medida {det.get('precision_medida')} y recall
  {det.get('recall_medido')} sobre una fuente que no vio. O sea que este conteo esta
  <strong>por debajo</strong> del real. Lo que la medicion sostiene es la comparacion
  adentro de la sesion: un round contra otro, un peleador contra el otro.
</div>

<h2>Volumen</h2>
<table><tr><th></th><th class="n">total</th><th class="n">izq</th><th class="n">der</th></tr>
{''.join(totales)}</table>

{'<h2>Por round</h2><table><tr><th>peleador</th><th>round</th><th class="n">total</th>'
 '<th class="n">izq</th><th class="n">der</th><th class="n">por minuto</th></tr>'
 + ''.join(filas_round) + '</table>' if filas_round else ''}

<h2>Identidad</h2>
<p>Cobertura A {ident.get('cobertura_A', 0):.1%}, B {ident.get('cobertura_B', 0):.1%}.
   Sin asignar {ident.get('sin_asignar', 0):.1%} del tiempo.</p>

<h2>Tipo de golpe</h2>
<p>El tipo es una <strong>estimacion</strong> del clasificador
   ({e(str(fc.get('clasificador', {}).get('checkpoint') or 'sin correr'))}), que no
   generaliza a fuentes nuevas. Cada golpe lleva su confianza en el CSV.</p>

<h2 class="no">Lo que esta Fight-Card no dice</h2>
<ul class="no">{''.join(f'<li>{e(x)}</li>' for x in fc.get('no_incluye', []))}</ul>

<p class="noprint"><em>Para PDF: imprimir esta pagina y elegir "Guardar como PDF".</em></p>
</body></html>
"""
