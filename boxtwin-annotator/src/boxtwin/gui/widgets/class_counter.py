"""
BoxTwin - Contador de eventos por clase.

POR QUE EXISTE
  El desbalance de clases se descubre tarde o se descubre a tiempo, y la diferencia son
  horas de anotacion. En BoxingVI la clase mas frecuente tiene 1371 ejemplos y la menos
  frecuente 296: casi cinco a uno, y eso se supo al terminar de cortar los clips.
  Verlo mientras se anota permite ir a buscar material de las clases flacas en vez de
  acumular mas de las que ya sobran.

  Se muestra en los dos espacios. El de lado es el del export por defecto y es el que se
  resalta; el de mano adelantada y atrasada se muestra igual porque con dos peleadores de
  guardias distintas los dos espacios no coinciden, y conviene ver si uno esta balanceado y
  el otro no.

QUE HACE
  Cuenta los eventos por tipo, lado y rol de mano, y marca en color la clase mas escasa.

USO
  contador.refrescar(doc)
"""

from __future__ import annotations

from collections import Counter

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from boxtwin.core.schema import AnnotationDoc
from boxtwin.core.types import ArmRole, Completeness, PunchType, Side

__all__ = ["ClassCounter"]


class ClassCounter(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.lbl = QLabel("sin eventos")
        self.lbl.setTextFormat(Qt.TextFormat.RichText)
        self.lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(0, 0, 0, 0)
        raiz.addWidget(self.lbl)

    def refrescar(self, doc: AnnotationDoc) -> None:
        # Solo los golpes completos: los amagues y abortados no entran a los espacios de 6
        # y 12 clases, asi que contarlos aca daria una idea equivocada del balance.
        completos = [e for e in doc.events if e.completeness is Completeness.FULL]
        if not doc.events:
            self.lbl.setText("sin eventos")
            return

        por_lado = Counter((e.punch_type, e.side) for e in completos)
        por_rol = Counter((e.punch_type, e.arm_role) for e in completos)
        otros = Counter(e.completeness for e in doc.events if e.completeness is not Completeness.FULL)

        minimo = min(por_lado.values(), default=0)

        filas = ["<b>por lado</b> <small>(espacio del export)</small><table cellspacing='2'>"]
        for t in PunchType:
            celdas = ""
            for s in Side:
                n = por_lado.get((t, s), 0)
                # La clase mas escasa se resalta: es donde conviene buscar material.
                color = " style='color:#e08a3c'" if n == minimo and len(completos) else ""
                celdas += f"<td align='right'{color}>{n}</td>"
            filas.append(f"<tr><td>{t.value}</td>{celdas}</tr>")
        filas.append("</table>")
        filas.append("<small>izquierda · derecha</small><br><br>")

        filas.append("<b>por mano (lead/rear)</b><table cellspacing='2'>")
        for t in PunchType:
            celdas = "".join(
                f"<td align='right'>{por_rol.get((t, r), 0)}</td>" for r in ArmRole
            )
            filas.append(f"<tr><td>{t.value}</td>{celdas}</tr>")
        filas.append("</table>")
        filas.append("<small>adelantada · atrasada</small>")

        if otros:
            resumen = " · ".join(f"{c.value} {n}" for c, n in sorted(otros.items(), key=lambda x: x[0].value))
            filas.append(f"<br><br><small>fuera del espacio de clases: {resumen}</small>")

        filas.append(f"<br><br><b>{len(completos)}</b> completos de {len(doc.events)}")
        self.lbl.setText("".join(filas))
