"""
BoxTwin - Cuanto encontro el producto, contra la anotacion manual del mismo video.

POR QUE EXISTE
  El detector esta medido, la identidad esta medida, y el producto no. Entre las piezas y
  lo que el usuario ve hay una etapa que ninguna de las dos mediciones incluye: la
  identidad automatica. Un golpe que ocurre en cuadros donde el sistema no sabe quien es
  quien no lo puede encontrar ningun detector, por bueno que sea, y eso no aparece en el
  recall del detector porque alla la identidad venia resuelta a mano.

  Sin separar esas dos cosas, un recall bajo se le atribuye al detector y se trabaja sobre
  el modelo equivocado. Medido sobre las cinco sesiones de gimnasio, el 18% de los golpes
  anotados cae en cuadros sin identidad resuelta, y en la peor fuente ese numero es 47%.

QUE HACE
  Empareja los golpes de una Fight-Card contra los de un annot.json con el MISMO
  emparejador que usa el resto del proyecto -IoU 0,3- y parte lo que falta en dos:

    SIN IDENTIDAD   el peleador no estaba resuelto en esos cuadros: el detector no lo
                    pudo ver. Es error de la identidad.
    VISIBLE Y NO ENCONTRADO   estaba resuelto y el detector no lo marco. Es error del
                    detector.

  Prueba las dos orientaciones de A y B, porque los roles de la anotacion salen de quien
  los asigno a mano y los de la sesion de la semilla que eligio el usuario: no tienen por
  que coincidir, y compararlos cruzados daria cero.

USO
  from boxtwin.mvp.medicion import medir_sesion
  r = medir_sesion(fightcard, annot, valid)
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Resultado", "carriles_de_annot", "carriles_de_fightcard", "medir_sesion"]

LADO = {"left": "izq", "right": "der"}

# Fraccion de la ventana del golpe que tiene que tener identidad resuelta para considerar
# que el detector "lo pudo ver". La mitad: con menos que eso el carril esta casi todo
# enmascarado y el modelo no tiene sobre que disparar.
FRACCION_VISIBLE = 0.5


@dataclass
class Resultado:
    anotados: int = 0
    marcas: int = 0
    emparejados: int = 0
    sin_identidad: int = 0
    visibles: int = 0
    encontrados_visibles: int = 0
    invertido: bool = False
    por_carril: dict = field(default_factory=dict)

    @property
    def recall(self) -> float:
        return self.emparejados / self.anotados if self.anotados else 0.0

    @property
    def precision(self) -> float:
        return self.emparejados / self.marcas if self.marcas else 0.0

    @property
    def f1(self) -> float:
        r, p = self.recall, self.precision
        return 2 * r * p / (r + p) if r + p else 0.0

    @property
    def recall_sobre_visible(self) -> float:
        """El recall del DETECTOR, descontando lo que no pudo ver."""
        return self.encontrados_visibles / self.visibles if self.visibles else 0.0

    def a_dict(self) -> dict:
        return {
            "anotados": self.anotados,
            "marcas": self.marcas,
            "emparejados": self.emparejados,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
            "sin_identidad": self.sin_identidad,
            "fraccion_sin_identidad": (
                round(self.sin_identidad / self.anotados, 4) if self.anotados else None
            ),
            "visibles": self.visibles,
            "recall_sobre_visible": round(self.recall_sobre_visible, 4),
            "roles_invertidos": self.invertido,
        }


def _seg(inicio: int, fin: int):
    from boxtwin_detector.decodificacion import Segmento

    return Segmento(int(inicio), int(fin), 0)


def carriles_de_annot(doc: dict, invertir: bool = False) -> dict[str, list]:
    """Los golpes anotados, por carril `A-izq` y compania."""
    out: dict[str, list] = {}
    for e in doc.get("events", []):
        f = e["fighter"][-1]
        if invertir:
            f = "B" if f == "A" else "A"
        lado = LADO.get(e["side"], e["side"])
        out.setdefault(f"{f}-{lado}", []).append(
            _seg(e["start_frame"], e["end_frame"])
        )
    for v in out.values():
        v.sort(key=lambda s: s.inicio)
    return out


def carriles_de_fightcard(fc: dict) -> dict[str, list]:
    """Lo que marco el sistema, en los mismos carriles."""
    out: dict[str, list] = {}
    for p in ("A", "B"):
        for g in fc["peleadores"][p]["golpes"]:
            out.setdefault(f"{p}-{g['brazo']}", []).append(
                _seg(g["cuadro_inicio"], g["cuadro_fin"])
            )
    for v in out.values():
        v.sort(key=lambda s: s.inicio)
    return out


def _medir(gt: dict, pred: dict, valid) -> Resultado:
    from boxtwin.core.agreement import emparejar

    r = Resultado()
    r.anotados = sum(len(v) for v in gt.values())
    r.marcas = sum(len(v) for v in pred.values())
    for carril in sorted(set(gt) | set(pred)):
        g, p = gt.get(carril, []), pred.get(carril, [])
        parejas, _, _ = emparejar(list(g), list(p))
        r.emparejados += len(parejas)
        r.por_carril[carril] = {"anotados": len(g), "marcas": len(p),
                                "emparejados": len(parejas)}
        if valid is None:
            continue
        casados = {i for i, _, _ in parejas}
        idx = 0 if carril[0] == "A" else 1
        for k, s in enumerate(g):
            tramo = valid[idx, s.inicio : s.fin + 1]
            if len(tramo) and float(tramo.mean()) >= FRACCION_VISIBLE:
                r.visibles += 1
                r.encontrados_visibles += k in casados
            else:
                r.sin_identidad += 1
    return r


def medir_sesion(fightcard: dict, annot: dict, valid=None) -> Resultado:
    """
    La Fight-Card contra la anotacion. `valid` es la mascara (2, T) de `poses.npz`.

    Sin `valid` se mide igual, pero no se puede separar lo que el detector no vio de lo que
    no marco, que es justo la parte que dice sobre cual de las dos piezas hay que trabajar.
    """
    pred = carriles_de_fightcard(fightcard)
    opciones = [_medir(carriles_de_annot(annot, inv), pred, valid) for inv in (False, True)]
    mejor = max(opciones, key=lambda x: x.emparejados)
    mejor.invertido = mejor is opciones[1]
    return mejor
