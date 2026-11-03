"""Un motor falso para los tests del worker: escribe lo que dejaria boxtwin-annotator."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def fightcard(golpes_a, golpes_b=(), duracion=240.0, cobertura=0.9, sin_asignar=0.05):
    def g(lado, t, brazo, i, **extra):
        return {"id": f"{lado}-{brazo}-{i}", "t_inicio": t, "t_fin": t + 0.25, "brazo": brazo,
                "score": 0.95, "tipo": "jab", "descartado": False, "objetivo": "cabeza", **extra}
    return {
        "version": "0.3",
        "video": {"duracion_s": duracion, "fps": 30.0},
        "identidad": {"cobertura_A": cobertura, "cobertura_B": 0.8 if golpes_b else 0.0, "sin_asignar": sin_asignar},
        "peleadores": {
            "A": {"golpes": [g("A", t, b, i) for i, (t, b) in enumerate(golpes_a)]},
            "B": {"golpes": [g("B", t, b, i) for i, (t, b) in enumerate(golpes_b)]},
        },
    }


class MotorFalso:
    """Llamable como _correr_motor(cmd, timeout). Registra cada linea de comando."""

    def __init__(self, base_cmd, golpes_a=(), golpes_b=(), duracion=240.0, falla=None):
        self.n = len(base_cmd)
        self.golpes_a, self.golpes_b, self.duracion = list(golpes_a), list(golpes_b), duracion
        self.falla = falla
        self.llamadas: list[list[str]] = []

    def __call__(self, cmd, timeout=None):
        self.llamadas.append(cmd)
        etapa = cmd[self.n]
        if self.falla == etapa:
            return subprocess.CompletedProcess(cmd, 1, "", "CUDA out of memory")
        if etapa == "procesar":
            out = Path(cmd[cmd.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "candidatos").mkdir(exist_ok=True)
            (out / "candidatos" / "track_3.jpg").write_bytes(b"\xff\xd8jpg")
            (out / "candidatos" / "track_7.jpg").write_bytes(b"\xff\xd8jpg")
            (out / "sesion.json").write_text(json.dumps({
                "estado": "espera_siembra",
                "video": {"duracion_s": self.duracion, "fps": 30.0, "ancho": 1280, "alto": 720},
                "candidatos": [
                    {"track": 3, "sugerido": True, "t_desde": 0, "t_hasta": 200, "alto_relativo": 1.0,
                     "color_guante": {"nombre": "rojos"}},
                    {"track": 7, "sugerido": True, "t_desde": 10, "t_hasta": 150, "alto_relativo": 0.9,
                     "color_guante": {"nombre": "azules"}},
                ],
            }))
        elif etapa == "completar":
            out = Path(cmd[self.n + 1])
            d = json.loads((out / "sesion.json").read_text())
            d["estado"] = "listo"
            (out / "sesion.json").write_text(json.dumps(d))
            (out / "fightcard.json").write_text(json.dumps(
                fightcard(self.golpes_a, self.golpes_b if "--semilla-b" in cmd else (), self.duracion)))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")
