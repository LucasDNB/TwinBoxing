"""
B27 de las versiones: posicion en la cola (versiones-26-10, 04 B27). La GPU es una para
todas las cuentas: la posicion las cuenta a todas y no dice de quien es ningun trabajo.
"""

from __future__ import annotations

from versiones_util import H, crear_cuenta, crear_miembro, subir


def test_la_posicion_cuenta_las_dos_cuentas_y_no_dice_nada_de_la_otra(cliente, entorno):
    c1 = crear_cuenta(entorno, "gimnasio", "Norte")
    c2 = crear_cuenta(entorno, "entrenador", "Sur")
    _, t1 = crear_miembro(entorno, "ana@norte.ar", c1)
    _, t2 = crear_miembro(entorno, "beto@sur.ar", c2)
    subir(cliente, t2, duracion_declarada_s="600")       # 10 min de la otra cuenta
    subir(cliente, t1, duracion_declarada_s="1200")      # 20 min propios
    subir(cliente, t2)                                   # sin duracion
    ultima = subir(cliente, t1, duracion_declarada_s="60").json()["job_id"]
    # La primera ya la tomo el worker.
    entorno["cola"].reclamar(entorno["db"].hacer_sesion(), "w1")

    j = cliente.get(f"/jobs/{ultima}", headers=H(t1)).json()
    f = entorno["cfg"]
    assert j["cola"] == {"delante": 3,
                         "espera_min": [round(1800 * f.factor_min / 60), round(1800 * f.factor_max / 60)],
                         "sin_duracion": 1}
    texto = str(j["cola"])
    assert "beto" not in texto and "Sur" not in texto


def test_sin_cola_no_hay_bloque(cliente, entorno):
    c = crear_cuenta(entorno, "entrenador", "Sur")
    _, t = crear_miembro(entorno, "beto@sur.ar", c)
    sid = subir(cliente, t).json()["job_id"]
    from versiones_util import poner

    poner(entorno, sid, estado="listo")
    assert "cola" not in cliente.get(f"/jobs/{sid}", headers=H(t)).json()
