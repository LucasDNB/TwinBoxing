"""
La cola y el worker.

Lo que se fija aca es lo que decide si la cola sirve: que un trabajo lo tome uno solo, que
el trabajo de un worker muerto vuelva, que un fallo reintente pocas veces y despues se
declare, y que el estado de la sesion salga del archivo que escribio el comando y no del
codigo de salida del proceso.
"""

from __future__ import annotations

import json

import pytest
from conftest import subir_video


@pytest.fixture
def con_sesion(entorno, cliente, registrado):
    sid = subir_video(cliente).json()["job_id"]
    return sid


# -- exclusion --------------------------------------------------------------


def test_un_trabajo_lo_toma_un_solo_worker(entorno, con_sesion):
    cola, db_mod = entorno["cola"], entorno["db"]
    with db_mod.hacer_sesion() as db1, db_mod.hacer_sesion() as db2:
        t1 = cola.reclamar(db1, "worker-1")
        t2 = cola.reclamar(db2, "worker-2")
    assert t1 is not None
    assert t2 is None, "el segundo worker no puede llevarse el mismo trabajo"
    assert t1.tomado_por == "worker-1"


def test_la_cola_vacia_devuelve_none(entorno):
    with entorno["db"].hacer_sesion() as db:
        assert entorno["cola"].reclamar(db, "w") is None


def test_se_atiende_en_orden_de_llegada(entorno, cliente, registrado):
    a = subir_video(cliente, nombre="a.mp4").json()["job_id"]
    b = subir_video(cliente, nombre="b.mp4").json()["job_id"]
    cola, db_mod = entorno["cola"], entorno["db"]
    with db_mod.hacer_sesion() as db:
        assert cola.reclamar(db, "w").sesion_id == a
        assert cola.reclamar(db, "w").sesion_id == b


def test_una_etapa_desconocida_no_se_encola(entorno, con_sesion):
    with entorno["db"].hacer_sesion() as db:
        with pytest.raises(ValueError, match="etapa desconocida"):
            entorno["cola"].encolar(db, con_sesion, "magia")


# -- worker muerto ----------------------------------------------------------


def test_el_trabajo_de_un_worker_muerto_vuelve_a_la_cola(entorno, con_sesion, monkeypatch):
    # Sin esto, un kill -9 en medio del preproceso deja la sesion colgada para siempre y
    # nadie se entera hasta que el usuario pregunta.
    cola, db_mod = entorno["cola"], entorno["db"]
    with db_mod.hacer_sesion() as db:
        t = cola.reclamar(db, "worker-que-se-murio")
        assert t is not None

    monkeypatch.setattr(entorno["cfg"], "minutos_para_reclamar", 0.0)
    with db_mod.hacer_sesion() as db:
        assert cola.liberar_abandonados(db) == 1
        otro = cola.reclamar(db, "worker-vivo")
    assert otro is not None and otro.tomado_por == "worker-vivo"


def test_un_trabajo_reciente_no_se_le_saca_a_nadie(entorno, con_sesion):
    cola, db_mod = entorno["cola"], entorno["db"]
    with db_mod.hacer_sesion() as db:
        cola.reclamar(db, "worker-1")
        assert cola.liberar_abandonados(db) == 0


# -- fallo y reintento ------------------------------------------------------


def test_un_fallo_reintenta_una_vez_y_despues_se_declara(entorno, con_sesion):
    from boxtwin_api.modelos import Sesion

    cola, db_mod = entorno["cola"], entorno["db"]
    with db_mod.hacer_sesion() as db:
        t = cola.reclamar(db, "w")
        nuevo = cola.fallar(db, t, "la GPU estaba ocupada")
        assert nuevo is not None and nuevo.intento == 2
        assert db.get(Sesion, con_sesion).estado == "en_cola"

        t2 = cola.reclamar(db, "w")
        assert cola.fallar(db, t2, "otra vez") is None, "no hay tercer intento"
        s = db.get(Sesion, con_sesion)
        assert s.estado == "fallo"
        assert "otra vez" in s.error


def test_terminar_deja_la_sesion_en_el_estado_que_se_le_pasa(entorno, con_sesion):
    from boxtwin_api.modelos import Sesion

    cola, db_mod = entorno["cola"], entorno["db"]
    with db_mod.hacer_sesion() as db:
        t = cola.reclamar(db, "w")
        cola.terminar(db, t, estado_sesion="espera_siembra")
        assert db.get(Sesion, con_sesion).estado == "espera_siembra"
        assert t.estado == "listo"


# -- comandos ---------------------------------------------------------------


def test_el_comando_de_procesar_lleva_el_video_y_la_sesion(entorno, con_sesion):
    from boxtwin_api.modelos import Sesion

    worker, cola, db_mod = entorno["worker"], entorno["cola"], entorno["db"]
    with db_mod.hacer_sesion() as db:
        t = cola.reclamar(db, "w")
        s = db.get(Sesion, con_sesion)
        cmd = worker.comando_de(t, s)[0]
    assert cmd[1] == "procesar"
    assert cmd[2].endswith("spar.mp4")
    assert "--modelo-guantes" in cmd


def test_el_comando_de_completar_lleva_las_dos_semillas(entorno, con_sesion):
    from boxtwin_api.modelos import Sesion

    worker, cola, db_mod = entorno["worker"], entorno["cola"], entorno["db"]
    with db_mod.hacer_sesion() as db:
        cola.reclamar(db, "w")
        t = cola.encolar(db, con_sesion, "completar", track_a=3, track_b=7)
        db.commit()
        cmd = worker.comando_de(t, db.get(Sesion, con_sesion))[0]
    assert cmd[cmd.index("--semilla-a") + 1] == "3"
    assert cmd[cmd.index("--semilla-b") + 1] == "7"


def test_sin_clasificador_configurado_la_etapa_lo_dice(entorno, con_sesion):
    # El plan B declarado en el spec: si la imagen con los dos entornos no sale, la
    # clasificacion corre afuera. Mientras tanto, el resto de la Fight-Card no depende.
    from boxtwin_api.modelos import Sesion

    worker, cola, db_mod = entorno["worker"], entorno["cola"], entorno["db"]
    with db_mod.hacer_sesion() as db:
        t = cola.encolar(db, con_sesion, "clasificar")
        db.commit()
        with pytest.raises(RuntimeError, match="BOXTWIN_CMD_CLASIFICADOR"):
            worker.comando_de(t, db.get(Sesion, con_sesion))


# -- el bucle ---------------------------------------------------------------


def test_una_vuelta_corre_el_trabajo_y_lee_el_estado_del_archivo(
    entorno, con_sesion, monkeypatch
):
    # El estado NO sale del codigo de salida: un proceso que termina en 0 y dejo la sesion
    # a medias es un caso real, y creerle al codigo de salida lo taparia.
    from boxtwin_api.modelos import Sesion

    worker, db_mod, cfg = entorno["worker"], entorno["db"], entorno["cfg"]
    d = cfg.dir_sesion(con_sesion)
    d.mkdir(parents=True, exist_ok=True)

    def falso_run(cmd, **kw):
        (d / "sesion.json").write_text(json.dumps({"estado": "espera_siembra"}))

        class R:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return R()

    monkeypatch.setattr(worker.subprocess, "run", falso_run)
    assert worker.correr(una_vuelta=True, worker="w") == 1
    with db_mod.hacer_sesion() as db:
        assert db.get(Sesion, con_sesion).estado == "espera_siembra"


def test_un_comando_que_falla_no_tumba_al_worker(entorno, con_sesion, monkeypatch):
    from boxtwin_api.modelos import Trabajo
    from sqlalchemy import select

    worker, db_mod = entorno["worker"], entorno["db"]

    def falso_run(cmd, **kw):
        class R:
            returncode = 1
            stdout = ""
            stderr = "CUDA out of memory"

        return R()

    monkeypatch.setattr(worker.subprocess, "run", falso_run)
    assert worker.correr(una_vuelta=True, worker="w") == 1
    with db_mod.hacer_sesion() as db:
        fallidos = db.scalars(select(Trabajo).where(Trabajo.estado == "fallo")).all()
    assert len(fallidos) == 1
    assert "CUDA out of memory" in fallidos[0].error


def test_el_bucle_sin_trabajos_no_hace_nada(entorno):
    assert entorno["worker"].correr(una_vuelta=True, worker="w") == 0
