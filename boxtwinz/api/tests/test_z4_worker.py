"""Z4: cola, lock de GPU, invocacion del motor copiada de A, adaptador y tiempo de GPU."""

import fcntl
import json
import threading
import time

import pytest
from sqlalchemy import select

from boxtwinz_api import worker
from boxtwinz_api.cola import encolar, reclamar
from boxtwinz_api.config import cfg
from boxtwinz_api.models import Golpe, NotificacionAdmin, Perfil, Sesion, Trabajo, Usuario, UsoGpuDiario
from boxtwinz_api.services.engine_adapter import candidato_individual, leer_salida
from boxtwinz_api.services.quota import cupo
from conftest import registrar
from motor_falso import MotorFalso, fightcard


def _sesion(db, cliente, modalidad="sombra", segundos=240.0):
    registrar(cliente)
    u = db.scalar(select(Usuario))
    p = db.scalar(select(Perfil).where(Perfil.ranura == 1))
    s = Sesion(usuario_id=u.id, perfil_id=p.id, modalidad=modalidad, video_ruta="/v/x.mp4",
               estado="subida", video_duracion_s=segundos, segundos_cupo=segundos)
    db.add(s)
    db.flush()
    encolar(db, s.id, "procesar")
    db.commit()
    return s


def _vuelta(db, motor):
    t = reclamar(db, "test")
    try:
        estado = worker.ejecutar_trabajo(db, t, correr_motor=motor)
    except Exception as e:  # noqa: BLE001
        db.rollback()
        from boxtwinz_api.cola import fallar
        fallar(db, t, str(e))
        return "fallo"
    from boxtwinz_api.cola import terminar
    terminar(db, t, estado_sesion=estado)
    return estado


def test_comando_igual_que_a(db, cliente, monkeypatch):
    monkeypatch.setattr(cfg, "cmd_boxtwin", ["/env/bin/boxtwin-annotator"])
    s = _sesion(db, cliente)
    t = db.scalar(select(Trabajo))
    cmd = worker.comando_de(t, s)
    assert cmd[:3] == ["/env/bin/boxtwin-annotator", "procesar", "/v/x.mp4"]
    for bandera in ("--out", "--modelo-guantes", "--modelo-pose", "--device"):
        assert bandera in cmd
    assert "clasificar" not in cmd and "--round" not in cmd
    t2 = Trabajo(sesion_id=s.id, etapa="completar", parametros={"track_a": 3, "track_b": 7})
    c2 = worker.comando_de(t2, s)
    assert c2[1:3] == ["completar", str(cfg.dir_trabajo(s.id))]
    assert c2[c2.index("--semilla-a") + 1] == "3" and c2[c2.index("--semilla-b") + 1] == "7"
    assert "--apariencia" in c2 and "--cortar-mixtos" not in c2
    t3 = Trabajo(sesion_id=s.id, etapa="completar", parametros={"track_a": 3, "track_b": None})
    assert "--semilla-b" not in worker.comando_de(t3, s)


def test_sombra_de_punta_a_punta_sin_siembra(db, cliente):
    s = _sesion(db, cliente, "sombra")
    motor = MotorFalso(cfg.cmd_boxtwin, golpes_a=[(1.0, "izq"), (1.3, "der"), (70.0, "izq")])
    assert _vuelta(db, motor) == "completando"
    db.refresh(s)
    assert s.semilla_usuario == 3          # el sugerido con mas permanencia
    assert _vuelta(db, motor) == "lista"
    db.refresh(s)
    assert s.estado == "lista" and s.calidad_pose == 0.9 and s.fraccion_sin_identidad == pytest.approx(0.1)
    gs = db.scalars(select(Golpe).where(Golpe.sesion_id == s.id).order_by(Golpe.t_inicio)).all()
    assert [g.mano for g in gs] == ["izquierda", "derecha", "izquierda"]
    assert all(g.quien == "usuario" and g.altura_estimada == "cabeza" for g in gs)
    assert "--semilla-b" not in motor.llamadas[1]


def test_sparring_espera_siembra(db, cliente):
    s = _sesion(db, cliente, "sparring")
    motor = MotorFalso(cfg.cmd_boxtwin)
    assert _vuelta(db, motor) == "esperando_siembra"
    assert db.scalar(select(Trabajo).where(Trabajo.estado == "en_cola")) is None


def test_tiempo_de_gpu_se_registra(db, cliente):
    s = _sesion(db, cliente)

    def lento(cmd, timeout=None):
        time.sleep(0.05)
        return MotorFalso(cfg.cmd_boxtwin)(cmd, timeout)

    _vuelta(db, lento)
    t = db.scalar(select(Trabajo).where(Trabajo.etapa == "procesar"))
    assert t.inicio_gpu and t.fin_gpu and t.fin_gpu > t.inicio_gpu
    db.refresh(s)
    assert s.segundos_gpu >= 0.05
    assert sum(f.segundos_ocupados for f in db.scalars(select(UsoGpuDiario))) == pytest.approx(s.segundos_gpu)


def test_fallo_definitivo_devuelve_cupo_y_avisa(db, cliente, monkeypatch):
    monkeypatch.setattr(cfg, "intentos_maximos", 1)
    s = _sesion(db, cliente)
    from boxtwinz_api.services.quota import descontar
    u = db.get(Usuario, s.usuario_id)
    descontar(db, u, s, 240.0)
    db.commit()
    assert cupo(db, u)["usados_min"] == 4.0
    assert _vuelta(db, MotorFalso(cfg.cmd_boxtwin, falla="procesar")) == "fallo"
    db.refresh(s)
    assert s.estado == "error" and s.error_etapa == "procesar" and "out of memory" in s.error
    assert cupo(db, u)["usados_min"] == 0.0
    assert db.scalar(select(NotificacionAdmin)).tipo == "fallo_trabajo"


def test_fallo_con_reintento(db, cliente):
    s = _sesion(db, cliente)
    _vuelta(db, MotorFalso(cfg.cmd_boxtwin, falla="procesar"))
    assert db.scalar(select(Trabajo).where(Trabajo.estado == "en_cola")).intento == 2
    db.refresh(s)
    assert s.estado != "error"


def test_lock_gpu_excluye(tmp_path):
    ruta = tmp_path / "gpu.lock"
    tomado = threading.Event()
    soltar = threading.Event()

    def tener():
        with worker.lock_gpu(ruta):
            tomado.set()
            soltar.wait(2)

    h = threading.Thread(target=tener)
    h.start()
    tomado.wait(2)
    with open(ruta, "a+") as f:
        with pytest.raises(BlockingIOError):
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    soltar.set()
    h.join()


def test_adaptador_ignora_tipo_y_descartados(tmp_path):
    fc = fightcard([(1.0, "izq"), (2.0, "der")], [(1.5, "der")])
    fc["peleadores"]["A"]["golpes"][1]["descartado"] = True
    (tmp_path / "fightcard.json").write_text(json.dumps(fc))
    s = leer_salida(tmp_path)
    assert [(g.quien, g.mano) for g in s.golpes] == [("rival", "derecha"), ("usuario", "izquierda")]
    assert s.descartados == 1
    assert not hasattr(s.golpes[0], "tipo")


def test_adaptador_rechaza_otro_contrato(tmp_path):
    (tmp_path / "fightcard.json").write_text(json.dumps({"version": "0.2", "peleadores": {}}))
    with pytest.raises(ValueError):
        leer_salida(tmp_path)


def test_adaptador_lee_una_fightcard_real_si_esta():
    """Contra una Fight-Card real de la estacion, si existe. No es un numero de tesis."""
    from pathlib import Path

    reales = sorted(Path("~/boxtwin/datos/sesiones").expanduser().glob("*/fightcard.json"))
    if not reales:
        pytest.skip("sin Fight-Cards reales en esta maquina")
    s = leer_salida(reales[-1].parent)
    assert s.golpes and {g.mano for g in s.golpes} <= {"izquierda", "derecha"}
    assert s.calidad_pose is None or 0 <= s.calidad_pose <= 1


def test_candidato_individual_sin_sugeridos():
    assert candidato_individual({"candidatos": [
        {"track": 1, "t_desde": 0, "t_hasta": 10}, {"track": 2, "t_desde": 0, "t_hasta": 50}]}) == 2
    assert candidato_individual({}) is None


def test_sin_identidad_es_uno_menos_la_peor_cobertura(tmp_path):
    fc = fightcard([(1.0, "izq")], [(1.5, "der")], cobertura=0.95, sin_asignar=0.70)
    fc["identidad"]["cobertura_B"] = 0.80
    (tmp_path / "fightcard.json").write_text(json.dumps(fc))
    # sin_asignar (terceros al fondo) no entra: 1 - min(0,95, 0,80).
    assert leer_salida(tmp_path).fraccion_sin_identidad == pytest.approx(0.20)
