"""T8: guardas de ruta y candado de GPU (01-percepcion.md §4 y 04-convivencia.md, regla 3)."""

import json
import sqlite3

import pytest


def test_proyecto_fuera_del_gemelo_se_niega(gemelo, tmp_path):
    import guardas
    afuera = tmp_path / "TwinBoxing" / "anotacion-spar-01"
    afuera.mkdir(parents=True)
    with pytest.raises(guardas.GuardaError):
        guardas.exigir_bajo_gemelo(afuera)
    adentro = gemelo / "raiz-y26" / "anotacion-spar-01"
    assert guardas.exigir_bajo_gemelo(adentro) == adentro


def test_un_enlace_que_sale_del_gemelo_se_niega(gemelo, tmp_path):
    """Lo que paso el 19-08: el camino parece aislado y el enlace lleva al proyecto real."""
    import guardas
    real = tmp_path / "TwinBoxing" / "anotacion"
    real.mkdir(parents=True)
    (gemelo / "raiz-y26").mkdir()
    (gemelo / "raiz-y26" / "anotacion").symlink_to(real)
    with pytest.raises(guardas.GuardaError):
        guardas.exigir_bajo_gemelo(gemelo / "raiz-y26" / "anotacion" / "cache")


def test_dentro_del_repo_se_niega(gemelo):
    import guardas
    with pytest.raises(guardas.GuardaError):
        guardas.exigir_fuera_del_repo(guardas.REPO / "anotacion" / "annotations" / "x.json")


def test_export_sin_out_se_niega(gemelo):
    import guardas
    with pytest.raises(guardas.GuardaError):
        guardas.exigir_out(None)
    with pytest.raises(guardas.GuardaError):
        guardas.exigir_out(guardas.REPO / "anotacion" / "exports")
    assert guardas.exigir_out(gemelo / "e8" / "y26") == gemelo / "e8" / "y26"


def _annot(ruta, anotador):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps({"identity": {"assignments": [
        {"origin": {"annotator": anotador}}]}}))
    return ruta


def test_annot_que_no_escribio_el_puente_se_niega(gemelo):
    import guardas
    destino = gemelo / "raiz-y26" / "anotacion" / "annotations" / "Sparring.annot.json"
    guardas.exigir_annot_del_puente(destino)                 # no existe: se puede escribir
    _annot(destino, "puente-e8")
    guardas.exigir_annot_del_puente(destino)                 # lo escribio el puente
    _annot(destino, "lucasb")
    with pytest.raises(guardas.GuardaError):
        guardas.exigir_annot_del_puente(destino)


def _base(ruta, estados):
    con = sqlite3.connect(ruta)
    con.execute("create table trabajos (id text, estado text)")
    con.executemany("insert into trabajos values (?, ?)", [(str(i), e) for i, e in enumerate(estados)])
    con.commit()
    con.close()
    return ruta


def test_trabajos_de_a_en_cola_o_tomados(gemelo, tmp_path):
    import guardas
    assert guardas.trabajos_de_a(_base(tmp_path / "a.db", ["listo", "error"])) == 0
    assert guardas.trabajos_de_a(_base(tmp_path / "b.db", ["en_cola", "tomado", "listo"])) == 2


def test_la_base_de_a_se_abre_solo_lectura(gemelo, tmp_path):
    import guardas
    db = _base(tmp_path / "a.db", ["en_cola"])
    antes = db.read_bytes()
    guardas.trabajos_de_a(db)
    assert db.read_bytes() == antes and not (tmp_path / "a.db-journal").exists()


def test_procesos_en_la_gpu_desde_la_salida_de_nvidia_smi(gemelo):
    import guardas
    assert guardas.pids_en_gpu("") == []
    assert guardas.pids_en_gpu("4242\n4343\n") == [4242, 4343]


def test_el_candado_espera_con_un_trabajo_de_a_y_lo_deja_en_el_log(gemelo, tmp_path):
    import guardas
    db = _base(tmp_path / "a.db", ["tomado"])
    log = gemelo / "e8" / "candado.log"
    vueltas = []

    def dormir(s):
        vueltas.append(s)
        con = sqlite3.connect(db)                      # A termina su trabajo
        con.execute("update trabajos set estado='listo'")
        con.commit()
        con.close()

    esperas = guardas.esperar_candado("S7 recache", db, log, gpu=lambda: "", dormir=dormir)
    assert esperas == 1 and len(vueltas) == 1
    lineas = log.read_text().splitlines()
    assert any("espera" in l and "S7 recache" in l and "trabajos_a=1" in l for l in lineas)
    assert any("libre" in l and "S7 recache" in l for l in lineas)


def test_el_candado_espera_con_otro_proceso_en_la_gpu(gemelo, tmp_path):
    import guardas
    db = _base(tmp_path / "a.db", [])
    salidas = iter(["777\n", "777\n", ""])
    log = gemelo / "e8" / "candado.log"
    esperas = guardas.esperar_candado("S10 folds", db, log, gpu=lambda: next(salidas),
                                      dormir=lambda s: None)
    assert esperas == 2
    assert log.read_text().count("gpu=[777]") == 2


def test_sin_espera_queda_igual_la_linea_de_libre(gemelo, tmp_path):
    """CG8 se lee del log: cada etapa tiene que dejar constancia de que arranco libre."""
    import guardas
    log = gemelo / "e8" / "candado.log"
    assert guardas.esperar_candado("S12", _base(tmp_path / "a.db", []), log, gpu=lambda: "",
                                   dormir=lambda s: None) == 0
    assert "libre" in log.read_text()
