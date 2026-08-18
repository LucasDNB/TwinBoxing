"""
Los cuatro formatos de export.

Lo que se fija es la forma de los arrays, que es contrato con MMAction2, y el determinismo,
que es lo que hace que dos modelos entrenados sobre el mismo comando sean comparables. Si
el export no es reproducible, ninguna comparacion entre corridas significa nada.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from boxtwin.core.export import ExportContext, annot_hash, exportadores
from boxtwin.core.export.labels import BACKGROUND, FEINT, LabelSpace, class_list
from boxtwin.core.export.sequence import bio_names
from boxtwin.core.identity import IdentityResolver
from boxtwin.core.identity_ops import AssignRole
from boxtwin.core.schema import AnnotationDoc, Event, EventMetrics
from boxtwin.core.types import (
    Completeness,
    FighterId,
    Guard,
    Landed,
    PunchType,
    Quality,
    Side,
    Target,
    TrackRole,
)
from boxtwin.core.undo import UndoStack

from tests.test_identity_ops import CAJA, cache_con

AR = timezone(timedelta(hours=-3))
T0 = datetime(2026, 8, 11, 14, 0, 0, tzinfo=AR)
N_FRAMES = 400


def ev(eid: str, ini: int, fin: int, **kw) -> Event:
    base = dict(
        id=eid, fighter=FighterId.A, start_frame=ini, end_frame=fin,
        side=Side.LEFT, punch_type=PunchType.STRAIGHT, target=Target.HEAD,
        completeness=Completeness.FULL, landed=Landed.LANDED,
        guard=Guard.ORTHODOX, quality=Quality.CLEAN,
    )
    base.update(kw)
    return Event(**base)


@pytest.fixture
def ctx(doc_min: AnnotationDoc, tmp_path: Path) -> ExportContext:
    doc = doc_min
    doc.video.total_frames = N_FRAMES
    doc.events = [
        ev("ev_0001", 20, 40),
        ev("ev_0002", 60, 80, side=Side.RIGHT, punch_type=PunchType.HOOK),
        ev("ev_0003", 100, 120, target=Target.BODY),
        ev("ev_0004", 150, 170, completeness=Completeness.FEINT),
        ev("ev_0005", 200, 220, completeness=Completeness.ABORTED),
    ]
    doc.process.event_metrics = {
        e.id: EventMetrics(
            annotator="tester", session_id="se_0001", created_at=T0, confirmed_at=T0,
            active_ms=5000 + i * 100, replays=i,
        )
        for i, e in enumerate(doc.events)
    }
    cache = cache_con({f: [(1, CAJA)] for f in range(N_FRAMES)}, N_FRAMES)
    UndoStack(doc).do(
        AssignRole(track_id=1, role=TrackRole.A, start_frame=0, end_frame_excl=N_FRAMES)
    )
    return ExportContext(
        doc=doc, cache=cache, resolver=IdentityResolver(doc, cache),
        video_path=tmp_path / "videos" / "mini.mp4", out_dir=tmp_path / "exports",
    )


def correr(ctx: ExportContext, formato: str, **opciones):
    ctx.opciones.update(opciones)
    return exportadores()[formato](ctx)


# -- trazabilidad ----------------------------------------------------------


def test_todos_llevan_el_hash_de_la_anotacion(ctx: ExportContext) -> None:
    """
    Es lo unico que permite saber meses despues sobre que datos se entreno un modelo.
    """
    esperado = annot_hash(ctx.doc)
    for formato in ("mmaction", "sequence", "stats"):
        r = correr(ctx, formato)
        meta = next(a for a in r.archivos if a.suffix == ".json")
        assert json.loads(meta.read_text(encoding="utf-8"))["annot_sha256"] == esperado


def test_el_hash_cambia_si_cambia_la_anotacion(ctx: ExportContext) -> None:
    antes = annot_hash(ctx.doc)
    ctx.doc.events[0].target = Target.BODY
    assert annot_hash(ctx.doc) != antes


# -- mmaction --------------------------------------------------------------


def test_mmaction_formas(ctx: ExportContext) -> None:
    """Contrato con PoseConv3D: (M, T, V, C) y (M, T, V)."""
    r = correr(ctx, "mmaction", classes=12)
    pkl = next(a for a in r.archivos if a.suffix == ".pkl")
    d = pickle.loads(pkl.read_bytes())

    assert set(d) == {"split", "annotations"}
    for m in d["annotations"]:
        M, T, V, C = m["keypoint"].shape
        assert (M, V, C) == (1, 17, 2)
        assert m["keypoint_score"].shape == (M, T, V)
        assert m["total_frames"] == T
        assert m["img_shape"] == m["original_shape"] == (
            ctx.doc.video.height, ctx.doc.video.width
        )
        assert m["keypoint"].dtype == np.float32


def test_mmaction_excluye_amagues_y_abortados(ctx: ExportContext) -> None:
    r = correr(ctx, "mmaction", classes=12)
    assert r.resumen["samples"] == 3  # de los cinco eventos
    assert r.resumen["events_skipped"] == 2


def test_mmaction_con_14_incluye_el_amague(ctx: ExportContext) -> None:
    r = correr(ctx, "mmaction", classes=14)
    assert r.resumen["per_class"][FEINT] == 1
    # el abortado sigue afuera: no describe ninguna categoria
    assert r.resumen["samples"] == 4


def test_mmaction_con_fondo(ctx: ExportContext) -> None:
    r = correr(ctx, "mmaction", classes=14, background=5, background_len=10, seed=3)
    assert r.resumen["per_class"][BACKGROUND] == 5


def test_pedir_fondo_sin_clase_de_fondo_avisa(ctx: ExportContext) -> None:
    r = correr(ctx, "mmaction", classes=12, background=5)
    assert any("no tiene clase de fondo" in a for a in r.avisos)
    assert r.resumen["background"] == 0


def test_mmaction_con_dos_personas(ctx: ExportContext) -> None:
    r = correr(ctx, "mmaction", classes=12, persons="both")
    d = pickle.loads(next(a for a in r.archivos if a.suffix == ".pkl").read_bytes())
    assert d["annotations"][0]["keypoint"].shape[0] == 2


def test_mmaction_con_guantes_cambia_v(ctx: ExportContext) -> None:
    """
    V pasa de 17 a 19 y los pesos preentrenados en NTU dejan de cargar directo: por eso no
    es el default.
    """
    r = correr(ctx, "mmaction", classes=12, keypoints="coco17+gloves")
    d = pickle.loads(next(a for a in r.archivos if a.suffix == ".pkl").read_bytes())
    assert d["annotations"][0]["keypoint"].shape[2] == 19


def test_mmaction_avisa_sobre_el_split(ctx: ExportContext) -> None:
    """
    Partir por evento filtra datos: en una combinacion las ventanas de dos golpes comparten
    cuadros. El aviso va en el archivo porque el que arme el split no va a leer la consola.
    """
    r = correr(ctx, "mmaction", classes=12)
    meta = json.loads(next(a for a in r.archivos if a.suffix == ".json").read_text())
    assert "por video" in meta["warning_split"]


def test_mmaction_es_determinista(ctx: ExportContext, tmp_path: Path) -> None:
    a = correr(ctx, "mmaction", classes=14, background=4, seed=9)
    primero = next(x for x in a.archivos if x.suffix == ".pkl").read_bytes()
    ctx.out_dir = tmp_path / "otra"
    b = correr(ctx, "mmaction", classes=14, background=4, seed=9)
    assert next(x for x in b.archivos if x.suffix == ".pkl").read_bytes() == primero


# -- sequence --------------------------------------------------------------


def test_sequence_formas_y_carriles(ctx: ExportContext) -> None:
    r = correr(ctx, "sequence", classes=12)
    z = np.load(next(a for a in r.archivos if a.suffix == ".npz"))
    assert z["labels"].shape == (2, 2, N_FRAMES)  # 2 peleadores, 2 brazos
    assert z["valid"].shape == (2, N_FRAMES)
    assert z["keypoints"].shape == (2, N_FRAMES, 17, 2)
    assert list(z["lanes"]) == ["left", "right"]
    assert len(z["bio_names"]) == 1 + 2 * 12


def test_sequence_escribe_bio(ctx: ExportContext) -> None:
    r = correr(ctx, "sequence", classes=12)
    z = np.load(next(a for a in r.archivos if a.suffix == ".npz"))
    clases = list(z["classes"])
    idx = clases.index("straight-left-head")
    b, i = 1 + 2 * idx, 2 + 2 * idx
    carril = z["labels"][0, 0]  # fighter_A, brazo izquierdo
    assert carril[20] == b  # primer cuadro del evento
    assert carril[21] == i
    assert carril[40] == i
    assert carril[41] == 0  # fuera


def test_sequence_separa_los_brazos(ctx: ExportContext) -> None:
    """
    Un 1-2 tiene los dos golpes solapados. En un solo carril habria que descartar uno, y es
    el golpe mas frecuente del boxeo.
    """
    ctx.doc.events = [
        ev("ev_0001", 20, 45, side=Side.LEFT),
        ev("ev_0002", 30, 55, side=Side.RIGHT, punch_type=PunchType.HOOK),
    ]
    r = correr(ctx, "sequence", classes=12)
    z = np.load(next(a for a in r.archivos if a.suffix == ".npz"))
    assert z["labels"][0, 0, 35] != 0  # el jab sigue etiquetado
    assert z["labels"][0, 1, 35] != 0  # y el cross tambien
    assert r.resumen["same_arm_truncations"] == 0


def test_sequence_trunca_solapamiento_del_mismo_brazo(ctx: ExportContext) -> None:
    """Doble jab: se trunca en el export y se reporta, sin tocar el annot.json."""
    ctx.doc.events = [
        ev("ev_0001", 20, 45, side=Side.LEFT),
        ev("ev_0002", 40, 60, side=Side.LEFT),
    ]
    r = correr(ctx, "sequence", classes=12)
    assert r.resumen["same_arm_truncations"] == 1
    assert any("no se modifico" in a for a in r.avisos)
    assert len(ctx.doc.events) == 2  # el documento quedo intacto


def test_sequence_per_fighter_avisa_de_la_perdida(ctx: ExportContext) -> None:
    r = correr(ctx, "sequence", classes=12, channels="per-fighter")
    z = np.load(next(a for a in r.archivos if a.suffix == ".npz"))
    assert z["labels"].shape[1] == 1
    assert any("pierde uno de los dos golpes" in a for a in r.avisos)


def test_sequence_mascara_de_validez(ctx: ExportContext) -> None:
    """Un cuadro sin identidad no es fondo, es ausencia de dato."""
    r = correr(ctx, "sequence", classes=12)
    z = np.load(next(a for a in r.archivos if a.suffix == ".npz"))
    assert z["valid"][0].all()  # fighter_A tiene identidad en todo el video
    assert not z["valid"][1].any()  # fighter_B no tiene ninguna


def test_bio_names() -> None:
    assert bio_names(["a", "b"]) == ["O", "B-a", "I-a", "B-b", "I-b"]


def test_sequence_es_determinista(ctx: ExportContext, tmp_path: Path) -> None:
    a = correr(ctx, "sequence", classes=12)
    primero = next(x for x in a.archivos if x.suffix == ".npz").read_bytes()
    ctx.out_dir = tmp_path / "otra"
    b = correr(ctx, "sequence", classes=12)
    assert next(x for x in b.archivos if x.suffix == ".npz").read_bytes() == primero


# -- stats -----------------------------------------------------------------


def test_stats_reporta_los_dos_espacios(ctx: ExportContext) -> None:
    """
    El lado es lo que se anota; el desbalance que le importa al clasificador es el del
    espacio del export, y con guardias distintas los dos no coinciden.
    """
    r = correr(ctx, "stats")
    d = json.loads(next(a for a in r.archivos if a.suffix == ".json").read_text())
    assert set(d["distribucion"]) == {"side", "lead-rear"}
    assert set(d["distribucion"]["side"]) == {"6", "12", "14"}


def test_stats_cuenta_por_completitud(ctx: ExportContext) -> None:
    r = correr(ctx, "stats")
    d = json.loads(next(a for a in r.archivos if a.suffix == ".json").read_text())
    assert d["eventos"]["total"] == 5
    assert d["eventos"]["por_completitud"] == {"aborted": 1, "feint": 1, "full": 3}


def test_stats_incluye_duraciones_y_proceso(ctx: ExportContext) -> None:
    r = correr(ctx, "stats")
    d = json.loads(next(a for a in r.archivos if a.suffix == ".json").read_text())
    assert d["duracion_global"]["n"] == 3
    assert d["duracion_global"]["media"] == 21.0
    assert d["proceso"]["eventos_con_metricas"] == 5
    assert d["proceso"]["ms_por_evento"]["mediana"] is not None


def test_stats_escribe_texto_legible(ctx: ExportContext) -> None:
    r = correr(ctx, "stats")
    txt = next(a for a in r.archivos if a.suffix == ".txt").read_text(encoding="utf-8")
    assert "distribucion en side" in txt
    assert "proceso de anotacion" in txt
    # El hash va en el reporte legible tambien: es el que se pega en el capitulo.
    assert annot_hash(ctx.doc)[:16] in txt


def test_stats_avisa_del_desbalance(ctx: ExportContext) -> None:
    ctx.doc.events = [ev(f"ev_{i:04d}", i * 30, i * 30 + 20) for i in range(1, 8)]
    r = correr(ctx, "stats")
    assert any("clases no tienen ningun ejemplo" in a for a in r.avisos)


def test_stats_es_determinista(ctx: ExportContext, tmp_path: Path) -> None:
    """El json lleva created_at, asi que se compara todo menos eso."""
    a = correr(ctx, "stats")
    d1 = json.loads(next(x for x in a.archivos if x.suffix == ".json").read_text())
    ctx.out_dir = tmp_path / "otra"
    b = correr(ctx, "stats")
    d2 = json.loads(next(x for x in b.archivos if x.suffix == ".json").read_text())
    d1.pop("created_at"), d2.pop("created_at")
    assert d1 == d2


# -- registro --------------------------------------------------------------


def test_los_cuatro_formatos_estan_registrados() -> None:
    assert set(exportadores()) == {"clips", "mmaction", "sequence", "stats"}


def test_el_espacio_por_defecto_es_side(ctx: ExportContext) -> None:
    """
    side no tiene paso de derivacion: lo que el anotador vio es lo que se exporta. lead-rear
    depende de la guardia, que puede estar mal o cambiar a mitad del combate, y un error ahi
    intercambia sistematicamente dos clases sin que nada lo delate.
    """
    r = correr(ctx, "sequence", classes=6)
    z = np.load(next(a for a in r.archivos if a.suffix == ".npz"))
    assert list(z["classes"]) == [
        "straight-left", "straight-right",
        "hook-left", "hook-right",
        "uppercut-left", "uppercut-right",
    ]
    meta = json.loads(next(a for a in r.archivos if a.suffix == ".json").read_text())
    assert meta["label_space"] == "side"


def test_lead_rear_sigue_disponible(ctx: ExportContext) -> None:
    """El enunciado pide los dos espacios; cambio el default, no la oferta."""
    r = correr(ctx, "sequence", classes=6, label_space="lead-rear")
    z = np.load(next(a for a in r.archivos if a.suffix == ".npz"))
    assert list(z["classes"])[0] == "straight-lead"


# -- marca del peleador anotado --------------------------------------------


def test_marca_una_caja_por_cuadro() -> None:
    """
    Una por cuadro y no una fija: el peleador se mueve, y justo en el golpe es cuando mas se
    desplaza, asi que una caja promedio marca el lugar equivocado en el momento que importa.
    """
    from boxtwin.core.export.clips import _marca

    filtro = _marca([(10.0, 20.0, 50.0, 80.0), (12.0, 22.0, 52.0, 82.0)], "0xEC584C")
    assert filtro.count("drawbox") == 2
    assert "enable='eq(n\\,0)'" in filtro
    assert "enable='eq(n\\,1)'" in filtro
    assert "x=10:y=20:w=40:h=60" in filtro
    assert filtro.startswith(",")


def test_marca_saltea_los_cuadros_sin_deteccion() -> None:
    """Sin deteccion no se dibuja nada: no se sabe donde esta y no se inventa."""
    from boxtwin.core.export.clips import _marca

    filtro = _marca([(10.0, 20.0, 50.0, 80.0), None, (12.0, 22.0, 52.0, 82.0)], "0xEC584C")
    assert filtro.count("drawbox") == 2
    assert "enable='eq(n\\,1)'" not in filtro
    assert "enable='eq(n\\,2)'" in filtro


def test_marca_vacia_no_rompe_el_filtro() -> None:
    from boxtwin.core.export.clips import _marca

    assert _marca([None, None], "0xEC584C") == ""


def test_marca_con_caja_degenerada_no_emite_ancho_cero() -> None:
    """ffmpeg rechaza w=0; una caja de ancho nulo tiene que salir con 1 y no romper el clip."""
    from boxtwin.core.export.clips import _marca

    filtro = _marca([(10.0, 20.0, 10.0, 20.0)], "0xEC584C")
    assert "w=1:h=1" in filtro
