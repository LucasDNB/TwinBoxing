"""
BoxTwin - Linea de comandos.

POR QUE EXISTE
  El preproceso corre una vez por video, tarda minutos u horas y no necesita GUI. Tenerlo
  como comando permite encolar varios videos, correrlo por ssh y reanudarlo sin abrir la
  aplicacion.
  El subcomando probe existe porque en este proyecto los metadatos declarados ya mintieron
  dos veces y conviene poder verificarlos en dos segundos antes de comprometer horas de
  computo.

QUE HACE
  preprocess  corre pose y tracking sobre un video y escribe el cache, reanudable.
  annotate    abre el anotador de escritorio sobre un video ya preprocesado.
  export      genera el dataset en alguno de los cuatro formatos.
  reanno      sortea la muestra de reanotacion ciega y calcula el acuerdo.
  probe       muestra los metadatos reales del video sin procesar nada.

USO
  boxtwin-annotator preprocess videos/spar.mp4 --project ./proyecto
  boxtwin-annotator probe videos/spar.mp4 --count
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from boxtwin.version import __version__

__all__ = ["main", "build_parser"]

log = logging.getLogger("boxtwin")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="boxtwin-annotator",
        description="Anotador de eventos de golpe sobre video con overlay de pose.",
    )
    p.add_argument("--version", action="version", version=f"boxtwin-annotator {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="log a nivel DEBUG")
    sub = p.add_subparsers(dest="comando", required=True)

    # -- preprocess --------------------------------------------------------
    pre = sub.add_parser(
        "preprocess",
        help="corre pose y tracking sobre un video y escribe el cache",
        description=(
            "Escribe cache/<video>.pose.npz, cache/<video>.meta.json y el proxy. "
            "Es reanudable: si el proceso se corta, volver a correr el mismo comando "
            "retoma desde el ultimo shard completo."
        ),
    )
    pre.add_argument("video", type=Path)
    pre.add_argument(
        "--project", type=Path, default=None,
        help="directorio del proyecto. Por defecto se infiere del video "
             "(si esta en videos/, el proyecto es el directorio padre).",
    )
    pre.add_argument("--model", default="yolov8l-pose.pt", help="checkpoint de pose")
    pre.add_argument("--imgsz", type=int, default=640)
    pre.add_argument("--conf", type=float, default=0.25)
    pre.add_argument("--iou", type=float, default=0.7)
    pre.add_argument("--device", default="0", help="0 para la primera GPU, cpu para CPU")
    pre.add_argument(
        "--half", action="store_true",
        help="inferencia en FP16. Mas rapido pero el cache deja de ser reproducible bit a bit.",
    )
    pre.add_argument("--tracker", type=Path, default=None, help="yaml de BoT-SORT")
    pre.add_argument("--shard-frames", type=int, default=None)
    pre.add_argument("--no-proxy", action="store_true", help="no generar el proxy")
    pre.add_argument(
        "--no-detect-cuts", action="store_true",
        help="no detectar cortes de plano. Por defecto se detectan y el tracker se "
             "reinicia en cada uno: sin eso, en una transmision con varias camaras la "
             "identidad se arrastra entre planos y le pone a un peleador el cuerpo del otro.",
    )
    pre.add_argument("--cut-threshold", type=float, default=0.3)
    pre.add_argument("--proxy-width", type=int, default=960)
    pre.add_argument(
        "--restart", action="store_true",
        help="descarta el trabajo a medias y arranca de cero",
    )
    pre.add_argument(
        "--force", action="store_true",
        help="rehace el cache aunque cuelgue una anotacion. Renumera los track_id, asi que "
             "despues hay que reasignar identidad desde cero.",
    )

    # -- annotate ----------------------------------------------------------
    an = sub.add_parser(
        "annotate",
        help="abre el anotador sobre un video ya preprocesado",
        description=(
            "Necesita el cache de pose. Si falta, correr antes 'preprocess'. "
            "Reproduce desde el proxy si existe."
        ),
    )
    an.add_argument("video", type=Path)
    an.add_argument(
        "--buffer-mb", type=int, default=512,
        help="presupuesto de memoria del buffer de cuadros",
    )
    an.add_argument(
        "--platform", default=None,
        help="plugin de plataforma de Qt: xcb, wayland, vnc, offscreen. Equivale a "
             "QT_QPA_PLATFORM. Con 'vnc' la ventana se sirve por un puerto y se ve con "
             "cualquier cliente VNC, sin necesidad de sesion grafica.",
    )
    an.add_argument(
        "--vnc-size", default="1600x1000",
        help="tamano de la pantalla virtual con --platform vnc. El default del plugin de "
             "Qt es 1024x768 y ahi la ventana del anotador queda recortada.",
    )
    an.add_argument("--vnc-port", type=int, default=5900, help="puerto VNC")

    # -- export ------------------------------------------------------------
    ex = sub.add_parser(
        "export",
        help="genera el dataset a partir de la anotacion",
        description=(
            "Corre sin GUI. Todos los exports llevan en su metadata el sha256 de la "
            "anotacion que los genero, que es lo unico que permite saber meses despues "
            "sobre que datos se entreno un modelo."
        ),
    )
    ex.add_argument("video", type=Path)
    ex.add_argument(
        "--format", required=True, choices=["clips", "mmaction", "sequence", "stats"],
    )
    ex.add_argument("--project", type=Path, default=None)
    ex.add_argument("--out", type=Path, default=None, help="por defecto <proyecto>/exports")
    ex.add_argument(
        "--label-space", default="side", choices=["side", "lead-rear"],
        help="side (default) es el lado observado, sin ningun paso de derivacion. "
             "lead-rear deriva de la guardia del peleador, que la pone el anotador y puede "
             "estar mal o cambiar a mitad del combate; en ese espacio el conjunto de 6 "
             "coincide con las clases de BoxingVI.",
    )
    ex.add_argument("--classes", type=int, default=12, choices=[6, 12, 14])
    ex.add_argument(
        "--persons", default="attacker", choices=["attacker", "both"],
        help="both agrega al rival: da contexto de si el golpe llega, pero cambia el "
             "problema y deja de ser comparable con lo ya entrenado.",
    )
    ex.add_argument(
        "--keypoints", default="coco17", choices=["coco17", "coco17+gloves"],
        help="coco17+gloves agrega los indices 17 y 18. Cambia V y los pesos preentrenados "
             "en NTU dejan de cargar directo.",
    )
    ex.add_argument("--pad", type=int, default=0, help="cuadros de relleno a cada lado")
    ex.add_argument(
        "--background", type=int, default=0,
        help="ejemplos de fondo a muestrear. Solo aplica al espacio de 14 clases.",
    )
    ex.add_argument("--background-len", type=int, default=20)
    ex.add_argument("--seed", type=int, default=42, help="semilla del muestreo de fondo")
    ex.add_argument(
        "--channels", default="per-arm", choices=["per-arm", "per-fighter"],
        help="per-arm es el default: en un solo carril BIO, un 1-2 pierde uno de los dos "
             "golpes porque no entran dos segmentos solapados.",
    )
    ex.add_argument("--crf", type=int, default=20)
    ex.add_argument("--preset", default="veryfast")
    ex.add_argument(
        "--no-mark-fighter", action="store_true",
        help="no dibujar la caja del peleador anotado en los clips. Por defecto se dibuja: "
             "los golpes se solapan y sin la marca el clip contiene el golpe del otro y la "
             "etiqueta parece equivocada.",
    )

    # -- reanno ------------------------------------------------------------
    rn = sub.add_parser(
        "reanno",
        help="muestra de reanotacion ciega y reporte de acuerdo",
        description=(
            "Sin --report sortea la muestra, o muestra el avance si ya existe. La muestra se "
            "congela: se niega a resortear salvo --force, porque resortear despues de ver "
            "resultados parciales convierte el numero en lo que uno quiera que sea."
        ),
    )
    rn.add_argument("video", type=Path)
    rn.add_argument("--fraction", type=float, default=0.10)
    rn.add_argument("--seed", type=int, default=42)
    rn.add_argument("--annotator", default=None)
    rn.add_argument(
        "--force", action="store_true",
        help="resortea descartando la muestra y los intentos anteriores",
    )
    rn.add_argument("--report", action="store_true", help="calcula el acuerdo y lo escribe")
    rn.add_argument(
        "--include-revealed", action="store_true",
        help="incluye en el reporte los intentos con la etiqueta revelada, que no fueron ciegos",
    )
    rn.add_argument("--out", type=Path, default=None)

    # -- proxy -------------------------------------------------------------
    px = sub.add_parser(
        "proxy",
        help="genera solo el proxy de reproduccion (no necesita GPU ni ultralytics)",
        description=(
            "Existe separado del preproceso porque anotar no necesita GPU: se preprocesa "
            "en la maquina con la placa y se anota en otra. Alla alcanza con ffmpeg."
        ),
    )
    px.add_argument("video", type=Path)
    px.add_argument("--project", type=Path, default=None)
    px.add_argument("--width", type=int, default=960)
    px.add_argument("--gop", type=int, default=12)
    px.add_argument("--crf", type=int, default=20)
    px.add_argument("--preset", default="veryfast")
    px.add_argument("--force", action="store_true", help="regenera aunque ya exista")

    # -- bundle ------------------------------------------------------------
    bu = sub.add_parser(
        "bundle",
        help="junta lo minimo para anotar en otra maquina",
        description=(
            "Copia el cache y la anotacion a un directorio con el layout del proyecto, "
            "listo para mover por scp. Sin --with-video no lleva el original: sobre 4K son "
            "490 MB cada 10 minutos contra 97 del proxy, y lo unico que se pierde es el "
            "zoom en resolucion original."
        ),
    )
    bu.add_argument("video", type=Path)
    bu.add_argument("--out", type=Path, required=True, help="directorio destino")
    bu.add_argument("--project", type=Path, default=None)
    bu.add_argument(
        "--with-video", action="store_true",
        help="incluye el video original, que habilita el zoom en resolucion completa",
    )

    # -- probe -------------------------------------------------------------
    pr = sub.add_parser(
        "probe",
        help="muestra los metadatos reales del video",
        description="No procesa nada. Con --count decodifica para verificar el conteo de frames.",
    )
    pr.add_argument("video", type=Path)
    pr.add_argument(
        "--count", action="store_true",
        help="cuenta los frames decodificando y reconcilia el fps. Tarda lo que tarde el video.",
    )

    ia = sub.add_parser(
        "identidad-auto",
        help="asigna identidad a los tracks usando el detector de guantes",
        description=(
            "Tres filtros medidos: altura saca al publico, guante saca al arbitro y al "
            "entrenador, y el color del guante decide A contra B. Lo que no se puede decidir "
            "queda SIN ASIGNAR y lo resuelve el anotador, que es distinto de adivinarlo."
        ),
    )
    ia.add_argument("video", type=Path)
    ia.add_argument("--project", type=Path, default=None)
    ia.add_argument("--modelo", type=Path, required=True,
                    help="pesos del detector de guantes, entrenado sobre recortes de persona")
    ia.add_argument("--segundos", type=float, default=5.0,
                    help="de que parte del principio del video sale el grupo de referencia, "
                         "el que define cual peleador es A. No recorta la evidencia: la "
                         "particion usa el video entero")
    ia.add_argument("--separacion-minima", type=float, default=0.55, dest="separacion_minima",
                    help="separacion de color por debajo de la cual NO se asigna A ni B. Con "
                         "0,449 el reparto salio 13 tracks a A contra 1 a B: una asignacion "
                         "equivocada con cara de correcta")
    ia.add_argument("--paso", type=int, default=5, help="se evalua un cuadro de cada N")
    ia.add_argument("--umbral-guante", type=float, default=0.45, dest="umbral_guante",
                    help="fraccion minima de recortes con guante para considerar peleador. "
                         "El hueco medido va de 0,33 a 0,59, asi que 0,45 esta en el medio")
    ia.add_argument("--fraccion-altura", type=float, default=0.55, dest="fraccion_altura",
                    help="alto minimo relativo al track mas alto del video. Relativo y no "
                         "absoluto porque un gimnasio y una transmision no comparten escala")
    ia.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="calcula y reporta sin escribir el archivo de anotacion")
    ia.add_argument("--forzar", action="store_true",
                    help="aplicar aunque el documento ya tenga correcciones manuales, que "
                         "se descartan")
    ia.add_argument("--device", default=None, help="device de torch, p.ej. 0 o cpu")
    ia.add_argument("--guardar-evidencia", type=Path, default=None, dest="guardar_evidencia",
                    help="volcar lo medido a un json. Recorrer el video cuesta minutos y "
                         "elegir umbrales sobre lo ya medido cuesta milisegundos")
    ia.add_argument("--evidencia", type=Path, default=None,
                    help="usar una evidencia ya guardada en vez de recorrer el video")
    ia.add_argument("--contra-anotacion", action="store_true", dest="contra_anotacion",
                    help="puntuar la propuesta contra las asignaciones manuales del "
                         "documento, sin escribir nada")

    return p


def infer_project_dir(video: Path) -> Path:
    """
    Deduce el directorio del proyecto.

    Si el video esta en <proyecto>/videos/, el proyecto es el padre. Si no, se usa el
    directorio del video. Explicito con --project siempre gana.
    """
    video = video.resolve()
    if video.parent.name == "videos":
        return video.parent.parent
    return video.parent


def _cmd_preprocess(args: argparse.Namespace) -> int:
    from boxtwin.preprocess.runner import PreprocessConfig, default_tracker_path, preprocess

    cfg = PreprocessConfig(
        model=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        half=args.half,
        tracker=args.tracker or default_tracker_path(),
        make_proxy=not args.no_proxy,
        proxy_width=args.proxy_width,
        detect_cuts=not args.no_detect_cuts,
        cut_threshold=args.cut_threshold,
    )
    if args.shard_frames:
        cfg.shard_frames = args.shard_frames

    proyecto = args.project or infer_project_dir(args.video)
    barra = _progress_bar()

    print(f"video    : {args.video}")
    print(f"proyecto : {proyecto}")
    print(f"modelo   : {cfg.model}  imgsz={cfg.imgsz} device={cfg.device} half={cfg.half}")
    print(f"tracker  : {cfg.tracker}")

    r = preprocess(
        args.video, proyecto, cfg, restart=args.restart, force=args.force,
        on_progress=barra,
    )
    barra(None, None)

    fps_efectivo = r.total_frames / r.runtime_s if r.runtime_s else 0.0
    print(f"\nlisto en {r.runtime_s:.1f} s ({fps_efectivo:.1f} fps efectivos)")
    print(f"  frames      : {r.total_frames}")
    print(f"  detecciones : {r.n_detections}")
    print(f"  npz         : {r.npz_path}")
    print(f"  meta        : {r.meta_path}")
    if r.proxy_path:
        print(f"  proxy       : {r.proxy_path}")

    if r.seams:
        print(
            f"\n  ATENCION: {len(r.seams)} costura(s) de reanudacion en los frames "
            f"{[s['frame'] for s in r.seams]}."
        )
        print(
            "  El tracker se reinicio ahi, asi que ningun track cruza esos cuadros y hay\n"
            "  un corte de identidad garantizado. Se resuelve con un re-seed en el anotador."
        )
    if r.suspected_vfr:
        print(
            "\n  ATENCION: el fps declarado no concuerda con el conteo real de frames.\n"
            "  Puede ser framerate variable. Revisar antes de anotar: las fronteras\n"
            "  temporales dependen de que el fps sea constante."
        )
    if r.short_decode:
        print(
            f"\n  ATENCION: se decodificaron {r.total_frames} cuadros y se esperaban "
            f"{r.frames_expected}.\n"
            "  opencv no distingue el fin del archivo de un error de decodificacion a\n"
            "  mitad, asi que puede ser cualquiera de las dos. Verificar con\n"
            "  'boxtwin-annotator probe <video> --count' antes de anotar sobre este cache."
        )
    return 0


def _cmd_identidad_auto(args: argparse.Namespace) -> int:
    from boxtwin.core.annotations import load as load_doc, save as save_doc
    from boxtwin.core.deteccion_guantes import DetectorGuantes
    from boxtwin.core.identidad_auto import (
        AsignarIdentidadAuto, ConfigIdentidadAuto, analizar, cargar_evidencia,
        guardar_evidencia, proponer, puntuar_contra,
    )
    from boxtwin.core.posecache import PoseCache
    from boxtwin.core.project import project_paths

    paths = project_paths(args.video)
    if not paths.npz.is_file():
        print(f"error: falta el cache de pose ({paths.npz.name}). Corre primero preprocess",
              file=sys.stderr)
        return 1
    if not paths.annot.is_file():
        print(f"error: falta el archivo de anotacion ({paths.annot.name})", file=sys.stderr)
        return 1
    if not Path(args.modelo).is_file():
        print(f"error: no existe el modelo {args.modelo}", file=sys.stderr)
        return 1

    doc, _ = load_doc(paths.annot)
    cache = PoseCache.open(paths.npz)
    cfg = ConfigIdentidadAuto(
        segundos_ancla=args.segundos, separacion_minima=args.separacion_minima,
        paso=args.paso,
        umbral_guante=args.umbral_guante, fraccion_altura=args.fraccion_altura,
    )
    detector = DetectorGuantes(str(args.modelo), device=args.device)

    def progreso(f: int, fin: int) -> None:
        print(f"  cuadro {f}/{fin}", file=sys.stderr)

    if args.evidencia:
        ev = cargar_evidencia(args.evidencia)
    else:
        ev = analizar(
            cache, paths.video, detector, cfg,
            alto_imagen=doc.video.height, fps=doc.video.fps,
            total_frames=doc.video.total_frames, progreso=progreso,
        )
        if args.guardar_evidencia:
            guardar_evidencia(ev, args.guardar_evidencia)
            print(f"evidencia guardada en {args.guardar_evidencia}", file=sys.stderr)
    prop = proponer(ev, cfg, doc.video.total_frames, doc.video.fps)
    d = prop.diagnostico

    print(f"\n{d.get('tracks', 0)} tracks en el cache")
    print(f"  pasan el filtro de altura (>= {d.get('umbral_altura', 0):.3f}): "
          f"{d.get('pasan_altura', 0)}")
    print(f"  pasan el filtro de guante (>= {cfg.umbral_guante}): "
          f"{d.get('candidatos_tras_guante', 0)}")
    if prop.semillas:
        print(f"\nparticion: {d.get('tracks_por_coexistencia', 0)} tracks resueltos por "
              f"coexistencia en {d.get('componentes_con_dos_lados', 0)} grupo/s"
              + ("" if d.get("ancla_del_principio")
                 else f"  [el grupo de referencia NO aparece en los primeros "
                      f"{cfg.segundos_ancla:g} s]"))
        if d.get("componentes_sin_orientar"):
            print(f"  {d['componentes_sin_orientar']} grupo/s sin orientar: el color no "
                  "distinguia entre las dos posibilidades")
        for rol, tid in prop.semillas.items():
            t, sat, val = prop.perfiles[rol]
            tono = "sin color" if t < 0 else f"tono {t:.1f}"
            print(f"  {rol}: track {tid}  ({tono}, sat {sat:.0f}, val {val:.0f})")
        print(f"  separacion entre perfiles: {d.get('separacion_de_perfiles', 0):.3f}")
    print(f"\nasignados: {d.get('asignados_A', 0)} tracks a A, "
          f"{d.get('asignados_B', 0)} a B")
    print(f"sin asignar, para que los resuelva el anotador: {len(prop.sin_asignar)}"
          + (f"  {prop.sin_asignar[:12]}" if prop.sin_asignar else ""))
    for aviso in prop.avisos:
        print(f"\nATENCION: {aviso}")

    if args.contra_anotacion:
        pt = puntuar_contra(prop, doc)
        print("\ncontra las asignaciones manuales del documento:")
        if not pt.get("evaluables"):
            print("  el documento no tiene tracks de peleador que la propuesta haya decidido")
        else:
            print(f"  tracks evaluables: {pt['evaluables']}")
            print(f"  acierto de la particion: {pt['aciertos']}/{pt['evaluables']} = "
                  f"{pt['acierto']:.1%}"
                  + ("   [las etiquetas A y B salieron invertidas, que no es un error: "
                     "la propuesta las asigna por posicion]" if pt["etiquetas_invertidas"]
                     else ""))
            print(f"  peleadores que los filtros descartaron: "
                  f"{pt['peleadores_descartados_por_los_filtros']} de "
                  f"{pt['peleadores_en_la_verdad']}")
            print(f"  peleadores que quedaron sin decidir: {pt['peleadores_sin_decidir']}")
        return 0

    if args.dry_run:
        print("\n--dry-run: no se escribio nada")
        return 0
    if not prop.semillas:
        # Igual se aplican los filtros de descarte: sacan la mayoria de los tracks y eso es
        # trabajo que el anotador no tiene que repetir, aunque A y B queden sin decidir.
        if not prop.roles:
            print("\nno hay nada que aplicar", file=sys.stderr)
            return 1
        print("\nno se pudo decidir A contra B, pero los descartes si se aplican")

    cmd = AsignarIdentidadAuto(prop, annotator="identidad-auto", forzar=args.forzar)
    try:
        cmd.do(doc)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    save_doc(doc, paths.annot)
    print(f"\n{cmd.aplicados} asignaciones escritas en {paths.annot.name}")
    print("Revisalo en el anotador: los tracks sin asignar salen en su propio color.")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from boxtwin.core.annotations import load as load_doc
    from boxtwin.core.export import ExportContext, exportadores
    from boxtwin.core.identity import IdentityResolver
    from boxtwin.core.posecache import PoseCache
    from boxtwin.core.project import project_paths

    paths = project_paths(args.video)
    if not paths.npz.is_file():
        print(f"error: falta el cache de pose ({paths.npz.name})", file=sys.stderr)
        return 1
    if not paths.annot.is_file():
        print(f"error: no hay anotacion ({paths.annot.name})", file=sys.stderr)
        return 1

    cache = PoseCache.open(paths.npz)
    doc, migradas = load_doc(paths.annot)
    if migradas:
        print(f"anotacion migrada desde la version {migradas[0]}, backup guardado")

    salida = args.out or (paths.project / "exports")
    ctx = ExportContext(
        doc=doc,
        cache=cache,
        resolver=IdentityResolver(doc, cache),
        video_path=paths.video,
        out_dir=salida,
        opciones={
            "label_space": args.label_space,
            "classes": args.classes,
            "persons": args.persons,
            "keypoints": args.keypoints,
            "pad": args.pad,
            "background": args.background,
            "background_len": args.background_len,
            "seed": args.seed,
            "channels": args.channels,
            "crf": args.crf,
            "preset": args.preset,
            "mark_fighter": not args.no_mark_fighter,
        },
    )

    print(f"exportando {args.format} de {paths.video.name}")
    print(f"  eventos   : {len(doc.events)}")
    print(f"  anotacion : {ctx.doc.video.sha256[:16]}")
    resultado = exportadores()[args.format](ctx)

    print(f"\nlisto: {resultado.formato}")
    for k, v in resultado.resumen.items():
        print(f"  {k}: {v}")
    for a in resultado.archivos:
        print(f"  -> {a}")
    if resultado.avisos:
        print()
        for a in resultado.avisos:
            print(f"  ATENCION: {a}")
    return 0


def _cmd_reanno(args: argparse.Namespace) -> int:
    import os

    from boxtwin.core.agreement import a_dict, a_texto, comparar
    from boxtwin.core.annotations import load as load_doc
    from boxtwin.core.export.base import escribir_json
    from boxtwin.core.project import project_paths
    from boxtwin.core.reanno import SampleFrozenError, cargar, cargar_o_sortear, guardar

    paths = project_paths(args.video)
    if not paths.annot.is_file():
        print(f"error: no hay anotacion ({paths.annot.name})", file=sys.stderr)
        return 1

    doc, _ = load_doc(paths.annot)
    ruta = paths.annot.with_name(paths.annot.name.replace(".annot.json", ".reanno.json"))

    if args.report:
        if not ruta.is_file():
            print(f"error: no hay reanotacion ({ruta.name})", file=sys.stderr)
            return 1
        re_doc = cargar(ruta)
        reporte = comparar(doc, re_doc, solo_ciegos=not args.include_revealed)
        salida = args.out or paths.exports
        salida.mkdir(parents=True, exist_ok=True)
        base = paths.video.stem
        escribir_json(salida / f"{base}.agreement.json", a_dict(reporte))
        texto = a_texto(reporte)
        (salida / f"{base}.agreement.txt").write_text(texto, encoding="utf-8")
        print(texto)
        print(f"-> {salida / (base + '.agreement.json')}")
        return 0

    anotador = args.annotator or os.environ.get("USER") or "desconocido"
    try:
        re_doc, nueva = cargar_o_sortear(
            ruta, doc, fraction=args.fraction, seed=args.seed,
            annotator=anotador, force=args.force,
        )
    except SampleFrozenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if nueva:
        guardar(re_doc, ruta)
        print(f"muestra sorteada: {re_doc.sample.n} de {len(doc.events)} eventos")
        print(f"  semilla {re_doc.sample.seed} · fraccion {re_doc.sample.fraction}")
        print(f"  -> {ruta}")
        print()
        print("  La muestra queda congelada. Resortearla despues de ver resultados")
        print("  parciales anula el pre-registro; hace falta --force explicito.")
    else:
        print(f"muestra existente: {re_doc.sample.n} eventos, semilla {re_doc.sample.seed}")
        print(f"  reanotados {len(re_doc.trials)}, pendientes {len(re_doc.pendientes())}")
        revelados = sum(1 for t in re_doc.trials if t.revealed)
        if revelados:
            print(f"  {revelados} con la etiqueta revelada: no cuentan como ciegos")
    print()
    print("  Reanotar desde la pestana Reanotacion del anotador.")
    return 0


def _cmd_proxy(args: argparse.Namespace) -> int:
    import time

    from boxtwin.preprocess.proxy import ProxyJob

    proyecto = args.project or infer_project_dir(args.video)
    destino = proyecto / "cache" / f"{args.video.stem}.proxy.mp4"
    if destino.is_file() and not args.force:
        print(f"ya existe: {destino}\nUsar --force para regenerarlo.")
        return 0

    if args.force:
        destino.unlink(missing_ok=True)

    print(f"generando proxy de {args.video.name} a {args.width} px...")
    t0 = time.perf_counter()
    ProxyJob(
        src=args.video, dst=destino, width=args.width, gop=args.gop,
        crf=args.crf, preset=args.preset,
    ).start().wait()
    dt = time.perf_counter() - t0
    print(f"listo en {dt:.1f} s -> {destino} ({destino.stat().st_size / 2**20:.1f} MB)")
    return 0


def _cmd_bundle(args: argparse.Namespace) -> int:
    import shutil

    from boxtwin.core.project import project_paths

    origen = project_paths(args.video)
    destino = Path(args.out).resolve()

    piezas: list[tuple[Path, Path, bool]] = [
        (origen.npz, destino / "cache" / origen.npz.name, True),
        (origen.meta, destino / "cache" / origen.meta.name, True),
        (origen.proxy, destino / "cache" / origen.proxy.name, False),
        (origen.annot, destino / "annotations" / origen.annot.name, False),
        (origen.config, destino / "config.yaml", False),
    ]
    if args.with_video:
        piezas.append((origen.video, destino / "videos" / origen.video.name, True))

    faltan = [p.name for p, _, obligatorio in piezas if obligatorio and not p.is_file()]
    if faltan:
        print(f"error: faltan archivos obligatorios: {', '.join(faltan)}", file=sys.stderr)
        return 1

    total = 0
    for src, dst, _ in piezas:
        if not src.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        tam = dst.stat().st_size
        total += tam
        print(f"  {tam / 2**20:8.1f} MB  {dst.relative_to(destino)}")

    if not args.with_video and not origen.proxy.is_file():
        print(
            "\nerror: sin el video original hace falta el proxy, y no esta.\n"
            f"Generarlo con: boxtwin-annotator proxy {origen.video}",
            file=sys.stderr,
        )
        return 1

    print(f"\n{total / 2**20:.1f} MB en {destino}")
    if not args.with_video:
        print("Sin el video original: se anota igual, pero sin zoom en resolucion completa.")
    print(f"\nPara moverlo:\n  rsync -a --info=progress2 {destino}/ usuario@maquina:ruta/proyecto/")
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    from boxtwin.core.video import count_frames_by_decoding, probe, reconcile_fps

    info = probe(args.video)
    print(f"archivo   : {info.path}")
    print(f"resolucion: {info.width}x{info.height}  codec={info.codec}")
    print(f"fps       : {info.fps_rational} = {info.fps_container:.6f}  (avg declarado {info.fps_declared:.6f})")
    print(f"duracion  : {info.duration_s:.3f} s")
    declarado = info.nb_frames_declared
    print(f"nb_frames : {declarado if declarado is not None else 'N/A (el contenedor no lo declara)'}")

    if args.count:
        real = count_frames_by_decoding(args.video)
        v = reconcile_fps(info, real)
        print(f"\nframes reales      : {real}")
        print(f"fps segun conteo   : {v.fps_from_count:.6f}  (error relativo {v.relative_error:.4%})")
        print(f"veredicto          : {v.source.value}  -> se usa fps={v.fps:.6f}")
        if v.suspected_vfr:
            print("ATENCION: sospecha de framerate variable.")
    return 0


def _progress_bar():
    """Barra con tqdm si esta, y si no una linea de texto cada tanto."""
    estado: dict[str, object] = {"bar": None}

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    def actualizar(frame: int | None, total: int | None) -> None:
        if frame is None:
            bar = estado.get("bar")
            if bar is not None:
                bar.close()  # type: ignore[union-attr]
            return
        if tqdm is None:
            pct = 100 * frame / total if total else 0
            print(f"\r  {frame}/{total} ({pct:.1f}%)", end="", flush=True)
            return
        bar = estado.get("bar")
        if bar is None:
            bar = tqdm(total=total, unit="f", desc="  pose", dynamic_ncols=True)
            estado["bar"] = bar
        bar.update(frame - bar.n)  # type: ignore[union-attr]

    return actualizar


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.comando == "preprocess":
            return _cmd_preprocess(args)
        if args.comando == "annotate":
            from boxtwin.gui.app import run

            return run(
                args.video,
                buffer_mb=args.buffer_mb,
                platform=args.platform,
                vnc_size=args.vnc_size,
                vnc_port=args.vnc_port,
            )
        if args.comando == "export":
            return _cmd_export(args)
        if args.comando == "reanno":
            return _cmd_reanno(args)
        if args.comando == "proxy":
            return _cmd_proxy(args)
        if args.comando == "bundle":
            return _cmd_bundle(args)
        if args.comando == "probe":
            return _cmd_probe(args)
        if args.comando == "identidad-auto":
            return _cmd_identidad_auto(args)
    except KeyboardInterrupt:
        print("\ninterrumpido. El trabajo hecho quedo persistido: volve a correr el "
              "mismo comando para reanudar.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        log.debug("fallo", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
