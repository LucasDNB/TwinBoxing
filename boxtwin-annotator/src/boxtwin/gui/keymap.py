"""
BoxTwin - Mapa de teclas, remapeable.

POR QUE EXISTE
  La anotacion con mouse es demasiado lenta para volumen. La medicion previa del proyecto
  sobre reanotacion de clips dio 2,2 s por decision, y ese numero solo se sostiene si la
  mano no se mueve del teclado.
  El mapa es un archivo y no constantes en el codigo porque las teclas comodas dependen de
  la distribucion del teclado y de la mano del que anota, y cambiarlas no puede requerir
  tocar fuente.

  Las teclas se validan POR CONTEXTO y no globalmente. Espacio es reproducir en el
  reproductor y pausar el preview en el dialogo de clasificacion, y son la misma tecla a
  proposito: el dialogo esta modal encima, asi que nunca compiten. Validar en una sola
  bolsa obligaria a inventar teclas distintas para la misma accion mental.

QUE HACE
  Define las acciones, su contexto y sus teclas por defecto, las carga desde config.yaml si
  existe y rechaza los conflictos dentro de cada contexto en vez de dejar que gane una en
  silencio.

USO
  keymap = Keymap.load(Path("proyecto/config.yaml"))
  keymap.sequence("player.step_forward")
  keymap.accion_de("dialog", "a")      # -> "event.type_straight"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Keymap", "DEFAULT_KEYMAP", "CONTEXTOS", "KeymapError"]


class KeymapError(ValueError):
    pass


DEFAULT_KEYMAP: dict[str, str] = {
    # -- reproductor ------------------------------------------------------
    "player.play_pause": "Space",
    "player.play_backward": "Shift+Space",
    "player.step_forward": "Right",
    "player.step_back": "Left",
    "player.step_forward_5": "Shift+Right",
    "player.step_back_5": "Shift+Left",
    "player.step_forward_1s": "Ctrl+Right",
    "player.step_back_1s": "Ctrl+Left",
    "player.go_start": "Home",
    "player.go_end": "End",
    "player.goto_frame": "G",
    "player.speed_up": "+",
    "player.speed_down": "-",
    # -- vista ------------------------------------------------------------
    "view.zoom_in": "Ctrl++",
    "view.zoom_out": "Ctrl+-",
    "view.fit": "Ctrl+0",
    "view.toggle_skeleton": "K",
    "view.toggle_boxes": "X",
    "view.toggle_ids": "I",
    "view.toggle_gloves": "L",
    "view.only_selected": "O",
    # -- seleccion de peleador --------------------------------------------
    # Identidad: se asigna el track que esta debajo del ultimo click sobre el video.
    # Sin esto hay que buscar el track_id en una lista lateral, y son 23 s por asignacion
    # medidos sobre un round real, contra 2 o 3 clickeando al boxeador.
    "identity.assign_a": "Ctrl+1",
    "identity.assign_b": "Ctrl+2",
    "identity.ignore": "Ctrl+3",
    "identity.ignore_rest": "Ctrl+4",

    "fighter.select_a": "1",
    "fighter.select_b": "2",
    # -- marcado, sobre el reproductor ------------------------------------
    "event.mark_start": "[",
    "event.mark_end": "]",
    "event.cancel_open": "Z",
    "event.goto_start": "Ctrl+[",
    "event.goto_end": "Ctrl+]",
    "event.delete": "Del",
    # -- clasificacion, dentro del dialogo --------------------------------
    "event.side_left": "Q",
    "event.side_right": "W",
    "event.type_straight": "A",
    "event.type_hook": "S",
    "event.type_uppercut": "D",
    "event.target_head": "H",
    "event.target_body": "B",
    "event.feint": "F",
    "event.aborted": "Shift+F",
    "event.mark_peak": "P",
    "event.landed": "4",
    "event.blocked": "5",
    "event.slipped": "6",
    "event.missed": "7",
    "event.landed_unknown": "8",
    "event.quality_clean": "C",
    "event.quality_partial": "V",
    "event.quality_ambiguous": "N",
    "event.preview_toggle": "Space",
    "event.confirm": "Return",
    "event.abort_dialog": "Escape",
    # -- generales --------------------------------------------------------
    "edit.undo": "Ctrl+Z",
    "edit.redo": "Ctrl+Shift+Z",
    "file.save": "Ctrl+S",
}

# Contexto en que vive cada accion. Las del dialogo solo responden con el dialogo abierto.
CONTEXTOS: dict[str, str] = {
    **{a: "player" for a in DEFAULT_KEYMAP if a.split(".")[0] in ("player", "view", "fighter")},
    **{
        a: "player"
        for a in (
            "event.mark_start", "event.mark_end", "event.cancel_open",
            "event.goto_start", "event.goto_end", "event.delete",
        )
    },
    **{
        a: "dialog"
        for a in (
            "event.side_left", "event.side_right", "event.type_straight",
            "event.type_hook", "event.type_uppercut", "event.target_head",
            "event.target_body", "event.feint", "event.aborted", "event.mark_peak",
            "event.landed", "event.blocked", "event.slipped", "event.missed",
            "event.landed_unknown", "event.quality_clean", "event.quality_partial",
            "event.quality_ambiguous", "event.preview_toggle", "event.confirm",
            "event.abort_dialog",
        )
    },
    "edit.undo": "player",
    "edit.redo": "player",
    "file.save": "player",
}


@dataclass
class Keymap:
    bindings: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_KEYMAP))

    @classmethod
    def load(cls, config_path: Path | None) -> Keymap:
        """
        Carga el mapa desde config.yaml.

        Lo del archivo se superpone al default, no lo reemplaza: agregar una accion nueva en
        una version posterior no puede dejar sin teclas a quien ya tenia su config.
        """
        bindings = dict(DEFAULT_KEYMAP)
        if config_path and Path(config_path).is_file():
            import yaml

            data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
            propios = (data.get("keymap") or {}) if isinstance(data, dict) else {}
            desconocidas = set(propios) - set(DEFAULT_KEYMAP)
            if desconocidas:
                raise KeymapError(f"acciones desconocidas en el keymap: {sorted(desconocidas)}")
            bindings.update({k: str(v) for k, v in propios.items()})

        km = cls(bindings)
        km.validate()
        return km

    def validate(self) -> None:
        """
        Rechaza teclas repetidas DENTRO de un mismo contexto.

        Con dos acciones del mismo contexto en la misma tecla gana una y la otra deja de
        responder sin decir nada, que en medio de una sesion se siente como que la
        aplicacion se colgo. Entre contextos distintos la repeticion es deliberada.
        """
        vistos: dict[tuple[str, str], str] = {}
        choques: list[str] = []
        for accion, tecla in self.bindings.items():
            if not tecla:
                continue
            clave = (CONTEXTOS.get(accion, "player"), tecla)
            if clave in vistos:
                choques.append(f"{tecla!r} en {clave[0]}: {vistos[clave]} y {accion}")
            else:
                vistos[clave] = accion
        if choques:
            raise KeymapError("teclas repetidas en el keymap: " + "; ".join(sorted(choques)))

    def sequence(self, accion: str) -> str:
        if accion not in self.bindings:
            raise KeymapError(f"accion desconocida: {accion!r}")
        return self.bindings[accion]

    def contexto(self, accion: str) -> str:
        return CONTEXTOS.get(accion, "player")

    def acciones_de(self, contexto: str) -> dict[str, str]:
        """Acciones de un contexto y su tecla, para armar el manejador del dialogo."""
        return {
            a: t for a, t in self.bindings.items() if CONTEXTOS.get(a, "player") == contexto and t
        }

    def as_table(self) -> list[tuple[str, str]]:
        """Para mostrar la ayuda de teclas en la interfaz."""
        return sorted(self.bindings.items())
