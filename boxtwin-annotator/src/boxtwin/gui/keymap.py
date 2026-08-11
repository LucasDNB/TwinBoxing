"""
BoxTwin - Mapa de teclas, remapeable.

POR QUE EXISTE
  La anotacion con mouse es demasiado lenta para volumen. La medicion previa del proyecto
  sobre reanotacion de clips dio 2,2 s por decision, y ese numero solo se sostiene si la
  mano no se mueve del teclado.
  El mapa es un archivo y no constantes en el codigo porque las teclas comodas dependen de
  la distribucion del teclado y de la mano del que anota, y cambiarlas no puede requerir
  tocar fuente.

QUE HACE
  Define las acciones y sus teclas por defecto, las carga desde config.yaml si existe y
  rechaza los conflictos en vez de dejar que gane una en silencio.

USO
  keymap = Keymap.load(Path("proyecto/config.yaml"))
  keymap.sequence("player.step_forward")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Keymap", "DEFAULT_KEYMAP", "KeymapError"]


class KeymapError(ValueError):
    pass


# Las acciones de anotacion e identidad se declaran ya aunque sus manejadores lleguen en
# los bloques 4 y 5: asi el archivo de configuracion del usuario no cambia de forma
# despues, y quien remapea lo hace una vez.
DEFAULT_KEYMAP: dict[str, str] = {
    # reproduccion
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
    # vista
    "view.zoom_in": "Ctrl++",
    "view.zoom_out": "Ctrl+-",
    "view.fit": "Ctrl+0",
    "view.toggle_skeleton": "K",
    "view.toggle_boxes": "X",
    "view.toggle_ids": "I",
    "view.toggle_gloves": "L",
    "view.only_selected": "O",
    # seleccion de peleador
    "fighter.select_a": "1",
    "fighter.select_b": "2",
    # anotacion (bloque 4)
    "event.mark_start": "[",
    "event.mark_end": "]",
    "event.mark_peak": "P",
    "event.side_left": "Q",
    "event.side_right": "W",
    "event.type_straight": "A",
    "event.type_hook": "S",
    "event.type_uppercut": "D",
    "event.target_head": "H",
    "event.target_body": "B",
    "event.feint": "F",
    "event.delete": "Del",
    "event.goto_start": "Ctrl+[",
    "event.goto_end": "Ctrl+]",
    # generales
    "edit.undo": "Z",
    "edit.undo_global": "Ctrl+Z",
    "edit.redo": "Ctrl+Shift+Z",
    "file.save": "Ctrl+S",
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
                raise KeymapError(
                    f"acciones desconocidas en el keymap: {sorted(desconocidas)}"
                )
            bindings.update({k: str(v) for k, v in propios.items()})

        km = cls(bindings)
        km.validate()
        return km

    def validate(self) -> None:
        """
        Rechaza teclas repetidas.

        Con dos acciones en la misma tecla gana una y la otra deja de responder sin decir
        nada, que en medio de una sesion de anotacion se siente como que la aplicacion se
        colgo.
        """
        vistos: dict[str, str] = {}
        choques: list[str] = []
        for accion, tecla in self.bindings.items():
            if not tecla:
                continue
            if tecla in vistos:
                choques.append(f"{tecla!r}: {vistos[tecla]} y {accion}")
            else:
                vistos[tecla] = accion
        if choques:
            raise KeymapError("teclas repetidas en el keymap: " + "; ".join(sorted(choques)))

    def sequence(self, accion: str) -> str:
        if accion not in self.bindings:
            raise KeymapError(f"accion desconocida: {accion!r}")
        return self.bindings[accion]

    def as_table(self) -> list[tuple[str, str]]:
        """Para mostrar la ayuda de teclas en la interfaz."""
        return sorted(self.bindings.items())
