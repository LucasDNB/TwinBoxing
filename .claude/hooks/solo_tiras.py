#!/usr/bin/env python3
"""
BoxTwin - Hook PreToolUse del subagente etiquetador-golpes: lo deja ciego.

POR QUE EXISTE
  El piloto E0 mide cuanto le acierta Claude a la anotacion de Lucas mirando solo la tira. Si
  el subagente pudiera abrir la clave del lote, una anotacion, un export de clips (las carpetas
  llevan el nombre de la clase) o el resultado de otro lote, el acuerdo no mediria nada. Las
  reglas de permisos del proyecto valen para todas las sesiones del repo; este hook vale solo
  para este subagente y bloquea todo lo que no sea su trabajo.

QUE HACE
  Lee el pedido de herramienta por stdin. Deja pasar:
    Read  de <trabajo>/lotes/<lote>/tiras/*.png y de <trabajo>/lotes/<lote>/lote.jsonl
    Glob  dentro de <trabajo>/lotes/<lote>/tiras/
    Write de <trabajo>/lotes/<lote>/respuestas/*.jsonl
    SubagentHandback, que devuelve el reporte al que lo lanzo y no abre archivos. Sin
    ella el etiquetador no puede reportar y gasta turnos reintentando (canario, 14-10).
  Todo lo demas sale con codigo 2, que Claude Code trata como bloqueo y le muestra el motivo.

USO
  Lo declara el frontmatter de .claude/agents/etiquetador-golpes.md. Prueba manual:
  echo '{"tool_name":"Read","tool_input":{"file_path":"/x/clave/clave.csv"}}' | python3 .claude/hooks/solo_tiras.py
"""

import json
import os
import sys
from pathlib import Path

TRABAJO = Path(os.path.realpath(os.path.expanduser(
    os.environ.get("BOXTWIN_CLASIFICADOR", "~/boxtwin/clasificador-2026-10"))))
LOTES = TRABAJO / "lotes"


def _real(p: str) -> Path:
    return Path(os.path.realpath(os.path.expanduser(p)))


def _partes(p: Path):
    """(lote, subcarpeta, resto) si p esta bajo <trabajo>/lotes/, si no None."""
    try:
        rel = p.relative_to(LOTES)
    except ValueError:
        return None
    return rel.parts


def permitido(herramienta: str, entrada: dict) -> tuple[bool, str]:
    if herramienta == "SubagentHandback":
        return True, ""
    if herramienta == "Read":
        partes = _partes(_real(entrada.get("file_path", "")))
        if partes and len(partes) == 3 and partes[1] == "tiras" and partes[2].endswith(".png"):
            return True, ""
        if partes and len(partes) == 2 and partes[1] == "lote.jsonl":
            return True, ""
        return False, "solo podes leer las tiras (tiras/*.png) y lote.jsonl del lote"
    if herramienta == "Glob":
        base = entrada.get("path") or ""
        patron = entrada.get("pattern", "")
        partes = _partes(_real(base)) if base else None
        if partes and len(partes) == 2 and partes[1] == "tiras" and ".." not in patron and not patron.startswith("/"):
            return True, ""
        return False, "Glob solo dentro de la carpeta tiras/ del lote, con path explicito"
    if herramienta == "Write":
        partes = _partes(_real(entrada.get("file_path", "")))
        if partes and len(partes) == 3 and partes[1] == "respuestas" and partes[2].endswith(".jsonl"):
            return True, ""
        return False, "solo podes escribir respuestas/<nombre>.jsonl dentro del lote"
    return False, f"{herramienta} no esta permitida para el etiquetador"


def main() -> int:
    try:
        pedido = json.load(sys.stdin)
    except json.JSONDecodeError:
        print("pedido ilegible: bloqueado", file=sys.stderr)
        return 2
    ok, motivo = permitido(pedido.get("tool_name", ""), pedido.get("tool_input") or {})
    if ok:
        return 0
    print(f"bloqueado por el cegado del piloto: {motivo}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
