"""
T1 de docs/specs/2026-10-25-dos-instancias/02-academica.md: `_commit()` resuelve un worktree.

La instancia academica corre desde un worktree desacoplado, donde `.git` es un archivo con
`gitdir:`. Sin esto, /salud de B diria `commit: null` y CA2 no se podria verificar. Se prueba
con repos falsos en tmp_path, sin subprocesos, como lee la funcion.
"""

from __future__ import annotations

from boxtwin_api.app import _commit

H1 = "1111111aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
H2 = "2222222bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
H3 = "3333333ccccccccccccccccccccccccccccccccc"


def test_carpeta_git_con_referencia_suelta(tmp_path):
    git = tmp_path / ".git"
    (git / "refs/heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs/heads/main").write_text(H1 + "\n")
    assert _commit(tmp_path) == H1[:7]


def test_carpeta_git_con_referencia_en_packed_refs(tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "packed-refs").write_text(f"# pack-refs with: peeled\n{H2} refs/heads/main\n")
    assert _commit(tmp_path) == H2[:7]


def _worktree(tmp_path, cabeza: str):
    """Repo principal en tmp/repo y worktree en tmp/wt, como lo arma `git worktree add`."""
    comun = tmp_path / "repo/.git"
    propio = comun / "worktrees/wt"
    propio.mkdir(parents=True)
    (propio / "HEAD").write_text(cabeza + "\n")
    (propio / "commondir").write_text("../..\n")
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {propio}\n")
    return wt, comun


def test_worktree_desacoplado_da_su_hash(tmp_path):
    wt, _ = _worktree(tmp_path, H3)
    assert _commit(wt) == H3[:7]


def test_worktree_en_una_rama_busca_la_referencia_en_el_directorio_comun(tmp_path):
    wt, comun = _worktree(tmp_path, "ref: refs/heads/cambios-academicos")
    (comun / "refs/heads").mkdir(parents=True)
    (comun / "refs/heads/cambios-academicos").write_text(H1 + "\n")
    assert _commit(wt) == H1[:7]
    (comun / "refs/heads/cambios-academicos").unlink()
    (comun / "packed-refs").write_text(f"{H2} refs/heads/cambios-academicos\n")
    assert _commit(wt) == H2[:7]


def test_sin_git_o_gitdir_roto_da_none(tmp_path):
    assert _commit(tmp_path) is None
    (tmp_path / ".git").write_text(f"gitdir: {tmp_path / 'no-existe'}\n")
    assert _commit(tmp_path) is None
    (tmp_path / ".git").write_text("basura\n")
    assert _commit(tmp_path) is None
