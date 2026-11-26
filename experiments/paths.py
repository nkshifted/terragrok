"""Shared repo paths (no torch). Resolves the vendored model checkout from git worktrees too."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
OUT_DIR = REPO_ROOT / "out"
VIEWER_DIR = REPO_ROOT / "viewer"
VIEWER_WORLDS_DIR = VIEWER_DIR / "public" / "worlds"

_VENDOR_REL = Path(".vendor") / "terrain-diffusion"


def main_checkout_root(repo_root: Path = REPO_ROOT) -> Path | None:
    """If `repo_root` is a linked git worktree, return the main checkout's root."""
    dot_git = repo_root / ".git"
    if not dot_git.is_file():
        return None
    try:
        text = dot_git.read_text().strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    git_dir = Path(text.split(":", 1)[1].strip())
    if not git_dir.is_absolute():
        git_dir = (repo_root / git_dir).resolve()
    # <main>/.git/worktrees/<name>  ->  <main>
    if git_dir.parent.name == "worktrees" and git_dir.parent.parent.name == ".git":
        return git_dir.parent.parent.parent
    return None


def vendor_terrain_diffusion(repo_root: Path = REPO_ROOT) -> Path:
    """
    Locate the terrain-diffusion checkout (gitignored, cloned by `just vendor`).

    Order: $TERRAGROK_VENDOR_DIR, this checkout's .vendor/, then the main checkout's
    .vendor/ when running from a git worktree. Returns the first candidate even if it
    does not exist so error messages name the expected location.
    """
    candidates: list[Path] = []
    env = os.environ.get("TERRAGROK_VENDOR_DIR")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(repo_root / _VENDOR_REL)
    main_root = main_checkout_root(repo_root)
    if main_root is not None:
        candidates.append(main_root / _VENDOR_REL)
    for c in candidates:
        if (c / "terrain_diffusion").is_dir():
            return c
    return candidates[0]


VENDOR_TD = vendor_terrain_diffusion()


def add_vendor_to_syspath() -> Path:
    """Put terrain-diffusion on sys.path; raise a helpful error if it was never cloned."""
    if not (VENDOR_TD / "terrain_diffusion").is_dir():
        raise RuntimeError(
            f"terrain-diffusion not found at {VENDOR_TD}. Run `just vendor` (or set TERRAGROK_VENDOR_DIR)."
        )
    p = str(VENDOR_TD)
    if p not in sys.path:
        sys.path.insert(0, p)
    return VENDOR_TD


if __name__ == "__main__":
    # `python experiments/paths.py --check` exits 0 when the model code is available
    found = (VENDOR_TD / "terrain_diffusion").is_dir()
    print(f"terrain-diffusion: {VENDOR_TD}{'' if found else '  (missing)'}")
    if "--check" in sys.argv:
        raise SystemExit(0 if found else 1)
