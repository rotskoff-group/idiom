"""Resolution of external eval tools (not pip deps — licensed/heavy/binary downloads).

The permanent tools dir; override the root with ``$IDIOM_TOOLS``. Individual tools also honor their
own env var (e.g. ``$IUPRED3_PATH``, ``$MMSEQS``, ``$DEEPLOC``) so they can live elsewhere.
"""

from __future__ import annotations

import os

TOOLS_DIR = os.environ.get("IDIOM_TOOLS", "/data2/scratch/group_scratch/idr_plm/0000_dump/tools")


def tool_path(*parts: str) -> str:
    return os.path.join(TOOLS_DIR, *parts)


def torch_cache() -> str:
    """Shared torch.hub cache (ESM1b etc. pre-downloaded) — set ``$TORCH_HOME`` to this offline."""
    return os.environ.get("IDIOM_TORCH_CACHE") or tool_path("torch_cache")


def deeploc_venv() -> str:
    """The ``deeploc2`` entrypoint inside its dedicated venv (override via ``$DEEPLOC``)."""
    return os.environ.get("DEEPLOC") or tool_path("deeploc-venv", "bin", "deeploc2")


def catgranule_dir() -> str:
    """The cloned catGRANULE 2.0 repo (holds ``src/`` models; override via ``$CATGRANULE``)."""
    return os.environ.get("CATGRANULE") or tool_path("catgranule2")


def catgranule_python() -> str:
    """Python of the catGRANULE venv (pinned sklearn 1.1.1; override via ``$CATGRANULE_VENV``)."""
    venv = os.environ.get("CATGRANULE_VENV") or tool_path("catgranule-venv")
    return os.path.join(venv, "bin", "python")
