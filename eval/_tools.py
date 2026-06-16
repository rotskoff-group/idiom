"""Resolution of external eval tools (not pip deps — licensed/heavy/binary downloads).

The permanent tools dir; override the root with ``$IDIOM_TOOLS``. Individual tools also honor their
own env var (e.g. ``$IUPRED3_PATH``, ``$MMSEQS``, ``$DEEPLOC``) so they can live elsewhere.
"""

from __future__ import annotations

import os

TOOLS_DIR = os.environ.get("IDIOM_TOOLS", "/data2/scratch/group_scratch/idr_plm/0000_dump/tools")


def tool_path(*parts: str) -> str:
    return os.path.join(TOOLS_DIR, *parts)
