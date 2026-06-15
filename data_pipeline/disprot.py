"""Canonical DisProt reference parsing (single source of truth).

Ported verbatim from the v1 draft (`idr-plm-figures` `utils.utils`:
`load_disprot_json` / `extract_disprot_sequences_and_idrs` / `filter_disprot_idrs` /
`classify_idr_type_json`) so the v2 dedup query and the SI figures use the *same* DisProt IDR
set the original paper did. The DisProt IDR benchmark = experimentally-annotated 'D' (disordered)
consensus regions, length >= 30, on proteins <= the corpus length cap, with full-length IDPs
removed (those belong to the separate "IDP" benchmark).

Only deliberate v2 change vs v1: `max_seq_length` default **512 -> 1020**, matching the 1020
protein-length cap of the filtered training corpus (`filter_length_plddt`). Full-IDP removal keeps
v1's *fuzzy* +/-1 rule (start<=2 and end>=len-1), not a strict whole-protein equality.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_DISPROT_JSON = (
    "/data2/scratch/group_scratch/idr_plm/2026-06-14_idiom_data/reference/disprot/"
    "DisProt release_2025_06 with_ambiguous_evidences.json"
)


def load_disprot_json(json_path: str) -> list:
    """Return the list of DisProt entries (top-level list, or first list-valued key). [v1 parity]"""
    data = json.loads(Path(json_path).read_text())
    if isinstance(data, list):
        return data
    for key in data:
        if isinstance(data[key], list):
            return data[key]
    raise ValueError("No list of entries found at the top level.")


def _extract(entries: list) -> list[tuple]:
    """All 'D' (disordered) consensus regions as (acc, idx, start, end, idr_seq, full_seq).

    `start`/`end` are DisProt's 1-based inclusive coords; `idr_seq = seq[start-1:end]`. [v1 parity]
    """
    out: list[tuple] = []
    for entry in entries:
        acc, seq = entry.get("acc"), entry.get("sequence")
        states = entry.get("disprot_consensus", {}).get("Structural state", [])
        if not (acc and seq and isinstance(states, list)):
            continue
        for idx, region in enumerate(states):
            s, e = region.get("start"), region.get("end")
            if region.get("type") == "D" and isinstance(s, int) and isinstance(e, int):
                out.append((acc, idx, s, e, seq[s - 1 : e], seq))
    return out


def _is_full_idp(item: tuple) -> bool:
    """v1 `classify_idr_type_json == 'Full IDP'`: IDR spans the whole protein (+/-1 leeway)."""
    _, _, start, end, _, full = item
    return start <= 2 and end >= len(full) - 1


def disprot_idr_records(
    json_path: str = DEFAULT_DISPROT_JSON,
    *,
    min_idr_length: int = 30,
    max_seq_length: int = 1020,
) -> list[tuple]:
    """The benchmark DisProt IDR set: 'D' regions, idr >= min, full seq <= max, non-full-IDP.

    Returns (acc, idx, start, end, idr_seq, full_seq) tuples. Mirrors v1's
    `filter_disprot_idrs(exclude_full_idps=False)` + `classify_idr_type_json != 'Full IDP'`.
    """
    records = _extract(load_disprot_json(json_path))
    records = [r for r in records if len(r[4]) >= min_idr_length and len(r[5]) <= max_seq_length]
    return [r for r in records if not _is_full_idp(r)]


def disprot_idr_seqs(json_path: str = DEFAULT_DISPROT_JSON, **kwargs) -> list[str]:
    """Just the IDR substrings (for composition / length figures)."""
    return [r[4] for r in disprot_idr_records(json_path, **kwargs)]
