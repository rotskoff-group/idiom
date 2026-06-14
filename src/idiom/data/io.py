"""FASTA / record I/O for IDiom (v2).

The training record store and the user-facing inference inputs share **one** FASTA
convention: each entry is a full protein whose header ends ``_IDR_{x}-{y}`` (1-indexed,
inclusive) marking the IDR span — the same format as the public ``generate idr`` input. A
:class:`Record` is the parsed, 0-indexed form (Option A: ``full_seq`` + coords).

This module is also the **single place** the non-canonical drop policy (D15) is enforced:
any sequence with a residue outside the 20 canonical AAs is dropped here, with a logged count,
so curation, training, and inference all behave identically.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from loguru import logger as log

from idiom.data.tokenizer import Tokenizer

_TOK = Tokenizer()


@dataclass(frozen=True)
class Record:
    """One IDR instance with its parent sequence.

    Coords are 0-indexed, **half-open**: ``idr = full_seq[idr_start:idr_end]`` (the header's
    1-based inclusive ``_IDR_x-y`` is converted to this Python-native form on read).
    """

    accession: str
    full_seq: str
    idr_start: int
    idr_end: int


def _iter_fasta_raw(path: str | Path) -> Iterator[tuple[str, str]]:
    header: str | None = None
    parts: list[str] = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)  # emit the record we just finished
                header, parts = line[1:], []
            else:
                parts.append(line)  # sequence may span multiple wrapped lines
        if header is not None:
            yield header, "".join(parts)  # emit the last record (no header follows it)


def read_fasta(path: str | Path, *, drop_noncanonical: bool = True) -> list[tuple[str, str]]:
    """Read ``(header, sequence)`` pairs, dropping non-canonical sequences (D15)."""
    out: list[tuple[str, str]] = []
    dropped = 0
    for header, seq in _iter_fasta_raw(path):
        if drop_noncanonical and not _TOK.is_canonical(seq):
            dropped += 1
            continue
        out.append((header, seq))
    if dropped:
        log.info(f"{Path(path).name}: dropped {dropped} non-canonical sequence(s)")
    return out


def parse_idr_header(header: str) -> tuple[str, int, int]:
    """``{accession}_IDR_{x}-{y}`` (1-indexed inclusive) -> ``(accession, start, end)``.

    Returns 0-indexed **half-open** coords: ``idr = full_seq[start:end]``.
    """
    token = header.split()[0]  # ignore any free-text description after whitespace
    if "_IDR_" not in token:
        raise ValueError(f"header missing '_IDR_x-y' span: {header!r}")
    accession, span = token.rsplit("_IDR_", 1)
    try:
        x_str, y_str = span.split("-")
        x, y = int(x_str), int(y_str)
    except ValueError:
        raise ValueError(f"cannot parse IDR span from {span!r} in header {header!r}") from None
    return accession, x - 1, y  # 1-indexed inclusive -> 0-indexed half-open [start, end)


def read_records(path: str | Path, *, drop_noncanonical: bool = True) -> Iterator[Record]:
    """Parse a record-store / inference FASTA into :class:`Record`s.

    Drops non-canonical sequences (D15) and skips entries with a missing/malformed or
    out-of-range ``_IDR_x-y`` span (logged).
    """
    skipped = 0
    for header, seq in read_fasta(path, drop_noncanonical=drop_noncanonical):
        try:
            accession, start, end = parse_idr_header(header)
        except ValueError:
            skipped += 1
            continue
        if not 0 <= start < end <= len(seq):  # empty or out-of-bounds span
            skipped += 1
            continue
        yield Record(accession, seq, start, end)
    if skipped:
        log.warning(f"{Path(path).name}: skipped {skipped} malformed/out-of-range record(s)")
