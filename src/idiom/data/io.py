"""FASTA I/O with IDR spans.

Headers use 1-based inclusive "_IDR_x-y" spans; Records use 0-based half-open
coordinates. Readers drop non-canonical sequences by default.
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

    Attributes:
        accession: The record's accession, taken from the FASTA header.
        full_seq: The full protein sequence.
        idr_start: IDR start index (0-based, inclusive).
        idr_end: IDR end index (0-based, exclusive), so idr = full_seq[idr_start:idr_end].
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
    """Read (header, sequence) pairs from a FASTA.

    Sequences wrapped across multiple lines are joined.

    Args:
        path: Path to the FASTA file.
        drop_noncanonical: If True, drop any sequence with a residue outside the 20 canonical amino
            acids and log the number dropped.

    Returns:
        The kept (header, sequence) pairs.
    """
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
    """Parse an "_IDR_x-y" header into an accession and 0-indexed half-open IDR coordinates.

    Free text after the first whitespace is ignored, and the split is on the last "_IDR_", so an
    accession may itself contain underscores.

    Args:
        header: The FASTA header, whose first token ends in "_IDR_x-y".

    Returns:
        The accession, the 0-based start, and the exclusive end.

    Raises:
        ValueError: If the header has no "_IDR_x-y" span or the span cannot be parsed.
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
    """Parse a FASTA into Records.

    Entries with a missing, malformed, or out-of-range "_IDR_x-y" span are skipped.

    Args:
        path: Path to the FASTA file.
        drop_noncanonical: If True, drop any sequence with a residue outside the 20 canonical amino
            acids.

    Yields:
        One record per valid entry.
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


def _sequence_record(seq: str, index: int) -> Record:
    """Wrap a bare sequence as a Record whose IDR span covers the whole sequence.

    Args:
        seq: A protein/IDR sequence of canonical amino acids.
        index: Position in the input, used to synthesize the accession "seq_{index}".

    Returns:
        A record with accession "seq_{index}" spanning the whole sequence.

    Raises:
        ValueError: If seq is empty or contains a non-canonical residue.
    """
    if not _TOK.is_canonical(seq):
        raise ValueError(
            f"sequence at index {index} is not canonical (only the 20 amino acids are allowed): "
            f"{seq[:30]!r}"
        )
    return Record(f"seq_{index}", seq, 0, len(seq))


def to_records(inputs, *, drop_noncanonical: bool = True) -> Iterator[Record]:
    """Normalize flexible sequence inputs into Records.

    Accepts, in order of precedence:

    - a single Record, passed through unchanged;
    - a str or Path naming an existing file, parsed with read_records;
    - a str that does not name an existing file, treated as one bare sequence;
    - an iterable of Records and/or bare sequence strings.

    A bare sequence becomes a Record spanning the whole sequence, with a synthetic accession
    "seq_0", "seq_1", and so on.

    Args:
        inputs (str | Path | Record | Iterable[str | Record]): The inputs to normalize.
        drop_noncanonical: Passed through to read_records for the FASTA-path case.

    Yields:
        One record per input sequence or FASTA entry.

    Raises:
        ValueError: If a Path does not exist, or a bare sequence is non-canonical.
        TypeError: If an iterable contains something other than a str or Record.
    """
    if isinstance(inputs, Record):
        yield inputs
        return
    if isinstance(inputs, (str, Path)):
        p = Path(inputs)
        if p.exists():
            yield from read_records(p, drop_noncanonical=drop_noncanonical)
        elif isinstance(inputs, Path):
            raise ValueError(f"path does not exist: {inputs}")
        else:
            yield _sequence_record(inputs, 0)  # a bare sequence string, not a file path
        return
    for i, item in enumerate(inputs):
        if isinstance(item, Record):
            yield item
        elif isinstance(item, str):
            yield _sequence_record(item, i)
        else:
            raise TypeError(f"to_records: expected str or Record, got {type(item).__name__}")
