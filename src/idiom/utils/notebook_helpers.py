"""Input auditing, exports, and small-set comparisons for cookbook workflows.

Requires the cookbook extra (pandas). This module is not imported by the core API.
"""

import json
from pathlib import Path

import pandas as pd

from idiom.data.records import Record, parse_sequence_record, read_fasta

AA = "ACDEFGHIKLMNPQRSTVWY"
DEMO = [
    ("proline_1", "GSPPPGQPPPGSPQPPGSPPPGQPS"),
    ("proline_2", "QPPGSPPPGQPPGSPQPPPGSQPP"),
    ("serine_1", "RSRSRSQSSRSRSRSSQRSRSRSS"),
    ("serine_2", "SSRSRSQRSRSRSSRSQSSRSRSQ"),
    ("glycine_1", "GRGGQRGGSGRGGQRGGSGRGGQ"),
    ("glycine_2", "GQRGGSGRGGQRGGSGRGGQRGG"),
]


def load_inputs(path, mode="idr", limit=32) -> tuple[list[Record], pd.DataFrame]:
    """Load valid FASTA records and audit every input entry.

    Assign unique record_<row> IDs, preserving original accessions in the audit.
    Invalid entries are recorded and skipped rather than aborting the load.

    Args:
        path: FASTA path, or None for the built-in examples.
        mode: "idr" for isolated sequences or "annotated" for proteins with IDR spans.
            In isolated mode, any supplied span must cover the whole sequence.
        limit: Maximum accepted records, or None to retain all valid entries.

    Returns:
        A (records, audit) tuple containing Records and a pandas DataFrame with
        record_id, accession, header, idr_start_1based, idr_end_1based, and status.
        Audit spans use one-based inclusive coordinates.

    Raises:
        ValueError: If mode is unknown or limit is below one.
    """
    if mode not in {"idr", "annotated"}:
        raise ValueError("INPUT_MODE must be 'idr' or 'annotated'.")
    if limit is not None and limit < 1:
        raise ValueError("Record limit must be positive or None.")
    raw = DEMO if path is None else read_fasta(Path(path), drop_noncanonical=False)
    records, audit = [], []
    for row, (header, seq) in enumerate(raw):
        status = "accepted"
        accession, start, end = header.split()[0] if header.split() else "", 0, len(seq)
        try:
            record = parse_sequence_record(header, seq, mode=mode if path is not None else "idr")
            accession, start, end = record.accession, record.idr_start, record.idr_end
            if limit is not None and len(records) >= limit:
                status = "outside sample limit"
            else:
                records.append(Record(f"record_{row}", seq, start, end))
        except (ValueError, IndexError) as exc:
            status = str(exc)
        audit.append(
            dict(
                record_id=f"record_{row}",
                accession=accession,
                header=header,
                idr_start_1based=start + 1,
                idr_end_1based=end,
                status=status,
            )
        )
    audit = pd.DataFrame(
        audit, columns=["record_id", "accession", "header", "idr_start_1based", "idr_end_1based", "status"]
    )
    return records, audit


def idr_sequence(record) -> str:
    """Extract the annotated IDR."""
    return record.full_seq[record.idr_start : record.idr_end]


def isolated(records) -> list[Record]:
    """Remove flanks while preserving unique record IDs."""
    return [Record(r.accession, idr_sequence(r), 0, len(idr_sequence(r))) for r in records]


def check_context(records, max_length, include_flanks=False) -> None:
    """Check that each selected sequence fits the model context.

    Args:
        records: Records with valid IDR spans.
        max_length: Maximum tokens, including START and three FIM markers.
        include_flanks: Check full proteins if True, otherwise only their IDRs.

    Raises:
        ValueError: If any sequence plus four control tokens exceeds max_length.
    """
    bad = [
        r.accession
        for r in records
        if (len(r.full_seq) if include_flanks else len(idr_sequence(r))) + 4 > max_length
    ]
    if bad:
        raise ValueError(f"Inputs exceed model context ({max_length} including markers): {bad[:10]}")


def summaries(records, audit) -> pd.DataFrame:
    """Return a DataFrame of IDR composition and original record metadata.

    Args:
        records: Records with nonempty IDRs and unique accessions used as record IDs.
        audit: Input audit with matching record_id values.

    Returns:
        A DataFrame with record_id, sequence, length, charged_fraction,
        net_charge_per_residue, fraction_<AA> columns, original accession, and
        one-based inclusive idr_start_1based and idr_end_1based coordinates.
    """
    rows = []
    for r in records:
        s = idr_sequence(r)
        rows.append(
            dict(
                record_id=r.accession,
                sequence=s,
                length=len(s),
                charged_fraction=sum(s.count(a) for a in "DEKR") / len(s),
                net_charge_per_residue=(sum(s.count(a) for a in "KR") - sum(s.count(a) for a in "DE"))
                / len(s),
                **{f"fraction_{a}": s.count(a) / len(s) for a in AA},
            )
        )
    return pd.DataFrame(rows).merge(
        audit[["record_id", "accession", "idr_start_1based", "idr_end_1based"]],
        on="record_id",
        validate="one_to_one",
    )


def write_fasta(records, path) -> Path:
    """Write full records with one-based inclusive IDR spans and return the output Path.

    Overwrite an existing file. The parent directory must already exist.
    """
    Path(path).write_text(
        "".join(f">{r.accession}_IDR_{r.idr_start + 1}-{r.idr_end}\n{r.full_seq}\n" for r in records)
    )
    return Path(path)


def save_run(out, settings, *, elapsed=None) -> None:
    """Write settings, dependency versions, and optional elapsed seconds to run.json.

    Create the output directory if needed and overwrite an existing run.json.
    """
    import importlib.metadata

    out.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ("idiom", "torch", "numpy", "pandas", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "source checkout"
    payload = {"settings": settings, "versions": versions, "elapsed_seconds": elapsed}
    (out / "run.json").write_text(json.dumps(payload, indent=2, default=str))


def split_records(records, *, validation_fraction=0.2, seed=0):
    """Split by exact IDR sequence so duplicate sequences never cross splits.

    This does not separate homologous sequences. Use externally clustered splits when
    evaluating generalization beyond close relatives.
    """
    import numpy as np

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    groups = list(dict.fromkeys(idr_sequence(r) for r in records))
    if len(groups) < 2:
        raise ValueError("At least two distinct IDRs are required for a validation split")
    n = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
    chosen = set(np.random.default_rng(seed).choice(groups, size=n, replace=False))
    return (
        [r for r in records if idr_sequence(r) not in chosen],
        [r for r in records if idr_sequence(r) in chosen],
    )


def sequence_metrics(sequences) -> pd.DataFrame:
    """Summarize generated sequences, retaining empty outputs for accounting."""
    from idiom.train.grpo.reward.builtin import composition_entropy

    rows = []
    for i, s in enumerate(sequences):
        rows.append(
            dict(
                sequence_id=i,
                sequence=s,
                length=len(s),
                entropy=composition_entropy(s),
                charged_fraction=sum(s.count(a) for a in "DEKR") / max(len(s), 1),
            )
        )
    frame = pd.DataFrame(rows, columns=["sequence_id", "sequence", "length", "entropy", "charged_fraction"])
    frame["duplicate"] = frame.sequence.duplicated()
    return frame


def nearest_reference(sequences, references) -> pd.DataFrame:
    """Report closest reference by SequenceMatcher similarity, not alignment identity.

    Intended for small demonstrations; cost grows with candidates times references.
    Exact matches are reported separately. References must be nonempty.
    """
    from difflib import SequenceMatcher

    references = list(references)
    if not references:
        raise ValueError("At least one reference sequence is required")
    rows = []
    for i, sequence in enumerate(sequences):
        scores = [SequenceMatcher(None, sequence, ref, autojunk=False).ratio() for ref in references]
        best = max(range(len(scores)), key=scores.__getitem__)
        rows.append(
            dict(
                sequence_id=i, reference_row=best, similarity=scores[best], exact_match=sequence in references
            )
        )
    return pd.DataFrame(rows, columns=["sequence_id", "reference_row", "similarity", "exact_match"])


def example_file(name: str, directory, *, revision="v1") -> Path:
    """Download a cookbook FASTA from the same release used by notebook installation."""
    from urllib.request import urlretrieve

    allowed = {
        "effector/ad.fasta",
        "effector/rd.fasta",
        "protgps/nucleolus.fasta",
        "prompted_grpo/P45973.fasta",
    }
    if name not in allowed:
        raise ValueError(f"Unknown example: {name}")
    out = Path(directory) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        urlretrieve(
            f"https://raw.githubusercontent.com/rotskoff-group/idiom/{revision}/cookbook/example_data/{name}",
            out,
        )
    return out
