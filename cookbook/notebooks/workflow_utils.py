"""Input validation and exports shared by the interactive cookbook workflows."""

import json
from pathlib import Path

import pandas as pd

from idiom.data.io import Record, parse_idr_header, read_fasta

AA = "ACDEFGHIKLMNPQRSTVWY"
DEMO = [
    ("proline_1", "GSPPPGQPPPGSPQPPGSPPPGQPS"),
    ("proline_2", "QPPGSPPPGQPPGSPQPPPGSQPP"),
    ("serine_1", "RSRSRSQSSRSRSRSSQRSRSRSS"),
    ("serine_2", "SSRSRSQRSRSRSSRSQSSRSRSQ"),
    ("glycine_1", "GRGGQRGGSGRGGQRGGSGRGGQ"),
    ("glycine_2", "GQRGGSGRGGQRGGSGRGGQRGG"),
]


def load_inputs(path, mode="idr", limit=32):
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
            if not accession or not seq or set(seq) - set(AA):
                raise ValueError("empty header/sequence or noncanonical residues")
            if mode == "annotated" and path is not None:
                accession, start, end = parse_idr_header(header)
            elif "_IDR_" in accession:
                accession, start, end = parse_idr_header(header)
                if (start, end) != (0, len(seq)):
                    raise ValueError("partial IDR span: use annotated mode for full proteins")
            if not 0 <= start < end <= len(seq):
                raise ValueError("IDR coordinates outside sequence")
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


def idr_sequence(record):
    """Extract the annotated IDR."""
    return record.full_seq[record.idr_start : record.idr_end]


def isolated(records):
    """Remove flanks while preserving unique record IDs."""
    return [Record(r.accession, idr_sequence(r), 0, len(idr_sequence(r))) for r in records]


def check_context(records, max_length, include_flanks=False):
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


def summaries(records, audit):
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


def write_fasta(records, path):
    """Write full records with one-based inclusive IDR spans and return the output Path.

    Overwrite an existing file. The parent directory must already exist.
    """
    Path(path).write_text(
        "".join(f">{r.accession}_IDR_{r.idr_start + 1}-{r.idr_end}\n{r.full_seq}\n" for r in records)
    )
    return Path(path)


def save_run(out, settings, *, elapsed=None):
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
