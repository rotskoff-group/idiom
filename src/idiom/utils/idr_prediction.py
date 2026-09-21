"""Predict IDRs in full proteins and export FASTAs for IDiom cookbook workflows."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from idiom.data.records import Record, read_fasta
from idiom.data.tokenizer import RESIDUE_SET
from idiom.utils.notebook_helpers import write_fasta


def predict_idrs_fasta(fasta, out_dir, *, device="cpu", minimum_idr_length=12):
    """Run metapredict V3 and export isolated IDRs, annotated proteins, scores, and audits.

    Full proteins are read regardless of any existing span in their headers. Invalid
    entries are skipped without modifying their sequences. Repeated headers get distinct
    record IDs. Return (regions, audit, scores), where scores maps record IDs to arrays.
    Boundary settings other than minimum length use metapredict's defaults.
    """
    import metapredict

    if (
        isinstance(minimum_idr_length, bool)
        or not isinstance(minimum_idr_length, int)
        or minimum_idr_length < 1
    ):
        raise ValueError("minimum_idr_length must be a positive integer")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new or empty output directory")
    entries, audit_rows = [], []
    for i, (header, seq) in enumerate(read_fasta(fasta, drop_noncanonical=False)):
        record_id = f"protein_{i}"
        valid = bool(header.strip() and seq and not set(seq) - RESIDUE_SET)
        audit_rows.append(
            dict(
                record_id=record_id,
                header=header,
                length=len(seq),
                status="accepted" if valid else "empty header/sequence or noncanonical residues",
            )
        )
        if valid:
            entries.append((i, record_id, header, seq))
    predictions = (
        metapredict.predict_disorder_batch(
            [entry[3] for entry in entries],
            version="V3",
            device=device,
            return_domains=True,
            minimum_IDR_size=minimum_idr_length,
            show_progress_bar=False,
        )
        if entries
        else []
    )
    if len(predictions) != len(entries):
        raise ValueError("metapredict returned the wrong number of predictions")
    rows, annotated, isolated, scores = [], [], [], {}
    for (i, record_id, header, seq), prediction in zip(entries, predictions):
        values = np.asarray(prediction.disorder)
        if prediction.sequence != seq or values.shape != (len(seq),) or not np.isfinite(values).all():
            raise ValueError("metapredict returned invalid or misaligned scores")
        scores[record_id] = values
        boundaries = prediction.disordered_domain_boundaries
        audit_rows[i]["status"] = "predicted IDRs" if boundaries else "no predicted IDRs"
        for j, (start, end) in enumerate(boundaries):
            start, end = int(start), int(end)
            if not 0 <= start < end <= len(seq):
                raise ValueError("metapredict returned an out-of-range IDR")
            region_id = f"{record_id}_region_{j}"
            idr = seq[start:end]
            annotated.append(Record(region_id, seq, start, end))
            isolated.append((region_id, idr))
            rows.append(
                dict(
                    region_id=region_id,
                    record_id=record_id,
                    header=header,
                    start_1based=start + 1,
                    end_1based=end,
                    length=end - start,
                )
            )
    regions = pd.DataFrame(
        rows, columns=["region_id", "record_id", "header", "start_1based", "end_1based", "length"]
    )
    audit = pd.DataFrame(audit_rows, columns=["record_id", "header", "length", "status"])
    out.mkdir(parents=True, exist_ok=True)
    write_fasta(annotated, out / "annotated_proteins.fasta")
    (out / "idrs.fasta").write_text("".join(f">{name}\n{seq}\n" for name, seq in isolated))
    regions.to_csv(out / "idr_regions.csv", index=False)
    audit.to_csv(out / "input_audit.csv", index=False)
    np.savez_compressed(out / "disorder_scores.npz", **scores)
    settings = dict(
        metapredict_version=version("metapredict"),
        network="V3",
        device=str(device),
        minimum_IDR_size=minimum_idr_length,
        disorder_threshold=None,
        minimum_folded_domain=50,
        gap_closure=10,
        input=str(Path(fasta).resolve()),
    )
    (out / "prediction_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    return regions, audit, scores
