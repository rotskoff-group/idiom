"""Sequence and FASTA embeddings with source-residue alignment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from idiom.data.fim import (
    PROMPTED,
    fim_prompted,
    fim_unprompted,
    normalize_mode,
    residue_source_positions,
)
from idiom.data.records import to_records
from idiom.data.tokenizer import Tokenizer
from idiom.model.activations import extract_activations
from idiom.model.io import load_model
from idiom.utils.device import resolve_device


@torch.no_grad()
def extract_embeddings(
    model,
    inputs,
    layers,
    *,
    pool="mean",
    tokenizer=None,
    device="cpu",
    fim_mode=PROMPTED,
):
    """Extract residual-stream embeddings in input-record order.

    FASTA entries with invalid sequences or nonempty malformed headers are skipped;
    noncanonical bare sequences raise ValueError. Record objects pass through input
    normalization unchanged and must contain valid sequences and spans.

    Args:
        model: Transformer on the same device as the input tokens.
        inputs: FASTA path, Record, bare sequence, or iterable of Records and sequences.
            Each supplied record must have a nonempty, valid IDR span.
        layers: Zero-based transformer block indices.
        pool: "mean" averages IDR residues; "none" returns each IDR residue;
            "last" returns the final IDR residue, excluding EOS and markers.
        tokenizer: Tokenizer to use, or None for the default vocabulary.
        device: Device for input tokens; the model is not moved.
        fim_mode: "prompted" includes flanks; "unprompted" encodes only the IDR.

    Returns:
        A dictionary mapping each layer to (values, index). values is a NumPy array
        with shape [N_records, d_model] for mean/last pooling or [N_residues, d_model]
        otherwise. index contains one metadata dictionary per row.
        Pooled metadata contains accession, n_idr, n_residues, and record_idx. Per-residue metadata
        contains record_idx, accession, source_pos, residue, and is_idr. Residue rows
        follow original IDR sequence order, excluding markers and START;
        source_pos is the zero-based original protein position.

    Raises:
        ValueError: If fim_mode is invalid, a bare sequence is noncanonical, or a
            supplied Path does not exist, an option is invalid, or an IDR span is empty or invalid.
        IndexError: If a FASTA entry reaching span parsing has an empty header.
    """
    if pool not in ("mean", "none", "last"):
        raise ValueError(f"invalid pool: {pool!r}")
    tok = tokenizer or Tokenizer()
    variant = normalize_mode(fim_mode)
    build = fim_prompted if variant == PROMPTED else fim_unprompted
    out = {layer: {"values": [], "index": []} for layer in layers}

    for record_idx, rec in enumerate(to_records(inputs)):
        if not 0 <= rec.idr_start < rec.idr_end <= len(rec.full_seq):
            raise ValueError(f"record {record_idx} ({rec.accession!r}) has an empty or invalid IDR span")
        fim = build(rec.full_seq, rec.idr_start, rec.idr_end)
        tokens = torch.tensor([tok.start_id, *tok.encode(fim)], device=device)[None]
        acts = extract_activations(model, tokens, layers, tokenizer=tok, drop_markers=True)
        # extracted residue rows are in FIM order (markers dropped) — same order as src
        src = residue_source_positions(len(rec.full_seq), rec.idr_start, rec.idr_end, variant)
        is_idr = np.array([rec.idr_start <= p < rec.idr_end for p in src])

        selected = np.flatnonzero(is_idr)

        for layer in layers:
            vals = acts[layer].values.cpu()
            if pool != "none":
                value = vals[selected].mean(0) if pool == "mean" else vals[selected[-1]]
                out[layer]["values"].append(value)
                out[layer]["index"].append(
                    {
                        "accession": rec.accession,
                        "n_idr": int(is_idr.sum()),
                        "n_residues": len(selected),
                        "record_idx": record_idx,
                    }
                )
            else:
                ids = acts[layer].token_id
                for i in selected:
                    out[layer]["values"].append(vals[i])
                    out[layer]["index"].append(
                        {
                            "record_idx": record_idx,
                            "accession": rec.accession,
                            "source_pos": int(src[i]),
                            "residue": tok.decode([int(ids[i])]),
                            "is_idr": bool(is_idr[i]),
                        }
                    )

    return {
        layer: (
            torch.stack(d["values"]).numpy()
            if d["values"]
            else np.empty((0, model.cfg.d_model), dtype=np.float32),
            d["index"],
        )
        for layer, d in out.items()
    }


def write_embeddings(embeddings: dict, out_dir: str | Path) -> None:
    """Write each layer's embeddings as "layer_<l>.npy" and its metadata as "layer_<l>_index.csv".

    Args:
        embeddings: Mapping of layer to (values, index), as returned by extract_embeddings.
        out_dir: Directory to create and write into.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for layer, (values, index) in embeddings.items():
        np.save(out / f"layer_{layer}.npy", values)
        with (out / f"layer_{layer}_index.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(index[0].keys()) if index else [])
            writer.writeheader()
            writer.writerows(index)


def main(argv: list[str] | None = None) -> None:
    """Run the idiom_extract CLI: write residual-stream embeddings from a FASTA to a directory."""
    p = argparse.ArgumentParser(description="Export IDiom residual-stream embeddings from a FASTA.")
    p.add_argument("--fasta", required=True)
    p.add_argument(
        "--model",
        "--ckpt",
        dest="model",
        required=True,
        help="Hub model ID, released directory, or Lightning .ckpt (--ckpt is an alias)",
    )
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--pool", choices=["mean", "none", "last"], default="mean")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    device = resolve_device()
    model, _ = load_model(args.model, device=device)
    emb = extract_embeddings(model, args.fasta, args.layers, pool=args.pool, device=device)
    write_embeddings(emb, args.out)


if __name__ == "__main__":
    main()
