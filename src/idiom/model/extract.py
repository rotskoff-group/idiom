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
from idiom.data.io import to_records
from idiom.data.tokenizer import Tokenizer
from idiom.model.activations import extract_activations
from idiom.model.io import load_pretrained
from idiom.utils.device import resolve_device


@torch.no_grad()
def embed_fasta(model, inputs, layers, *, pool="mean", tokenizer=None, device="cpu", fim_mode=PROMPTED):
    """Embed sequences or FASTA records into residual-stream vectors at the requested layers.

    Records are processed one at a time, so rows appear in input order.

    Args:
        model: The transformer to run.
        inputs (str | Path | Record | Iterable[str | Record]): A record FASTA path, a bare sequence
            string, or an iterable of sequences and/or Records; see idiom.data.io.to_records.
        layers (list[int]): Zero-based block indices.
        pool (str): "mean" for one vector per sequence, averaged over its IDR residues, or "none"
            for one row per residue.
        tokenizer (Tokenizer | None): Defaults to Tokenizer().
        device (str | torch.device): Device to run the model on.
        fim_mode (str): Prompt format the activations are taken under, "prompted" or "unprompted".

    Returns:
        dict[int, tuple]: Per layer, a (values, index) pair. values is an [N, d_model] array. For
            pool="mean", index holds one dict per sequence with keys accession and n_idr; for
            pool="none", one dict per residue with keys accession, source_pos, residue, and is_idr.

    Raises:
        ValueError: If fim_mode is neither "prompted" nor "unprompted", or an input sequence is
            non-canonical.
    """
    tok = tokenizer or Tokenizer()
    variant = normalize_mode(fim_mode)
    build = fim_prompted if variant == PROMPTED else fim_unprompted
    out = {layer: {"values": [], "index": []} for layer in layers}

    for rec in to_records(inputs):
        fim = build(rec.full_seq, rec.idr_start, rec.idr_end)
        tokens = torch.tensor([tok.start_id, *tok.encode(fim)], device=device)[None]  # [1, L]
        acts = extract_activations(model, tokens, layers, tokenizer=tok, drop_markers=True)
        # extracted residue rows are in FIM order (markers dropped) — same order as src.
        src = residue_source_positions(len(rec.full_seq), rec.idr_start, rec.idr_end, variant)
        is_idr = np.array([rec.idr_start <= p < rec.idr_end for p in src])

        for layer in layers:
            vals = acts[layer].values.cpu()  # [n_res, d]
            if pool == "mean":
                out[layer]["values"].append(vals[is_idr].mean(0))  # per-sequence IDR embedding
                out[layer]["index"].append({"accession": rec.accession, "n_idr": int(is_idr.sum())})
            else:  # per-residue
                ids = acts[layer].token_id
                for i in range(vals.size(0)):
                    out[layer]["values"].append(vals[i])
                    out[layer]["index"].append({
                        "accession": rec.accession,
                        "source_pos": int(src[i]),
                        "residue": tok.decode([int(ids[i])]),
                        "is_idr": bool(is_idr[i]),
                    })

    return {layer: (torch.stack(d["values"]).numpy(), d["index"]) for layer, d in out.items()}


def write_embeddings(embeddings: dict, out_dir: str | Path) -> None:
    """Write each layer's embeddings as "layer_<l>.npy" and its metadata as "layer_<l>_index.csv".

    Args:
        embeddings: Mapping of layer to (values, index), as returned by embed_fasta.
        out_dir: Directory to create and write into.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for layer, (values, index) in embeddings.items():
        np.save(out / f"layer_{layer}.npy", values)
        with (out / f"layer_{layer}_index.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(index[0].keys()))
            writer.writeheader()
            writer.writerows(index)


def main() -> None:
    """Run the idiom_extract CLI: write residual-stream embeddings from a FASTA to a directory."""
    p = argparse.ArgumentParser(description="Export IDiom residual-stream embeddings from a FASTA.")
    p.add_argument("--fasta", required=True)
    p.add_argument("--ckpt", required=True, help="lightning .ckpt (arch read from it)")
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--pool", choices=["mean", "none"], default="mean")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = resolve_device()
    model, _ = load_pretrained(args.ckpt, device=device)
    emb = embed_fasta(model, args.fasta, args.layers, pool=args.pool, device=device)
    write_embeddings(emb, args.out)


if __name__ == "__main__":
    main()
