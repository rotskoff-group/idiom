"""Activation extraction for downstream use. Sequences or FASTA in, embeddings out, ESM extract.py style.

Reuses the SAE's extractor so exported vectors are identical to what the SAE trains on. Each
record is FIM-formatted (full context), the residual stream is taken at the requested layers,
and residues align 1:1 to their source positions (markers dropped). pool="mean" returns one
vector per sequence (mean over the IDR residues); pool="none" returns per-residue rows.
"""

from __future__ import annotations

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


@torch.no_grad()
def embed_fasta(model, inputs, layers, *, pool="mean", tokenizer=None, device="cpu", fim_mode=PROMPTED):
    """Embed sequences or FASTA records into residual-stream vectors at the requested layers.

    inputs may be a record FASTA path, a bare sequence string, or a list of sequence strings /
    Records (see idiom.data.io.to_records). A bare sequence is treated as an unprompted IDR (the
    whole sequence is the IDR). fim_mode is the prompt format the activations are taken under:
    "prompted" (1{prefix}3{suffix}2{IDR}, context) or "unprompted" (132{IDR}, de novo, no flanks).
    Legacy "idr"/"idp" accepted.

    Args:
        model: The IDiom transformer to run.
        inputs (str | Path | Record | Iterable[str | Record]): A FASTA path, sequence, or list.
        layers (list[int]): Layer indices whose residual stream to extract.
        pool (str): "mean" for one IDR-mean vector per sequence, "none" for per-residue rows.
        tokenizer (Tokenizer | None): Tokenizer (a default is used if None).
        device (str | torch.device): Device to run the model on.
        fim_mode (str): Prompt format, "prompted" or "unprompted" (legacy "idr"/"idp" accepted).

    Returns:
        dict: Mapping of layer to (values, index), where values is an [N, d] array and index is
            a list of per-row metadata dicts.
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
    """Write each layer's values as layer_<l>.npy plus a layer_<l>_index.csv.

    Args:
        embeddings (dict): Mapping of layer to (values, index), as returned by embed_fasta.
        out_dir (str | Path): Directory to create and write the .npy and .csv files into.
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
    """Command-line entry point: export residual-stream embeddings from a FASTA."""
    import argparse

    from idiom.model.io import load_pretrained
    from idiom.utils.device import resolve_device

    p = argparse.ArgumentParser(description="Export IDiom residual-stream embeddings from a FASTA.")
    p.add_argument("--fasta", required=True)
    p.add_argument("--ckpt", required=True, help="lightning .ckpt (arch read from it)")
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--pool", choices=["mean", "none"], default="mean")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = resolve_device()
    model = load_pretrained(args.ckpt, device=device)
    emb = embed_fasta(model, args.fasta, args.layers, pool=args.pool, device=device)
    write_embeddings(emb, args.out)


if __name__ == "__main__":
    main()
