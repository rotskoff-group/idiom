"""Write per-residue SAE features with positions in the FIM string, including markers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from idiom.data.fim import PROMPTED, fim_prompted, fim_unprompted, normalize_mode
from idiom.data.tokenizer import Tokenizer
from idiom.model.activations import extract_activations


@torch.no_grad()
def build_feature_dataset(
    model,
    sae,
    records,
    layer: int,
    out_dir: str | Path,
    *,
    tokenizer: Tokenizer | None = None,
    device: str | torch.device = "cpu",
    batch_size: int = 16,
    region: str = "all",
    fim_mode: str = "prompted",
) -> Path:
    """Write per-residue top-k features; see FeatureDataset for the file schema.

    Move both models to device in eval mode. Positions index FIM strings without START.

    Args:
        model: The host transformer.
        sae: The trained SparseCoder to encode activations with.
        records: Iterable of records to encode.
        layer: Zero-based block index to extract.
        out_dir: Directory to write the dataset into; created if needed.
        tokenizer: Tokenizer for encoding and region masking; a default if None.
        device: Device to run extraction and encoding on.
        batch_size: Number of records per forward batch.
        region: Residues to keep: "all", "idr", or "non_idr".
        fim_mode: Prompt format: "prompted" or "unprompted".

    Returns:
        The output directory.

    Raises:
        ValueError: If fim_mode is neither "prompted" nor "unprompted".
    """
    tok = tokenizer or Tokenizer()
    fim = fim_prompted if normalize_mode(fim_mode) == PROMPTED else fim_unprompted
    model = model.eval().to(device)
    sae = sae.eval().to(device)
    records = list(records)

    top_idx_parts, top_val_parts, seq_parts, pos_parts, strings = [], [], [], [], []

    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        seqs = [fim(r.full_seq, r.idr_start, r.idr_end) for r in chunk]
        token_lists = [torch.tensor([tok.start_id, *tok.encode(s)]) for s in seqs]
        tokens = pad_sequence(token_lists, batch_first=True, padding_value=tok.pad_id).to(device)

        acts = extract_activations(
            model, tokens, [layer], tokenizer=tok, drop_markers=True, region=region
        )[layer]
        top_val, top_ix, _ = sae.encode(acts.values.to(device))  # [N_res, k]

        top_idx_parts.append(top_ix.cpu().to(torch.int32).numpy())
        top_val_parts.append(top_val.cpu().to(torch.float32).numpy())
        seq_parts.append(acts.seq_idx.cpu().numpy().astype(np.int32) + start)  # batch-local -> global
        pos_parts.append(acts.pos_idx.cpu().numpy().astype(np.int32) - 1)  # fed pos -> FIM-string pos
        strings.extend(seqs)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "top_indices.npy", np.concatenate(top_idx_parts))
    np.save(out / "top_values.npy", np.concatenate(top_val_parts))
    np.save(out / "seq_idx.npy", np.concatenate(seq_parts))
    np.save(out / "pos_idx.npy", np.concatenate(pos_parts))
    (out / "strings.json").write_text(json.dumps(strings))
    (out / "meta.json").write_text(
        json.dumps(
            {"k": int(sae.k), "num_latents": int(sae.num_latents), "layer": int(layer),
             "region": region, "fim_mode": normalize_mode(fim_mode)}
        )
    )
    return out


def main() -> None:
    """Run the idiom_feature_dataset CLI: build a feature dataset from a FASTA and an SAE."""
    from idiom import IDiomSAE  # deferred: idiom/__init__ imports this module's package

    p = argparse.ArgumentParser(description="Build an SAE feature-activation dataset from a FASTA.")
    p.add_argument("--sae", required=True, help="trained SAE release dir (host model + layer read from it)")
    p.add_argument("--fasta", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    # the SAE release carries its host model + layer, so nothing else need be specified
    IDiomSAE.from_pretrained(args.sae).build_feature_dataset(
        args.fasta, args.out, batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
