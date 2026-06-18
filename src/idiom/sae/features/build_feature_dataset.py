"""Build the feature-activation dataset consumed by the viewer / annotation (P5 capstone).

Streams records through the frozen model, takes residue-only residual-stream activations at
one layer, encodes them through a trained SAE, and writes the per-residue top-k features +
the FIM strings as a directory of ``.npy`` + ``.json`` (no h5). ``pos_idx`` indexes the FIM
string (markers present), so the viewer can shade each residue by its activation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from idiom.data.fim import fim_132, fim_full
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
    fim_mode: str = "context",
) -> Path:
    """Encode records through ``sae`` at ``layer`` and write the feature dataset to ``out_dir``.

    ``region`` (``all`` | ``idr`` | ``non_idr``) and ``fim_mode`` (``context`` = ``1{prefix}3{suffix}2{IDR}``
    | ``denovo`` = ``132{IDR}``) are the SAE's training distribution; the dataset is built over exactly
    those residues, in that prompt format, so the features match what the SAE learned.
    """
    tok = tokenizer or Tokenizer()
    fim = fim_full if fim_mode == "context" else fim_132
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
             "region": region, "fim_mode": fim_mode}
        )
    )
    return out


def main() -> None:
    import argparse

    from idiom import IDiomSAE

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
