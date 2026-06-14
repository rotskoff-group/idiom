"""Public API: the ``IDiom`` model wrapper (FASTA-first, HF-friendly).

A thin layer over :class:`idiom.model.IDiomTransformer` + :class:`idiom.data.Tokenizer` that
hides training/Hydra/Lightning. Load a released model with :meth:`from_pretrained` (HF repo id
or local dir), generate IDPs/IDRs to strings or FASTA, or pull embeddings.

    from idiom import IDiom
    model = IDiom.from_pretrained("jxliu2/idiom-medium")
    idrs  = model.generate_idp(n=100)                       # de-novo IDPs
    idrs  = model.generate_idr(protein_seq, start, end, n=100)   # context-prompted (0-based, half-open)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from idiom.data.fim import fim_prompt
from idiom.data.io import read_records
from idiom.data.tokenizer import Tokenizer
from idiom.model.config import ModelConfig
from idiom.model.export import embed_fasta
from idiom.model.sampling import generate
from idiom.model.transformer import IDiomTransformer
from idiom.utils.device import resolve_device

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"


def _resolve(name_or_path: str | Path) -> Path:
    """Local dir → itself; otherwise treat as an HF repo id and download a snapshot."""
    p = Path(name_or_path)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    return Path(snapshot_download(str(name_or_path)))


class IDiom:
    def __init__(self, model: IDiomTransformer, tokenizer: Tokenizer | None = None, device="cpu"):
        self.model = model.eval()
        self.tok = tokenizer or Tokenizer()
        self.device = torch.device(device)
        self.model.to(self.device)

    # --- load / save (HF-style) ---
    @classmethod
    def from_pretrained(cls, name_or_path: str | Path, *, device="auto") -> "IDiom":
        from safetensors.torch import load_model  # noqa: PLC0415

        d = _resolve(name_or_path)
        cfg = ModelConfig(**json.loads((d / CONFIG_FILE).read_text()))
        model = IDiomTransformer(cfg)
        load_model(model, str(d / WEIGHTS_FILE))  # handles the tied embedding
        return cls(model, device=resolve_device(device))

    def save_pretrained(self, out_dir: str | Path) -> Path:
        from safetensors.torch import save_model  # noqa: PLC0415

        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / CONFIG_FILE).write_text(json.dumps(asdict(self.model.cfg), indent=2))
        save_model(self.model, str(d / WEIGHTS_FILE))
        return d

    @classmethod
    def from_lightning_checkpoint(cls, ckpt_path, cfg: ModelConfig, *, device="auto") -> "IDiom":
        """Wrap a training ``.ckpt`` (e.g. to then ``save_pretrained`` a release)."""
        from idiom.model.io import load_pretrained  # noqa: PLC0415

        dev = resolve_device(device)
        return cls(load_pretrained(ckpt_path, cfg, device=dev), device=dev)

    # --- generation ---
    def _decode_idr(self, row: torch.Tensor) -> str:
        ids: list[int] = []
        for i in row.tolist():
            if i in (self.tok.stop_id, self.tok.pad_id):
                break  # IDR ends at the first STOP/PAD
            if self.tok.is_residue(i):  # keep residues only; drop any stray FIM markers
                ids.append(i)
        return self.tok.decode(ids)

    @torch.no_grad()
    def _generate(self, prompt: str, n: int, **kw) -> list[str]:
        gen = torch.Generator(device=self.device).manual_seed(kw.pop("seed")) if "seed" in kw else None
        prompts = torch.tensor(self.tok.encode(prompt), device=self.device).unsqueeze(0).repeat(n, 1)
        out = generate(self.model, prompts, tokenizer=self.tok, generator=gen, **kw)
        return [self._decode_idr(row) for row in out]

    def generate_idp(self, n: int = 100, *, max_new_tokens: int = 256, temperature: float = 1.0,
                     top_k: int | None = None, top_p: float | None = None, seed: int | None = None) -> list[str]:
        """De-novo IDPs (prompt ``132``). Returns ``n`` IDR residue strings."""
        kw = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
        if seed is not None:
            kw["seed"] = seed
        return self._generate(fim_prompt(), n, **kw)

    def generate_idr(self, seq: str, idr_start: int, idr_end: int, n: int = 100, **kw) -> list[str]:
        """IDRs conditioned on flanks. ``idr_start/idr_end`` are 0-based, half-open (``seq[start:end]``)."""
        return self._generate(fim_prompt(seq, idr_start, idr_end), n, **kw)

    # --- FASTA-first wrappers ---
    def generate_idp_fasta(self, out_fasta, n: int = 100, *, prefix: str = "idiom_idp", **kw) -> Path:
        seqs = self.generate_idp(n, **kw)
        return _write_fasta([(f"{prefix}_{i}", s) for i, s in enumerate(seqs)], out_fasta)

    def generate_idr_fasta(self, in_fasta, out_fasta, n: int = 100, **kw) -> Path:
        rows = []
        for r in read_records(in_fasta):
            for i, s in enumerate(self.generate_idr(r.full_seq, r.idr_start, r.idr_end, n, **kw)):
                rows.append((f"{r.accession}_gen{i}", s))
        return _write_fasta(rows, out_fasta)

    # --- embeddings ---
    def embed(self, fasta, layers: list[int], *, pool: str = "mean"):
        """Residual-stream embeddings (D14). See ``idiom.model.export.embed_fasta``."""
        return embed_fasta(self.model, fasta, layers, pool=pool, tokenizer=self.tok, device=self.device)


def _write_fasta(records: list[tuple[str, str]], path) -> Path:
    path = Path(path)
    with path.open("w") as f:
        for header, seq in records:
            f.write(f">{header}\n{seq}\n")
    return path


def main(argv: list[str] | None = None) -> None:
    """``idiom_generate`` — FASTA-first inference CLI."""
    import argparse

    p = argparse.ArgumentParser(description="Generate IDPs/IDRs with IDiom (writes a FASTA).")
    p.add_argument("mode", choices=["idp", "idr"], help="idp = de-novo; idr = context-prompted")
    p.add_argument("--model", required=True, help="HF repo id (e.g. jxliu2/idiom-medium) or local dir")
    p.add_argument("--out", required=True, help="output FASTA")
    p.add_argument("--n", type=int, default=1000, help="sequences (idp) or per protein (idr)")
    p.add_argument("--fasta", help="idr mode: input proteins with _IDR_x-y headers")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    model = IDiom.from_pretrained(args.model)
    kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
              top_k=args.top_k, top_p=args.top_p)
    if args.seed is not None:
        kw["seed"] = args.seed

    if args.mode == "idp":
        model.generate_idp_fasta(args.out, n=args.n, **kw)
    else:
        if not args.fasta:
            p.error("idr mode requires --fasta")
        model.generate_idr_fasta(args.fasta, args.out, n=args.n, **kw)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
