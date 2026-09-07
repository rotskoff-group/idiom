"""IDiom model loading, generation, and embeddings."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import HfApi
from safetensors.torch import load_model, save_model

from idiom.api._shared import _oversample, _resolve
from idiom.data.fim import fim_prompt
from idiom.data.io import read_records
from idiom.data.tokenizer import Tokenizer
from idiom.model.config import ModelConfig
from idiom.model.extract import embed_fasta
from idiom.model.io import load_pretrained
from idiom.model.sampling import generate
from idiom.model.transformer import IDiomTransformer
from idiom.utils.device import resolve_device
from idiom.utils.validation import integer_at_least, validate_generation

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"


class IDiom:
    """A trained IDiom transformer and its tokenizer.

    Attributes:
        model (IDiomTransformer): The wrapped transformer, in eval mode.
        tok (Tokenizer): The tokenizer used for encoding and decoding.
        device (torch.device): The device the model is on.
    """

    def __init__(self, model: IDiomTransformer, tokenizer: Tokenizer | None = None, device="cpu"):
        """Move the model to device in eval mode and attach a tokenizer (default if omitted)."""
        self.model = model.eval()
        self.tok = tokenizer or Tokenizer()
        self.device = torch.device(device)
        self.model.to(self.device)

    # --- load / save (HF-style) ---
    @classmethod
    def load(cls, name_or_path: str | Path, *, device="auto") -> IDiom:
        """Load a checkpoint file, release directory, or Hub repository.

        Existing files are read as Lightning checkpoints; other inputs use from_pretrained.
        The returned wrapper is in eval mode; device="auto" uses resolve_device.
        """
        if Path(name_or_path).is_file():  # a Lightning .ckpt
            return cls.from_lightning_checkpoint(name_or_path, device=device)
        return cls.from_pretrained(name_or_path, device=device)

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path, *, device="auto") -> IDiom:
        """Load config.json and model.safetensors from a directory or Hub repository.

        The returned wrapper is in eval mode; device="auto" uses resolve_device.
        """
        d = _resolve(name_or_path)
        cfg = ModelConfig(**json.loads((d / CONFIG_FILE).read_text()))
        model = IDiomTransformer(cfg)
        load_model(model, str(d / WEIGHTS_FILE))  # handles the tied embedding
        return cls(model, device=resolve_device(device))

    def save_pretrained(self, out_dir: str | Path, *, model_card: str | None = None) -> Path:
        """Write config.json and model.safetensors to a directory.

        Args:
            out_dir: Directory to write the release into; created if needed.
            model_card: Text to write as README.md, or None to write no model card.

        Returns:
            The output directory.
        """
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / CONFIG_FILE).write_text(json.dumps(asdict(self.model.cfg), indent=2))
        save_model(self.model, str(d / WEIGHTS_FILE))
        if model_card is not None:
            (d / "README.md").write_text(model_card)
        return d

    def push_to_hub(self, repo_id: str, *, private: bool = True, model_card: str | None = None,
                    commit_message: str | None = None, token: str | None = None) -> str:
        """Save and upload a model release, creating the Hub repository if needed.

        Args:
            repo_id: Target Hub repo id.
            private: Whether a newly created repo is private.
            model_card: Text to upload as README.md, or None.
            commit_message: Commit message for the upload.
            token: Hub token; the cached login or HF_TOKEN is used if None.

        Returns:
            The URL of the uploaded repo.
        """
        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            self.save_pretrained(tmp, model_card=model_card)
            api.upload_folder(repo_id=repo_id, folder_path=tmp, repo_type="model",
                              commit_message=commit_message or f"Upload {repo_id}")
        return f"https://huggingface.co/{repo_id}"

    @classmethod
    def from_lightning_checkpoint(cls, ckpt_path, *, device="auto") -> IDiom:
        """Load a model in eval mode using the checkpoint's stored architecture.

        Raises:
            ValueError: If the checkpoint has no ModelConfig.
        """
        dev = resolve_device(device)
        model, _ = load_pretrained(ckpt_path, device=dev)
        return cls(model, device=dev)

    # --- generation ---
    def _decode_idr(self, row: torch.Tensor) -> str:
        """Decode one generated row to a residue string, stopping at the first STOP or PAD."""
        ids: list[int] = []
        for i in row.tolist():
            if i in (self.tok.stop_id, self.tok.pad_id):
                break  # IDR ends at the first STOP/PAD
            if self.tok.is_residue(i):  # keep residues only; drop any stray FIM markers
                ids.append(i)
        return self.tok.decode(ids)

    @torch.no_grad()
    def _generate(self, prompt: str, n: int, *, length_range: tuple[int, int] | None = None,
                  max_oversample: int = 20, batch_size: int | None = None, **kw) -> list[str]:
        validate_generation(n, length_range=length_range, max_oversample=max_oversample,
                            batch_size=batch_size, **kw)
        seed = kw.pop("seed", None)
        prompt_ids = torch.tensor(self.tok.encode(prompt), device=self.device)

        def _batch(k: int, s: int | None) -> list[str]:
            # one Generator per draw, reused across chunks so each chunk samples fresh tokens (and a
            # draw stays reproducible for a fixed batch_size). batch_size=None uses batches of eight.
            gen = torch.Generator(device=self.device).manual_seed(s) if s is not None else None
            bs = 8 if batch_size is None else batch_size
            if bs <= 0:
                raise ValueError("batch_size must be positive")
            out: list[str] = []
            for off in range(0, k, bs):
                prompts = prompt_ids.unsqueeze(0).repeat(min(bs, k - off), 1)
                rows = generate(self.model, prompts, tokenizer=self.tok, generator=gen, **kw)
                out.extend(self._decode_idr(row) for row in rows)
            return out

        return _oversample(_batch, n, length_range=length_range, max_oversample=max_oversample, seed=seed)

    def generate_unprompted(self, n: int = 100, *, max_new_tokens: int = 1000, temperature: float = 1.0,
                            top_k: int | None = None, top_p: float | None = None, seed: int | None = None,
                            length_range: tuple[int, int] | None = None, max_oversample: int = 20,
                            batch_size: int | None = None) -> list[str]:
        """Generate unprompted IDRs from the bare "132" prompt.

        Args:
            n: Number of IDRs to return.
            max_new_tokens: Maximum tokens to generate per sequence.
            temperature: Sampling temperature; 0 selects the argmax.
            top_k: Top-k sampling cutoff, or None.
            top_p: Nucleus sampling cutoff, or None.
            seed: Seed for reproducible sampling, or None.
            length_range: Inclusive (lo, hi) length filter; sequences are redrawn until n fall in
                range or the oversampling cap is reached.
            max_oversample: Cap on total draws, as a multiple of n, when length_range is set.
            batch_size: Maximum sequences per model forward; None uses eight.

        Returns:
            The generated IDR residue strings, at most n of them.
        """
        kw = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                  length_range=length_range, max_oversample=max_oversample, batch_size=batch_size)
        if seed is not None:
            kw["seed"] = seed
        return self._generate(fim_prompt(), n, **kw)

    def generate_prompted(self, seq: str, idr_start: int, idr_end: int, n: int = 100, **kw) -> list[str]:
        """Generate IDRs conditioned on the flanks of a protein.

        Args:
            seq: The full protein sequence providing the flanks.
            idr_start: IDR start index (0-based, inclusive).
            idr_end: IDR end index (0-based, exclusive).
            n: Number of IDRs to return.
            **kw: Sampling and length options; see generate_unprompted.

        Returns:
            The generated IDR residue strings, at most n of them.
        """
        if not isinstance(seq, str) or not self.tok.is_canonical(seq):
            raise ValueError("seq must be a non-empty string of canonical amino acids")
        integer_at_least("idr_start", idr_start, 0)
        integer_at_least("idr_end", idr_end, 1)
        if not idr_start < idr_end <= len(seq):
            raise ValueError("IDR coordinates must satisfy 0 <= idr_start < idr_end <= len(seq)")
        kw.setdefault("max_new_tokens", 1000)
        return self._generate(fim_prompt(seq, idr_start, idr_end), n, **kw)

    # --- FASTA-first wrappers ---
    def generate_unprompted_fasta(self, out_fasta, n: int = 100, *, prefix: str = "idiom_unprompted",
                                  **kw) -> Path:
        """Generate unprompted IDRs and write them to a record FASTA.

        Each record is headed "{prefix}_{i}_IDR_1-{len}". Empty generations are skipped, so the
        file may hold fewer than n records.

        Args:
            out_fasta (str | Path): Output FASTA path.
            n: Number of IDRs to generate.
            prefix: Header prefix for each generated record.
            **kw: Sampling and length options; see generate_unprompted.

        Returns:
            The output FASTA path.
        """
        seqs = self.generate_unprompted(n, **kw)
        # the whole generated sequence is the IDR -> header carries the span _IDR_1-len so the
        # output is a valid record FASTA (read_records-parseable). See _idr_header.
        records = [(_idr_header(f"{prefix}_{i}", s), s) for i, s in enumerate(seqs) if s]
        return _write_fasta(records, out_fasta)

    def generate_prompted_fasta(self, in_fasta, out_fasta, n: int = 100, *, return_full: bool = False,
                                marker: str = "idiom_prompted", **kw) -> Path:
        """Generate IDRs for each input FASTA record; skip empty generations.

        Headers use "{source_accession}_{marker}_gen{i}" plus the generated IDR span.

        Args:
            in_fasta (str | Path): Input proteins with "_IDR_x-y" headers.
            out_fasta (str | Path): Output FASTA path.
            n: Number of IDRs to generate per input record.
            return_full: If True, write the whole protein with the IDR spliced in.
            marker: Header marker for each generated record.
            **kw: Sampling and length options; see generate_unprompted.

        Returns:
            The output FASTA path.
        """
        validate_generation(n, **kw)
        rows = []
        for r in read_records(in_fasta):
            for i, s in enumerate(self.generate_prompted(r.full_seq, r.idr_start, r.idr_end, n, **kw)):
                if not s:
                    continue
                acc = f"{r.accession}_{marker}_gen{i}"
                if return_full:
                    seq = r.full_seq[: r.idr_start] + s + r.full_seq[r.idr_end :]
                    header = f"{acc}_IDR_{r.idr_start + 1}-{r.idr_start + len(s)}"
                    rows.append((header, seq))
                else:
                    rows.append((_idr_header(acc, s), s))
        return _write_fasta(rows, out_fasta)

    # --- embeddings ---
    def embed(self, inputs, layers: list[int], *, pool: str = "mean"):
        """Extract residual-stream embeddings; see embed_fasta for the output schema.

        Args:
            inputs: A FASTA path, Record, sequence, or iterable accepted by to_records.
            layers: Zero-based transformer block indices.
            pool: "mean" averages IDR residues; "none" returns per-residue rows.

        Returns:
            A mapping from layer index to (values, index).
        """
        return embed_fasta(self.model, inputs, layers, pool=pool, tokenizer=self.tok, device=self.device)



def _idr_header(accession: str, seq: str) -> str:
    """Return "{accession}_IDR_1-{len(seq)}" as a FASTA header."""
    return f"{accession}_IDR_1-{len(seq)}"



def _write_fasta(records: list[tuple[str, str]], path) -> Path:
    """Write (header, sequence) pairs to a FASTA, one unwrapped line per sequence."""
    path = Path(path)
    with path.open("w") as f:
        for header, seq in records:
            f.write(f">{header}\n{seq}\n")
    return path
