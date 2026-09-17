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
from idiom.model.extract import extract_embeddings
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

    @classmethod
    def load(cls, name_or_path: str | Path, *, device="auto") -> IDiom:
        """Load a Lightning checkpoint, release directory, or Hub repository.

        Args:
            name_or_path: Checkpoint file, local release directory, or Hub repository ID.
                Existing files use from_lightning_checkpoint; other inputs use from_pretrained.
            device: Target device. "auto" uses IDIOM_DEVICE, then CUDA if available, else CPU.

        Returns:
            An IDiom wrapper with the loaded transformer in evaluation mode.
        """
        if Path(name_or_path).is_file():
            return cls.from_lightning_checkpoint(name_or_path, device=device)
        return cls.from_pretrained(name_or_path, device=device)

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path, *, device="auto") -> IDiom:
        """Load config.json and model.safetensors from a local or Hub release.

        Hub artifacts are downloaded and cached on first use.

        Args:
            name_or_path: Local release directory or Hub repository ID.
            device: Target device. "auto" uses IDIOM_DEVICE, then CUDA if available, else CPU.

        Returns:
            An IDiom wrapper with the loaded transformer in evaluation mode.
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

    def push_to_hub(
        self,
        repo_id: str,
        *,
        private: bool = True,
        model_card: str | None = None,
        commit_message: str | None = None,
        token: str | None = None,
    ) -> str:
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
            api.upload_folder(
                repo_id=repo_id,
                folder_path=tmp,
                repo_type="model",
                commit_message=commit_message or f"Upload {repo_id}",
            )
        return f"https://huggingface.co/{repo_id}"

    @classmethod
    def from_lightning_checkpoint(cls, ckpt_path, *, device="auto") -> IDiom:
        """Load a Lightning checkpoint using its stored architecture.

        Args:
            ckpt_path: Path to the checkpoint file.
            device: Target device. "auto" uses IDIOM_DEVICE, then CUDA if available, else CPU.

        Returns:
            An IDiom wrapper with the loaded transformer in evaluation mode.

        Raises:
            ValueError: If the checkpoint has no stored ModelConfig.
        """
        dev = resolve_device(device)
        model, _ = load_pretrained(ckpt_path, device=dev)
        return cls(model, device=dev)

    def _decode_idr(self, row: torch.Tensor) -> str:
        """Decode one generated row to a residue string, stopping at the first STOP or PAD."""
        ids: list[int] = []
        for i in row.tolist():
            if i in (self.tok.stop_id, self.tok.pad_id):
                break
            if self.tok.is_residue(i):
                ids.append(i)
        return self.tok.decode(ids)

    @torch.no_grad()
    def _generate(
        self,
        prompt: str,
        n: int,
        *,
        length_range: tuple[int, int] | None = None,
        max_oversample: int = 20,
        batch_size: int | None = None,
        **kw,
    ) -> list[str]:
        """Generate and decode IDRs in batches with optional length filtering and a draw cap."""
        validate_generation(
            n, length_range=length_range, max_oversample=max_oversample, batch_size=batch_size, **kw
        )
        seed = kw.pop("seed", None)
        prompt_ids = torch.tensor(self.tok.encode(prompt), device=self.device)

        def _batch(k: int, s: int | None) -> list[str]:
            """Generate k decoded IDRs in bounded batches using an optional shared random seed."""
            # Reuse the RNG across chunks; reproducibility depends on batch_size
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

    def generate_unprompted(
        self,
        n: int = 100,
        *,
        max_new_tokens: int = 1000,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        length_range: tuple[int, int] | None = None,
        max_oversample: int = 20,
        batch_size: int | None = None,
    ) -> list[str]:
        """Generate unprompted IDRs from the bare "132" prompt.

        Args:
            n: Number of IDRs to return.
            max_new_tokens: Maximum sampled tokens per sequence, including STOP; limited by context size.
            temperature: Sampling temperature; 0 selects the argmax.
            top_k: Top-k sampling cutoff, or None.
            top_p: Nucleus sampling cutoff, or None.
            seed: Random seed, or None. Reproduction requires the same batch size and settings.
            length_range: Inclusive (lo, hi) bounds on decoded residue counts. Redraw until
                n sequences pass or the oversampling cap is reached.
            max_oversample: Cap on total draws, as a multiple of n, when length_range is set.
            batch_size: Maximum sequences per model forward; None uses eight.

        Returns:
            Up to n decoded IDR strings, possibly empty without a length filter.
            Returns an empty list for n=0 and may return fewer than n strings if the
            length-filter draw cap is reached.

        Raises:
            ValueError: If generation options are invalid; see validate_generation.
        """
        kw = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            length_range=length_range,
            max_oversample=max_oversample,
            batch_size=batch_size,
        )
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
            Up to n decoded IDR strings, without flanks. Empty strings and length-filter
            limits follow generate_unprompted.

        Raises:
            ValueError: If the sequence, IDR coordinates, or generation options are invalid,
                or the flank prompt exceeds the context limit.
        """
        if not isinstance(seq, str) or not self.tok.is_canonical(seq):
            raise ValueError("seq must be a non-empty string of canonical amino acids")
        integer_at_least("idr_start", idr_start, 0)
        integer_at_least("idr_end", idr_end, 1)
        if not idr_start < idr_end <= len(seq):
            raise ValueError("IDR coordinates must satisfy 0 <= idr_start < idr_end <= len(seq)")
        kw.setdefault("max_new_tokens", 1000)
        return self._generate(fim_prompt(seq, idr_start, idr_end), n, **kw)

    def generate_unprompted_fasta(
        self, out_fasta, n: int = 100, *, prefix: str = "idiom_unprompted", **kw
    ) -> Path:
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
        records = [(_idr_header(f"{prefix}_{i}", s), s) for i, s in enumerate(seqs) if s]
        return _write_fasta(records, out_fasta)

    def generate_prompted_fasta(
        self,
        in_fasta,
        out_fasta,
        n: int = 100,
        *,
        return_full: bool = False,
        marker: str = "idiom_prompted",
        **kw,
    ) -> Path:
        """Generate IDRs for each input FASTA record; skip empty generations.

        Headers use "{source_accession}_{marker}_gen{i}" plus the generated IDR span.
        Invalid sequences and IDR spans in nonempty headers are skipped with logged
        counts. Empty headers raise IndexError when they reach span parsing.

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

    def embed(
        self,
        inputs,
        layers: list[int],
        *,
        pool: str = "mean",
        region: str = "idr",
        order: str = "sequence",
    ):
        """Extract residual-stream embeddings for IDRs and their flanks.

        Invalid FASTA sequences and spans in nonempty headers are skipped with logged
        counts. Bare sequences must contain only uppercase canonical residues. Supplied
        Records must already have valid sequences and spans.

        Args:
            inputs: FASTA path, Record, bare sequence, or iterable accepted by to_records.
                Empty per-residue selections return zero rows.
            layers: Zero-based transformer block indices.
            pool: "mean" averages selected residues; "none" returns per-residue vectors.
            region: "idr" (default), "non_idr", or "all". Flanks remain model context.
            order: "sequence" (default) or "fim", for per-residue output.

        Returns:
            A dictionary mapping each layer to (values, index), with a NumPy array and
            one metadata dictionary per row. Mean pooling returns [N_records, d_model]
            values and metadata containing accession, n_idr, n_residues, and record_idx.
            Per-residue output has shape [N_residues, d_model] and metadata containing record_idx, accession,
            source_pos, residue, and is_idr.
            Residue rows follow original protein order by default, without markers.
            record_idx identifies the accepted input record; source_pos is its zero-based
            protein position. Bare sequences are treated as entirely IDR.

        Raises:
            ValueError: If a bare sequence is empty or noncanonical, a Path is missing,
                an option is invalid, or a mean selection is empty.
            IndexError: If a FASTA entry reaching span parsing has an empty header.
        """
        return extract_embeddings(
            self.model,
            inputs,
            layers,
            pool=pool,
            region=region,
            order=order,
            tokenizer=self.tok,
            device=self.device,
        )


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
