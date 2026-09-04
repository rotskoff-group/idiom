"""The public IDiom and IDiomSAE wrappers.

IDiom wraps an IDiomTransformer and its Tokenizer: loading and saving in the released format,
unprompted and prompted IDR generation, and residual-stream embeddings. IDiomSAE bundles a trained
SAE with its host model and layer: feature activations and feature-steered generation.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_model, save_model

from idiom.data.fim import UNPROMPTED, fim_prompt, normalize_mode
from idiom.data.io import read_records, to_records
from idiom.data.tokenizer import Tokenizer
from idiom.model.config import ModelConfig
from idiom.model.extract import embed_fasta
from idiom.model.io import load_pretrained
from idiom.model.sampling import generate
from idiom.model.transformer import IDiomTransformer
from idiom.sae.features.build_feature_dataset import build_feature_dataset as _build_feature_dataset
from idiom.sae.model.io import load_sae, save_sae
from idiom.sae.steer import SteeringSpec, steer_generation
from idiom.utils.device import resolve_device

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"


def _resolve(name_or_path: str | Path) -> Path:
    """Return a local directory unchanged, or download a Hub repo snapshot.

    Args:
        name_or_path (str | Path): A local directory, or a Hub repo id.

    Returns:
        Path: The local directory.
    """
    p = Path(name_or_path)
    if p.exists():
        return p
    return Path(snapshot_download(str(name_or_path)))


def _oversample(batch_fn, n: int, *, length_range: tuple[int, int] | None = None,
                max_oversample: int = 20, seed: int | None = None) -> list[str]:
    """Draw batches of sequences until n of them fall within a length range.

    With length_range None a single batch is drawn. Reaching the draw cap warns and returns fewer
    than n.

    Args:
        batch_fn (Callable): Draws a batch of sequences given (k, seed).
        n (int): Number of sequences to return.
        length_range (tuple[int, int] | None): Inclusive (lo, hi) length filter, or None.
        max_oversample (int): Cap on total draws, as a multiple of n.
        seed (int | None): Base seed, incremented once per re-draw.

    Returns:
        list[str]: Up to n sequences, each within the length range if one was given.
    """
    if length_range is None:
        return batch_fn(n, seed)
    lo, hi = length_range
    kept: list[str] = []
    drawn, rounds, cap = 0, 0, n * max(1, max_oversample)
    while len(kept) < n and drawn < cap:
        s = None if seed is None else seed + rounds  # vary the seed per batch
        kept.extend(x for x in batch_fn(n, s) if x and lo <= len(x) <= hi)
        drawn += n
        rounds += 1
    if len(kept) < n:
        warnings.warn(f"generate: only {len(kept)}/{n} sequences fell in length {length_range} "
                      f"after {drawn} draws (max_oversample={max_oversample}); returning those.")
    return kept[:n]


class IDiom:
    """A trained IDiom transformer and its tokenizer.

    Attributes:
        model (IDiomTransformer): The wrapped transformer, in eval mode.
        tok (Tokenizer): The tokenizer used for encoding and decoding.
        device (torch.device): The device the model is on.
    """

    def __init__(self, model: IDiomTransformer, tokenizer: Tokenizer | None = None, device="cpu"):
        """Wrap a transformer and tokenizer, moving the model to a device in eval mode.

        Args:
            model (IDiomTransformer): The transformer to wrap.
            tokenizer (Tokenizer | None): Tokenizer to use; a default Tokenizer if None.
            device (str | torch.device): Device to place the model on.
        """
        self.model = model.eval()
        self.tok = tokenizer or Tokenizer()
        self.device = torch.device(device)
        self.model.to(self.device)

    # --- load / save (HF-style) ---
    @classmethod
    def load(cls, name_or_path: str | Path, *, device="auto") -> IDiom:
        """Load from a Lightning checkpoint, a released directory, or a Hub repo id.

        A path naming an existing file is read as a checkpoint; anything else goes to
        from_pretrained.

        Args:
            name_or_path (str | Path): A .ckpt file, a released directory, or a Hub repo id.
            device (str): Target device, or "auto" to resolve one.

        Returns:
            IDiom: The loaded model wrapper.
        """
        if Path(name_or_path).is_file():  # a Lightning .ckpt
            return cls.from_lightning_checkpoint(name_or_path, device=device)
        return cls.from_pretrained(name_or_path, device=device)

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path, *, device="auto") -> IDiom:
        """Load a released directory holding config.json and model.safetensors, or a Hub repo id.

        Args:
            name_or_path (str | Path): A released model directory, or a Hub repo id to download.
            device (str): Target device, or "auto" to resolve one.

        Returns:
            IDiom: The loaded model wrapper.
        """
        d = _resolve(name_or_path)
        cfg = ModelConfig(**json.loads((d / CONFIG_FILE).read_text()))
        model = IDiomTransformer(cfg)
        load_model(model, str(d / WEIGHTS_FILE))  # handles the tied embedding
        return cls(model, device=resolve_device(device))

    def save_pretrained(self, out_dir: str | Path, *, model_card: str | None = None) -> Path:
        """Write config.json and model.safetensors to a directory.

        Args:
            out_dir (str | Path): Directory to write the release into; created if needed.
            model_card (str | None): Text to write as README.md, or None to write no model card.

        Returns:
            Path: The output directory.
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
        """Save in released form and upload to the Hub.

        The repo is created if it does not exist; config.json, model.safetensors, and any model
        card are uploaded to its root.

        Args:
            repo_id (str): Target Hub repo id.
            private (bool): Whether a newly created repo is private.
            model_card (str | None): Text to upload as README.md, or None.
            commit_message (str | None): Commit message for the upload.
            token (str | None): Hub token; the cached login or HF_TOKEN is used if None.

        Returns:
            str: The URL of the uploaded repo.
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
        """Load a Lightning checkpoint, reading the architecture from it.

        Args:
            ckpt_path (str | Path): Path to a Lightning checkpoint.
            device (str): Target device, or "auto" to resolve one.

        Returns:
            IDiom: The loaded model wrapper.

        Raises:
            ValueError: If the checkpoint carries no stored ModelConfig.
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
        seed = kw.pop("seed", None)
        prompt_ids = torch.tensor(self.tok.encode(prompt), device=self.device)

        def _batch(k: int, s: int | None) -> list[str]:
            # one Generator per draw, reused across chunks so each chunk samples fresh tokens (and a
            # draw stays reproducible for a fixed batch_size). batch_size=None -> one batch of k.
            gen = torch.Generator(device=self.device).manual_seed(s) if s is not None else None
            bs = batch_size or k
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
            n (int): Number of IDRs to return.
            max_new_tokens (int): Maximum tokens to generate per sequence.
            temperature (float): Sampling temperature; 0 selects the argmax.
            top_k (int | None): Top-k sampling cutoff, or None.
            top_p (float | None): Nucleus sampling cutoff, or None.
            seed (int | None): Seed for reproducible sampling, or None.
            length_range (tuple[int, int] | None): Inclusive (lo, hi) length filter; sequences are
                redrawn until n fall in range or the oversampling cap is reached.
            max_oversample (int): Cap on total draws, as a multiple of n, when length_range is set.
            batch_size (int | None): Maximum sequences per model forward; None uses one batch.

        Returns:
            list[str]: The generated IDR residue strings, at most n of them.
        """
        kw = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                  length_range=length_range, max_oversample=max_oversample, batch_size=batch_size)
        if seed is not None:
            kw["seed"] = seed
        return self._generate(fim_prompt(), n, **kw)

    def generate_prompted(self, seq: str, idr_start: int, idr_end: int, n: int = 100, **kw) -> list[str]:
        """Generate IDRs conditioned on the flanks of a protein.

        Args:
            seq (str): The full protein sequence providing the flanks.
            idr_start (int): IDR start index (0-based, inclusive).
            idr_end (int): IDR end index (0-based, exclusive).
            n (int): Number of IDRs to return.
            **kw: Sampling and length options; see generate_unprompted.

        Returns:
            list[str]: The generated IDR residue strings, at most n of them.
        """
        return self._generate(fim_prompt(seq, idr_start, idr_end), n, **kw)

    # --- FASTA-first wrappers ---
    def generate_unprompted_fasta(self, out_fasta, n: int = 100, *, prefix: str = "idiom_unprompted",
                                  **kw) -> Path:
        """Generate unprompted IDRs and write them to a record FASTA.

        Each record is headed "{prefix}_{i}_IDR_1-{len}". Empty generations are skipped, so the
        file may hold fewer than n records.

        Args:
            out_fasta (str | Path): Output FASTA path.
            n (int): Number of IDRs to generate.
            prefix (str): Header prefix for each generated record.
            **kw: Sampling and length options; see generate_unprompted.

        Returns:
            Path: The output FASTA path.
        """
        seqs = self.generate_unprompted(n, **kw)
        # the whole generated sequence is the IDR -> header carries the span _IDR_1-len so the
        # output is a valid record FASTA (read_records-parseable). See _idr_header.
        records = [(_idr_header(f"{prefix}_{i}", s), s) for i, s in enumerate(seqs) if s]
        return _write_fasta(records, out_fasta)

    def generate_prompted_fasta(self, in_fasta, out_fasta, n: int = 100, *, return_full: bool = False,
                                marker: str = "idiom_prompted", **kw) -> Path:
        """Generate IDRs for each record of a FASTA and write them to another FASTA.

        Each output record is headed "{source_accession}_{marker}_gen{i}" followed by an IDR span.
        With return_full False the IDR is written alone; with return_full True it is spliced back
        between its flanks and the whole protein is written. Empty generations are skipped.

        Args:
            in_fasta (str | Path): Input proteins with "_IDR_x-y" headers.
            out_fasta (str | Path): Output FASTA path.
            n (int): Number of IDRs to generate per input record.
            return_full (bool): If True, write the whole protein with the IDR spliced in.
            marker (str): Header marker for each generated record.
            **kw: Sampling and length options; see generate_unprompted.

        Returns:
            Path: The output FASTA path.
        """
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
        """Extract residual-stream embeddings for sequences or FASTA records.

        Args:
            inputs (str | Path | list[str]): A record FASTA path, a bare sequence string, or an
                iterable of sequences and/or Records; see idiom.data.io.to_records.
            layers (list[int]): Residual-stream layers to extract.
            pool (str): "mean" for one vector per sequence, or "none" for per-residue rows.

        Returns:
            dict[int, tuple]: Per layer, a (values, index) pair; see
                idiom.model.extract.embed_fasta.
        """
        return embed_fasta(self.model, inputs, layers, pool=pool, tokenizer=self.tok, device=self.device)


class IDiomSAE:
    """A sparse autoencoder bundled with its host model, layer, and training distribution.

    The recorded region and fim_mode are reapplied by encode, build_feature_dataset, and
    steer_generate.

    Attributes:
        sae (SparseCoder): The trained autoencoder, in eval mode on the host's device.
        host (IDiom): The host model whose residual stream the SAE reads.
        layer (int): The residual-stream layer the SAE was trained on.
        host_model (str | None): Recorded repo id or path of the host model.
        region (str): Residues the SAE reads: "all", "idr", or "non_idr".
        fim_mode (str): Prompt format the SAE was trained under: "prompted" or "unprompted".

    Example:
        from idiom import IDiomSAE

        sae = IDiomSAE.from_pretrained("jxliu2/idiomsae-300M-L18-k32")
        feats, accessions = sae.encode("proteins.fasta")
        seqs = sae.steer_generate(feature=1234, strength=0.5, n=100)
    """

    def __init__(self, sae, model: IDiom, layer: int, *, host_model: str | None = None,
                 region: str = "all", fim_mode: str = "prompted"):
        """Bundle an SAE with its host model, layer, and training distribution.

        Args:
            sae (SparseCoder): The trained autoencoder; moved to the host's device in eval mode.
            model (IDiom): The host model whose residual stream the SAE reads.
            layer (int): The residual-stream layer the SAE was trained on.
            host_model (str | None): Repo id or path of the host model, recorded on save.
            region (str): Residues the SAE reads: "all", "idr", or "non_idr".
            fim_mode (str): Prompt format the SAE was trained under: "prompted" or "unprompted".

        Raises:
            ValueError: If fim_mode is neither "prompted" nor "unprompted".
        """
        self.sae = sae.eval().to(model.device)
        self.host = model
        self.layer = int(layer)
        self.host_model = host_model
        # The distribution this SAE was trained on: which residues it reads (region) and the prompt
        # format those activations were taken under (fim_mode). Both are reapplied automatically
        # everywhere downstream — encode, steering, feature datasets — so the SAE is never
        # run on a distribution it did not see. They are independent axes, and "idr" in region (a
        # residue slice) means something different from "prompted" in fim_mode (a prompt format).
        self.region = region
        self.fim_mode = normalize_mode(fim_mode)

    def __repr__(self) -> str:
        """Return the host, layer, training distribution, and SAE shape."""
        return (f"IDiomSAE(host={self.host_model!r}, layer={self.layer}, region={self.region!r}, "
                f"fim_mode={self.fim_mode!r}, latents={self.sae.num_latents}, "
                f"k={getattr(self.sae, 'k', '?')})")

    # convenience pass-throughs to the host model
    @property
    def model(self) -> IDiomTransformer:
        """The host model's transformer."""
        return self.host.model

    @property
    def tok(self) -> Tokenizer:
        """The host model's tokenizer."""
        return self.host.tok

    @property
    def device(self) -> torch.device:
        """The device the host model is on."""
        return self.host.device

    # --- load / save (HF-style, mirrors IDiom) ---
    @classmethod
    def from_pretrained(cls, name_or_path, *, model: IDiom | None = None, device="auto") -> IDiomSAE:
        """Load a released SAE directory holding sae_config.json and sae.safetensors.

        Args:
            name_or_path (str | Path): A released SAE directory, or a Hub repo id to download.
            model (IDiom | None): The host model; loaded from the host_model recorded in the SAE
                config if None.
            device (str): Target device, or "auto" to resolve one.

        Returns:
            IDiomSAE: The loaded SAE wrapper.

        Raises:
            ValueError: If model is None and the config records no host_model.
        """
        d = _resolve(name_or_path)
        sae, cfg = load_sae(d, device=resolve_device(device))
        if model is None:
            if not cfg.get("host_model"):
                raise ValueError(
                    f"{d} records no host_model; pass model=IDiom.from_pretrained(...) explicitly."
                )
            model = IDiom.load(cfg["host_model"], device=device)
        return cls(sae, model, cfg["layer"], host_model=cfg.get("host_model"),
                   region=cfg.get("region", "all"), fim_mode=cfg.get("fim_mode", "prompted"))

    def save_pretrained(self, out_dir, *, host_model: str | None = None) -> Path:
        """Write sae_config.json and sae.safetensors to a directory.

        Args:
            out_dir (str | Path): Directory to write the release into.
            host_model (str | None): Repo id or path of the host model to record; the host_model
                this SAE already carries is used if None.

        Returns:
            Path: The output directory.
        """
        return save_sae(self.sae, out_dir, host_model=host_model or self.host_model,
                        layer=self.layer, region=self.region, fim_mode=self.fim_mode)

    def push_to_hub(self, repo_id: str, *, host_model: str | None = None, private: bool = True,
                    model_card: str | None = None, commit_message: str | None = None,
                    token: str | None = None) -> str:
        """Save in released form and upload the SAE to the Hub.

        The repo is created if it does not exist, and sae_config.json, sae.safetensors, and any
        model card are uploaded to its root, so the result loads with from_pretrained.

        Args:
            repo_id (str): Target Hub repo id for the SAE.
            host_model (str | None): Repo id of the host model to record, such as
                "jxliu2/idiom-300M". A Hub repo id here is what lets the uploaded SAE load its host
                from the Hub. The host_model this SAE already carries is used if None.
            private (bool): Whether a newly created repo is private.
            model_card (str | None): Text to upload as README.md, or None.
            commit_message (str | None): Commit message for the upload.
            token (str | None): Hub token; the cached login or HF_TOKEN is used if None.

        Returns:
            str: The URL of the uploaded repo.
        """
        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            self.save_pretrained(tmp, host_model=host_model)
            if model_card is not None:
                (Path(tmp) / "README.md").write_text(model_card)
            api.upload_folder(repo_id=repo_id, folder_path=tmp, repo_type="model",
                              commit_message=commit_message or f"Upload {repo_id}")
        return f"https://huggingface.co/{repo_id}"

    # --- feature activations ---
    @torch.no_grad()
    def encode(self, inputs, *, pool: str = "mean", region: str | None = None):
        """Compute SAE feature activations for the residues of each record.

        Args:
            inputs (str | Path | list[str]): A record FASTA path, a bare sequence string, or an
                iterable of sequences and/or Records.
            pool (str): "none" for per-residue rows, or "mean" to average over each record's
                residues within region.
            region (str | None): "all", "idr", or "non_idr"; the SAE's training region if None.
                An unprompted-mode SAE encodes "132{IDR}", so only "idr" is available.

        Returns:
            tuple: With pool="none", an [N_res, num_latents] array and a list of per-row metadata
                dicts carrying accession, source_pos, residue, and is_idr. With pool="mean", an
                [N_seq, num_latents] array and the list of accessions it corresponds to.

        Raises:
            ValueError: If region is not "idr" for an unprompted-mode SAE.
        """
        region = region or self.region
        # An unprompted-mode SAE only ever sees "132{IDR}", so there are no non-IDR residues to
        # select: silently returning IDR features under the name "all" (or an empty array for
        # "non_idr") would hide the SAE's training distribution from the caller.
        if self.fim_mode == UNPROMPTED and region != "idr":
            raise ValueError(
                f"region={region!r} is not available from this SAE: it was trained in unprompted "
                f"mode ('132{{IDR}}'), so only IDR residues are encoded and there are no flanking "
                f"residues to select. Use region='idr', or an SAE trained with fim_mode='prompted'.")
        emb = embed_fasta(self.model, inputs, [self.layer], pool="none", tokenizer=self.tok,
                          device=self.device, fim_mode=self.fim_mode)
        values, index = emb[self.layer]
        x = torch.from_numpy(values).to(self.device)
        feats = self.sae.encode_dense(x).cpu().numpy()
        if pool == "none":
            return feats, index

        rows: dict[str, list[int]] = {}
        for i, row in enumerate(index):
            is_idr = row.get("is_idr", True)
            if (region == "idr" and not is_idr) or (region == "non_idr" and is_idr):
                continue
            rows.setdefault(row["accession"], []).append(i)
        accs = list(rows)
        pooled = np.stack([feats[rows[a]].mean(0) for a in accs]) if accs else np.empty((0, feats.shape[1]))
        return pooled, accs

    @torch.no_grad()
    def build_feature_dataset(self, inputs, out_dir, *, batch_size: int = 16) -> Path:
        """Write the per-residue feature-activation dataset for these inputs.

        Args:
            inputs (str | Path | list[str]): A record FASTA path, a bare sequence string, or an
                iterable of sequences and/or Records.
            out_dir (str | Path): Directory to write the feature dataset into.
            batch_size (int): Records per forward pass.

        Returns:
            Path: The output directory.
        """
        return _build_feature_dataset(self.model, self.sae, to_records(inputs), self.layer, out_dir,
                    tokenizer=self.tok, device=self.device, batch_size=batch_size,
                    region=self.region, fim_mode=self.fim_mode)

    # --- steering ---
    @torch.no_grad()
    def steer_generate(self, feature, strength, *, n: int = 100, mode: str = "add_direction",
                       normalize: bool = False, relative: bool = False, preserve_norm: bool = False,
                       prompt: str | None = None, max_new_tokens: int = 1000, temperature: float = 1.0,
                       top_k: int | None = None, top_p: float | None = None, seed: int | None = None,
                       length_range: tuple[int, int] | None = None, max_oversample: int = 20) -> list[str]:
        """Generate IDRs with one or more features steered on the SAE's layer.

        For mode "add_direction", the flags set how strength is read: by default it scales each
        feature's decoder row; with normalize it is the magnitude of the summed unit direction;
        with relative it is a fraction of each position's residual norm, which overrides normalize.

        Args:
            feature (int | list[int]): Feature index or indices to steer.
            strength (float | list[float]): Steering strength, or one value per feature.
            n (int): Number of IDRs to return.
            mode (str): "add_direction", "clamp", or "ablate".
            normalize (bool): Scale the summed decoder rows to unit norm before applying strength.
            relative (bool): Scale the push by each position's residual norm.
            preserve_norm (bool): Restore each position's original residual norm after the push.
            prompt (str | None): Prompt string to steer from; the "132" prompt if None.
            max_new_tokens (int): Maximum tokens to generate per sequence.
            temperature (float): Sampling temperature; 0 selects the argmax.
            top_k (int | None): Top-k sampling cutoff, or None.
            top_p (float | None): Nucleus sampling cutoff, or None.
            seed (int | None): Seed for reproducible sampling, or None.
            length_range (tuple[int, int] | None): Inclusive (lo, hi) length filter, as in
                generate_unprompted.
            max_oversample (int): Cap on total draws, as a multiple of n, when length_range is set.

        Returns:
            list[str]: The steered IDR residue strings, at most n of them.
        """
        spec = SteeringSpec(layer=self.layer, feature_idx=feature, strength=strength, mode=mode,
                            normalize=normalize, relative=relative, preserve_norm=preserve_norm)
        prompt_tokens = self.tok.encode(prompt) if prompt else None

        def _batch(k: int, s: int | None) -> list[str]:
            gen = torch.Generator(device=self.device).manual_seed(s) if s is not None else None
            out = steer_generation(
                self.model, self.sae, spec, prompt_tokens=prompt_tokens, n_samples=k,
                max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                tokenizer=self.tok, region=self.region, generator=gen,
            )
            return [self.host._decode_idr(row) for row in out]

        return _oversample(_batch, n, length_range=length_range, max_oversample=max_oversample, seed=seed)


def _idr_header(accession: str, seq: str) -> str:
    """Return the header "{accession}_IDR_1-{len(seq)}", spanning the whole sequence.

    Args:
        accession (str): The record accession to prefix.
        seq (str): The sequence whose length sets the span.

    Returns:
        str: The FASTA header.
    """
    return f"{accession}_IDR_1-{len(seq)}"


def _write_fasta(records: list[tuple[str, str]], path) -> Path:
    """Write (header, sequence) pairs to a FASTA, one unwrapped line per sequence."""
    path = Path(path)
    with path.open("w") as f:
        for header, seq in records:
            f.write(f">{header}\n{seq}\n")
    return path


def main(argv: list[str] | None = None) -> None:
    """Run the idiom_generate CLI: generate IDRs and write them to a FASTA.

    Args:
        argv (list[str] | None): Argument list; sys.argv[1:] if None.
    """
    p = argparse.ArgumentParser(description="Generate IDRs with IDiom (writes a FASTA).")
    p.add_argument("mode", choices=["unprompted", "prompted"],
                   help="unprompted = de novo (no flanks); prompted = in-filled in flanking context")
    p.add_argument("--model", required=True,
                   help="HF repo id (e.g. jxliu2/idiom-300M), a released dir, or a training .ckpt")
    p.add_argument("--out", required=True, help="output FASTA")
    p.add_argument("--n", type=int, default=1000, help="sequences (unprompted) or per protein (prompted)")
    p.add_argument("--fasta", help="prompted mode: input proteins with _IDR_x-y headers")
    p.add_argument("--return-full", action="store_true",
                   help="prompted mode: splice each IDR back into its flanks and write the whole protein")
    p.add_argument("--max-new-tokens", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--min-len", type=int, default=None, help="oversample until len >= this (inclusive)")
    p.add_argument("--max-len", type=int, default=None, help="oversample until len <= this (inclusive)")
    p.add_argument("--max-oversample", type=int, default=20,
                   help="cap on total draws as a multiple of n when a length range is set")
    p.add_argument("--batch-size", type=int, default=None,
                   help="max sequences per model forward (chunks each draw to bound memory)")
    args = p.parse_args(argv)

    model = IDiom.load(args.model)  # a .ckpt from a training run, a released dir, or a Hub id
    kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
              top_k=args.top_k, top_p=args.top_p, batch_size=args.batch_size)
    if args.seed is not None:
        kw["seed"] = args.seed
    if args.min_len is not None or args.max_len is not None:
        kw["length_range"] = (args.min_len or 1, args.max_len or 10**9)
        kw["max_oversample"] = args.max_oversample

    if args.mode == UNPROMPTED:
        model.generate_unprompted_fasta(args.out, n=args.n, **kw)
    else:
        if not args.fasta:
            p.error("prompted mode requires --fasta")
        model.generate_prompted_fasta(args.fasta, args.out, n=args.n, return_full=args.return_full, **kw)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
