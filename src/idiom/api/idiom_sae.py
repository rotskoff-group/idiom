"""IDiomSAE loading, feature extraction, and steered generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi

from idiom.api._shared import _oversample, _resolve
from idiom.api.idiom import IDiom
from idiom.data.fim import UNPROMPTED, normalize_mode
from idiom.data.io import to_records
from idiom.data.tokenizer import Tokenizer
from idiom.model.extract import embed_fasta
from idiom.model.transformer import IDiomTransformer
from idiom.sae.features.build_feature_dataset import build_feature_dataset as _build_feature_dataset
from idiom.sae.model.io import load_sae, save_sae
from idiom.sae.steer import SteeringSpec, steer_generation
from idiom.utils.device import resolve_device
from idiom.utils.validation import validate_generation


class IDiomSAE:
    """An SAE bundled with its host model and training layer, region, and FIM mode."""

    def __init__(self, sae, model: IDiom, layer: int, *, host_model: str | None = None,
                 region: str = "all", fim_mode: str = "prompted"):
        """Bundle an SAE with its host model, layer, and training distribution.

        Args:
            sae (SparseCoder): The trained autoencoder; moved to the host's device in eval mode.
            model: The host model whose residual stream the SAE reads.
            layer: Zero-based training block index.
            host_model: Host model repository or path, recorded on save.
            region: Residues the SAE reads: "all", "idr", or "non_idr".
            fim_mode: Prompt format the SAE was trained under: "prompted" or "unprompted".

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
            model: The host model; loaded from the host_model recorded in the SAE config if None.
            device (str): Device; "auto" uses resolve_device.

        Returns:
            The loaded SAE wrapper.

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
            host_model: Host model repository or path to record; the host_model this SAE already
                carries is used if None.

        Returns:
            The output directory.
        """
        return save_sae(self.sae, out_dir, host_model=host_model or self.host_model,
                        layer=self.layer, region=self.region, fim_mode=self.fim_mode)

    def push_to_hub(self, repo_id: str, *, host_model: str | None = None, private: bool = True,
                    model_card: str | None = None, commit_message: str | None = None,
                    token: str | None = None) -> str:
        """Save and upload an SAE release, creating the Hub repository if needed.

        Args:
            repo_id: Target Hub repo id for the SAE.
            host_model: Repo id of the host model to record, such as "jxliu2/idiom-300M". A Hub repo
                id here is what lets the uploaded SAE load its host from the Hub. The host_model
                this SAE already carries is used if None.
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
            inputs: A FASTA path, Record, sequence, or iterable accepted by to_records.
            pool: "none" for per-residue rows, or "mean" to average over each record's residues
                within region.
            region: "all", "idr", or "non_idr"; the SAE's training region if None. An
                unprompted-mode SAE accepts only "idr". Filters mean pooling only;
                pool="none" returns all encoded residues.

        Returns:
            tuple: With pool="none", an [N_res, num_latents] array and a list of per-row metadata
                dicts carrying record_idx, accession, source_pos, residue, and is_idr. With
                pool="mean", an [N_seq, num_latents] array and its accessions in input order;
                repeated accessions remain separate records. Records with no selected residues
                are omitted.

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

        rows: dict[int, list[int]] = {}
        for i, row in enumerate(index):
            is_idr = row.get("is_idr", True)
            if (region == "idr" and not is_idr) or (region == "non_idr" and is_idr):
                continue
            rows.setdefault(row["record_idx"], []).append(i)
        accs = [index[indices[0]]["accession"] for indices in rows.values()]
        pooled = (np.stack([feats[indices].mean(0) for indices in rows.values()])
                  if rows else np.empty((0, feats.shape[1])))
        return pooled, accs

    @torch.no_grad()
    def build_feature_dataset(self, inputs, out_dir, *, batch_size: int = 16) -> Path:
        """Write the per-residue feature-activation dataset for these inputs.

        Args:
            inputs (str | Path | list[str]): A record FASTA path, a bare sequence string, or an
                iterable of sequences and/or Records.
            out_dir (str | Path): Directory to write the feature dataset into.
            batch_size: Records per forward pass.

        Returns:
            The output directory.
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
                       length_range: tuple[int, int] | None = None, max_oversample: int = 20,
                       batch_size: int | None = None) -> list[str]:
        """Generate IDRs with SAE feature steering; see SteeringSpec for strength semantics.

        Args:
            feature (int | list[int]): Feature index or indices to steer.
            strength (float | list[float]): Steering strength, or one value per feature.
            n: Number of IDRs to return.
            mode: "add_direction", "clamp", or "ablate".
            normalize: Scale the summed decoder rows to unit norm before applying strength.
            relative: Scale the push by each position's residual norm.
            preserve_norm: Restore residual norms after relative addition only.
            prompt: Prompt string to steer from; the "132" prompt if None.
            max_new_tokens: Maximum tokens to generate per sequence.
            temperature: Sampling temperature; 0 selects the argmax.
            top_k: Top-k sampling cutoff, or None.
            top_p: Nucleus sampling cutoff, or None.
            seed: Seed for reproducible sampling, or None.
            length_range: Inclusive (lo, hi) length filter, as in generate_unprompted.
            max_oversample: Cap on total draws, as a multiple of n, when length_range is set.
            batch_size: Maximum sequences per model forward; None uses eight.

        Returns:
            The steered IDR residue strings, at most n of them.
        """
        validate_generation(n, max_new_tokens=max_new_tokens, temperature=temperature,
                            top_k=top_k, top_p=top_p, seed=seed, length_range=length_range,
                            max_oversample=max_oversample, batch_size=batch_size)
        spec = SteeringSpec(layer=self.layer, feature_idx=feature, strength=strength, mode=mode,
                            normalize=normalize, relative=relative, preserve_norm=preserve_norm)
        prompt_tokens = self.tok.encode(prompt) if prompt else None

        def _batch(k: int, s: int | None) -> list[str]:
            gen = torch.Generator(device=self.device).manual_seed(s) if s is not None else None
            bs = 8 if batch_size is None else batch_size
            if bs <= 0:
                raise ValueError("batch_size must be positive")
            sequences = []
            for off in range(0, k, bs):
                out = steer_generation(
                    self.model, self.sae, spec, prompt_tokens=prompt_tokens, n_samples=min(bs, k - off),
                    max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                    tokenizer=self.tok, region=self.region, generator=gen,
                )
                sequences.extend(self.host._decode_idr(row) for row in out)
            return sequences

        return _oversample(_batch, n, length_range=length_range, max_oversample=max_oversample, seed=seed)
