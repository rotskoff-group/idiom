"""Public API: the IDiom model wrapper (FASTA-first, HF-friendly).

A thin layer over IDiomTransformer + Tokenizer that hides training/Hydra/Lightning. Load a released
model with from_pretrained (an HF repo id or a local dir), generate IDRs (unprompted or prompted) to
strings or FASTA, or pull residual-stream embeddings.

An unprompted IDR is generated de novo, from the bare "132" prompt with no flanks; a prompted IDR is
in-filled conditioned on its flanking context. Both are the same FIM model — they differ only in
what the prompt contains.

    from idiom import IDiom
    model = IDiom.from_pretrained("jxliu2/idiom-300M")
    idrs  = model.generate_unprompted(n=100)                          # de novo IDRs
    idrs  = model.generate_prompted(protein_seq, start, end, n=100)   # IDR in flanks (0-based, half-open)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from idiom.data.fim import UNPROMPTED, fim_prompt, normalize_mode
from idiom.data.io import read_records
from idiom.data.tokenizer import Tokenizer
from idiom.model.config import ModelConfig
from idiom.model.extract import embed_fasta
from idiom.model.sampling import generate
from idiom.model.transformer import IDiomTransformer
from idiom.utils.device import resolve_device

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"


def _resolve(name_or_path: str | Path) -> Path:
    """Return a local dir as-is, or download the HF repo snapshot and return its path.

    Args:
        name_or_path (str | Path): A local directory, or an HF repo id to download.

    Returns:
        Path: The local path to the (possibly downloaded) directory.
    """
    p = Path(name_or_path)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    return Path(snapshot_download(str(name_or_path)))


def _oversample(batch_fn, n: int, *, length_range: tuple[int, int] | None = None,
                max_oversample: int = 20, seed: int | None = None) -> list[str]:
    """Draw sequences via batch_fn(k, seed) until n fall within length_range (inclusive).

    With length_range=None this is a single batch_fn(n, seed) draw. Otherwise it re-draws batches of
    n (varying the seed per batch) and length-filters, capped at n * max_oversample total draws so an
    unreachable range cannot hang; it warns and returns fewer if the cap is hit. Shared by
    IDiom._generate and IDiomSAE.steer_generate so unsteered and steered generation length-filter
    identically.

    Args:
        batch_fn (Callable): Draws a batch of sequences given (k, seed).
        n (int): Number of sequences to return.
        length_range (tuple[int, int] | None): Inclusive (lo, hi) length filter, or None for none.
        max_oversample (int): Cap on total draws as a multiple of n.
        seed (int | None): Base seed, incremented per re-draw.

    Returns:
        list[str]: Up to n sequences within the length range.
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
        import warnings  # noqa: PLC0415
        warnings.warn(f"generate: only {len(kept)}/{n} sequences fell in length {length_range} "
                      f"after {drawn} draws (max_oversample={max_oversample}); returning those.")
    return kept[:n]


class IDiom:
    """FASTA-first wrapper around a trained IDiom transformer and its tokenizer.

    Loads released or checkpoint weights, generates unprompted/prompted IDRs (as strings or FASTA),
    and extracts residual-stream embeddings. See from_pretrained to load a released model.
    """

    def __init__(self, model: IDiomTransformer, tokenizer: Tokenizer | None = None, device="cpu"):
        """Wrap a transformer and tokenizer, moving the model to device in eval mode.

        Args:
            model (IDiomTransformer): The transformer to wrap (set to eval mode).
            tokenizer (Tokenizer | None): Tokenizer to use (a default Tokenizer if None).
            device (str | torch.device): Device to place the model on.
        """
        self.model = model.eval()
        self.tok = tokenizer or Tokenizer()
        self.device = torch.device(device)
        self.model.to(self.device)

    # --- load / save (HF-style) ---
    @classmethod
    def load(cls, name_or_path: str | Path, *, device="auto") -> "IDiom":
        """Load from a local training .ckpt file, a released dir, or an HF repo id.

        Args:
            name_or_path (str | Path): A Lightning .ckpt file, a released dir, or an HF repo id.
            device (str): Target device, or "auto" to pick automatically.

        Returns:
            IDiom: The loaded model wrapper.
        """
        if Path(name_or_path).is_file():  # a Lightning .ckpt
            return cls.from_lightning_checkpoint(name_or_path, device=device)
        return cls.from_pretrained(name_or_path, device=device)

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path, *, device="auto") -> "IDiom":
        """Load a released model dir (config.json + model.safetensors) or an HF repo id.

        Args:
            name_or_path (str | Path): A released model directory or an HF repo id.
            device (str): Target device, or "auto" to pick automatically.

        Returns:
            IDiom: The loaded model wrapper.
        """
        from safetensors.torch import load_model  # noqa: PLC0415

        d = _resolve(name_or_path)
        cfg = ModelConfig(**json.loads((d / CONFIG_FILE).read_text()))
        model = IDiomTransformer(cfg)
        load_model(model, str(d / WEIGHTS_FILE))  # handles the tied embedding
        return cls(model, device=resolve_device(device))

    def save_pretrained(self, out_dir: str | Path, *, model_card: str | None = None) -> Path:
        """Write the released form (config.json + model.safetensors) to out_dir.

        Args:
            out_dir (str | Path): Directory to write the release into (created if needed).
            model_card (str | None): Optional README.md model-card text to include.

        Returns:
            Path: The output directory.
        """
        from safetensors.torch import save_model  # noqa: PLC0415

        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / CONFIG_FILE).write_text(json.dumps(asdict(self.model.cfg), indent=2))
        save_model(self.model, str(d / WEIGHTS_FILE))
        if model_card is not None:
            (d / "README.md").write_text(model_card)
        return d

    def push_to_hub(self, repo_id: str, *, private: bool = True, model_card: str | None = None,
                    commit_message: str | None = None, token: str | None = None) -> str:
        """Save in released form and upload to the HF Hub, returning the repo URL.

        Creates the repo if missing (private by default), then uploads config.json and
        model.safetensors (plus a README.md model card if model_card is given) to the repo root, so
        the result loads directly via from_pretrained.

        Args:
            repo_id (str): Target Hub repo id.
            private (bool): Whether to create the repo as private.
            model_card (str | None): Optional model-card text (written as README.md).
            commit_message (str | None): Commit message for the upload.
            token (str | None): Hub token; falls back to the cached login or HF_TOKEN.

        Returns:
            str: The URL of the uploaded repo.
        """
        import tempfile  # noqa: PLC0415

        from huggingface_hub import HfApi  # noqa: PLC0415

        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            self.save_pretrained(tmp, model_card=model_card)
            api.upload_folder(repo_id=repo_id, folder_path=tmp, repo_type="model",
                              commit_message=commit_message or f"Upload {repo_id}")
        return f"https://huggingface.co/{repo_id}"

    @classmethod
    def from_lightning_checkpoint(cls, ckpt_path, *, device="auto") -> "IDiom":
        """Wrap a training .ckpt (e.g. to then save_pretrained a release); arch is read from it.

        Args:
            ckpt_path (str | Path): Path to a Lightning checkpoint.
            device (str): Target device, or "auto" to pick automatically.

        Returns:
            IDiom: The loaded model wrapper.
        """
        from idiom.model.io import load_pretrained  # noqa: PLC0415

        dev = resolve_device(device)
        model, _ = load_pretrained(ckpt_path, device=dev)
        return cls(model, device=dev)

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
        """Generate n unprompted (de novo) IDRs from the bare "132" prompt, as residue strings.

        Args:
            n (int): Number of IDRs to return.
            max_new_tokens (int): Maximum tokens to generate per sequence.
            temperature (float): Sampling temperature (0 = greedy).
            top_k (int | None): Top-k sampling cutoff, or None.
            top_p (float | None): Nucleus sampling cutoff, or None.
            seed (int | None): Random seed for reproducible sampling.
            length_range (tuple[int, int] | None): Inclusive (lo, hi); oversample and length-filter
                until n sequences fall in range, capped at n * max_oversample draws (warns and returns
                fewer if the cap is hit).
            max_oversample (int): Cap on total draws as a multiple of n when length_range is set.
            batch_size (int | None): Max sequences per model forward (chunks each draw to bound
                memory); None generates the whole draw in one batch.

        Returns:
            list[str]: The generated IDR residue strings.
        """
        kw = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                  length_range=length_range, max_oversample=max_oversample, batch_size=batch_size)
        if seed is not None:
            kw["seed"] = seed
        return self._generate(fim_prompt(), n, **kw)

    def generate_prompted(self, seq: str, idr_start: int, idr_end: int, n: int = 100, **kw) -> list[str]:
        """Generate n prompted IDRs conditioned on flanking context, as residue strings.

        idr_start and idr_end are 0-based, half-open (seq[start:end]).

        Args:
            seq (str): The full protein sequence providing the flanks.
            idr_start (int): IDR start index (0-based, inclusive).
            idr_end (int): IDR end index (0-based, exclusive).
            n (int): Number of IDRs to return.
            **kw: Sampling and length options; see generate_unprompted (temperature, top_k, top_p,
                seed, length_range, max_oversample, batch_size).

        Returns:
            list[str]: The generated IDR residue strings.
        """
        return self._generate(fim_prompt(seq, idr_start, idr_end), n, **kw)

    # --- FASTA-first wrappers ---
    def generate_unprompted_fasta(self, out_fasta, n: int = 100, *, prefix: str = "idiom_unprompted",
                                  **kw) -> Path:
        """Generate n unprompted IDRs and write them to a FASTA.

        Each generated sequence is the whole IDR, so its header carries the span _IDR_1-len, making
        the output a valid record FASTA.

        Args:
            out_fasta (str | Path): Output FASTA path.
            n (int): Number of IDRs to generate.
            prefix (str): Header prefix tag for each generated record.
            **kw: Sampling and length options; see generate_unprompted.

        Returns:
            Path: The output FASTA path.
        """
        seqs = self.generate_unprompted(n, **kw)
        # the whole generated sequence is the IDR -> header carries the span _IDR_1-len so the
        # output is a valid record FASTA (read_records-parseable). See _idr_header.
        return _write_fasta([(_idr_header(f"{prefix}_{i}", s), s) for i, s in enumerate(seqs) if s], out_fasta)

    def generate_prompted_fasta(self, in_fasta, out_fasta, n: int = 100, *, return_full: bool = False,
                                marker: str = "idiom_prompted", **kw) -> Path:
        """Generate n prompted IDRs per input record (each conditioned on that record's flanks).

        Each output record is tagged {source_accession}_{marker}_gen{i} (marker defaults to
        "idiom_prompted", symmetric with the "idiom_unprompted" prefix on de novo output), so
        generated records are distinguishable by mode at a glance and keep the source accession in
        front. With return_full=False (default) the generated IDR is written alone with header span
        _IDR_1-len. With return_full=True each IDR is spliced back into its flanks and the whole
        protein is written, with the header span pointing at the IDR region
        (_IDR_{idr_start+1}-{idr_start+len}, 1-indexed inclusive).

        Args:
            in_fasta (str | Path): Input proteins with _IDR_x-y headers.
            out_fasta (str | Path): Output FASTA path.
            n (int): Number of IDRs to generate per input record.
            return_full (bool): If True, splice each IDR into its flanks and write the whole protein.
            marker (str): Header marker tag for each generated record.
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
            inputs (str | Path | list[str]): A record FASTA path, a bare sequence string, or a list
                of sequences / Records. Bare sequences are treated as unprompted IDRs (the whole
                sequence is the IDR). See embed_fasta and idiom.data.io.to_records.
            layers (list[int]): Residual-stream layers to extract.
            pool (str): "mean" for one vector per sequence, or "none" for per-residue rows.

        Returns:
            dict[int, tuple]: Per layer, a (values, index) pair (see embed_fasta).
        """
        return embed_fasta(self.model, inputs, layers, pool=pool, tokenizer=self.tok, device=self.device)


class IDiomSAE:
    """Public API for a sparse autoencoder trained on an IDiom residual stream.

    An SAE is only meaningful together with its host model and the layer it reads, so this wrapper
    bundles (IDiom, layer, SparseCoder). The release dir records the host model and layer, so an SAE
    is self-describing about where it mounts; idiom_sae writes exactly this format.

        from idiom import IDiom, IDiomSAE
        sae   = IDiomSAE.from_pretrained("jxliu2/idiomsae-300M-L18-k32")  # host auto-loaded
        feats = sae.encode(fasta)                              # feature activations
        seqs  = sae.steer_generate(feature=1234, strength=0.5, n=100)
        fid   = sae.fidelity(records_fasta)
    """

    def __init__(self, sae, model: IDiom, layer: int, *, host_model: str | None = None,
                 region: str = "all", fim_mode: str = "prompted"):
        """Bundle an SAE with its host model, layer, and training distribution.

        Args:
            sae (SparseCoder): The trained sparse autoencoder (set to eval mode).
            model (IDiom): The host model whose residual stream the SAE reads.
            layer (int): The residual-stream layer the SAE was trained on.
            host_model (str | None): Repo id or path of the host model, recorded for self-loading.
            region (str): Residue slice the SAE reads: "all", "idr", or "non_idr".
            fim_mode (str): Prompt format the SAE was trained under: "prompted" or "unprompted".
        """
        self.sae = sae.eval().to(model.device)
        self.host = model
        self.layer = int(layer)
        self.host_model = host_model
        # The distribution this SAE was trained on: which residues it reads (region) and the prompt
        # format those activations were taken under (fim_mode). Both are reapplied automatically
        # everywhere downstream — encode, steering, fidelity, feature datasets — so the SAE is never
        # run on a distribution it did not see. They are independent axes, and "idr" in region (a
        # residue slice) means something different from "prompted" in fim_mode (a prompt format).
        self.region = region
        self.fim_mode = normalize_mode(fim_mode)

    # convenience pass-throughs to the host model
    @property
    def model(self) -> IDiomTransformer:
        return self.host.model

    @property
    def tok(self) -> Tokenizer:
        return self.host.tok

    @property
    def device(self) -> torch.device:
        return self.host.device

    # --- load / save (HF-style, mirrors IDiom) ---
    @classmethod
    def from_pretrained(cls, name_or_path, *, model: IDiom | None = None, device="auto") -> "IDiomSAE":
        """Load a released SAE dir (sae_config.json + sae.safetensors).

        Args:
            name_or_path (str | Path): The released SAE directory or an HF repo id.
            model (IDiom | None): The host IDiom; if None, it is loaded from the host_model recorded
                in the SAE config.
            device (str): Target device, or "auto" to pick automatically.

        Returns:
            IDiomSAE: The loaded SAE wrapper.

        Raises:
            ValueError: If model is None and the config records no host_model.
        """
        from idiom.sae.io import load_sae  # noqa: PLC0415

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
        """Write the release dir (sae_config.json + sae.safetensors).

        Args:
            out_dir (str | Path): Directory to write the release into.
            host_model (str | None): Repo id or path of the host model to record; defaults to the
                host_model this SAE already carries.

        Returns:
            Path: The output directory.
        """
        from idiom.sae.io import save_sae  # noqa: PLC0415

        return save_sae(self.sae, out_dir, host_model=host_model or self.host_model,
                        layer=self.layer, region=self.region, fim_mode=self.fim_mode)

    def push_to_hub(self, repo_id: str, *, host_model: str | None = None, private: bool = True,
                    model_card: str | None = None, commit_message: str | None = None,
                    token: str | None = None) -> str:
        """Save in released form and upload the SAE to the HF Hub; returns the repo URL.

        Creates the repo if missing (private by default), then uploads sae_config.json and
        sae.safetensors (plus a README.md model card if given), so the result loads directly via
        IDiomSAE.from_pretrained.

        Args:
            repo_id (str): Target Hub repo id for the SAE.
            host_model (str | None): Repo id of the host model to record (e.g. "jxliu2/idiom-300M").
                Set this when publishing: it is what lets the released SAE self-load its host from
                the Hub. Defaults to whatever this SAE carries, which may be a local training path.
            private (bool): Whether a newly created repo is private.
            model_card (str | None): Optional model-card text (written as README.md).
            commit_message (str | None): Commit message for the upload.
            token (str | None): Hub token; falls back to the cached login or HF_TOKEN.

        Returns:
            str: The URL of the uploaded repo.
        """
        import tempfile  # noqa: PLC0415

        from huggingface_hub import HfApi  # noqa: PLC0415

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
        """Compute SAE feature activations for each record's residues.

        Args:
            inputs (str | Path | list[str]): A record FASTA path, a bare sequence string, or a list
                of sequences / Records (bare sequences are treated as unprompted IDRs).
            pool (str): "none" for per-residue rows, or "mean" to average over each record's residues.
            region (str | None): "all", "idr", or "non_idr"; defaults to the SAE's training region.

        Returns:
            tuple: With pool="none", (acts[N_res, num_latents], index) where index rows carry
                accession/source_pos/is_idr. With pool="mean", (acts[N_seq, num_latents], accessions)
                averaged over each record's residues within region.
        """
        region = region or self.region
        emb = embed_fasta(self.model, inputs, [self.layer], pool="none", tokenizer=self.tok,
                          device=self.device, fim_mode=self.fim_mode)
        values, index = emb[self.layer]
        x = torch.from_numpy(values).to(self.device)
        feats = self.sae.encode_dense(x).cpu().numpy()
        if pool == "none":
            return feats, index
        import numpy as np  # noqa: PLC0415

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
        """Write the offline per-residue feature-activation dataset (for the feature viewer).

        Args:
            inputs (str | Path | list[str]): A record FASTA path, a bare sequence string, or a list
                of sequences / Records (bare sequences are treated as unprompted IDRs).
            out_dir (str | Path): Directory to write the feature dataset into.
            batch_size (int): Records per forward pass.

        Returns:
            Path: The output directory.
        """
        from idiom.data.io import to_records  # noqa: PLC0415
        from idiom.sae.features.build_feature_dataset import build_feature_dataset as _bfd  # noqa: PLC0415

        return _bfd(self.model, self.sae, to_records(inputs), self.layer, out_dir,
                    tokenizer=self.tok, device=self.device, batch_size=batch_size,
                    region=self.region, fim_mode=self.fim_mode)

    # --- steering ---
    @torch.no_grad()
    def steer_generate(self, feature, strength, *, n: int = 100, mode: str = "add_direction",
                       normalize: bool = False, relative: bool = False, preserve_norm: bool = False,
                       prompt: str | None = None, max_new_tokens: int = 1000, temperature: float = 1.0,
                       top_k: int | None = None, top_p: float | None = None, seed: int | None = None,
                       length_range: tuple[int, int] | None = None, max_oversample: int = 20) -> list[str]:
        """Generate IDRs with feature steered on the SAE's layer, returning residue strings.

        With normalize=True (add_direction only) the push is strength * unit(sum of decoder rows), so
        strength is the magnitude in residual-norm units and the number of features sets only the
        direction. With relative=True the push is strength * ||x_pos|| * unit(sum of decoder rows), so
        strength is a dimensionless fraction of the local residual norm (self-adapting; overrides
        normalize). With preserve_norm=True (relative only) each position is renormed back to
        ||x_pos|| after the push, so steering rotates x toward the feature at constant residual norm
        instead of inflating it.

        Args:
            feature (int | list[int]): Feature index or indices to steer.
            strength (float | list[float]): Steering strength(s); interpretation depends on the flags.
            n (int): Number of IDRs to return.
            mode (str): Steering mode, e.g. "add_direction".
            normalize (bool): Push by a unit direction scaled by strength (add_direction only).
            relative (bool): Push by a fraction of the local residual norm (overrides normalize).
            preserve_norm (bool): Renorm back to the original residual norm after the push (relative only).
            prompt (str | None): Optional generation prompt; defaults to unprompted.
            max_new_tokens (int): Maximum tokens to generate per sequence.
            temperature (float): Sampling temperature (0 = greedy).
            top_k (int | None): Top-k sampling cutoff, or None.
            top_p (float | None): Nucleus sampling cutoff, or None.
            seed (int | None): Random seed for reproducible sampling.
            length_range (tuple[int, int] | None): Inclusive (lo, hi); oversample and length-filter
                until n steered IDRs fall in range, as in generate_unprompted.
            max_oversample (int): Cap on total draws as a multiple of n when length_range is set.

        Returns:
            list[str]: The steered IDR residue strings.
        """
        from idiom.sae.steering import SteeringSpec, steer_generation  # noqa: PLC0415

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

    # --- fidelity ---
    @torch.no_grad()
    def fidelity(self, inputs, *, batch_size: int = 16, prompted_prob: float | None = None):
        """Compute substitution-loss fidelity over sequences or a record FASTA.

        Returns loss_clean, loss_sae, loss_ablate, and pct_loss_recovered. prompted_prob defaults to
        match the SAE's training prompt format (0.0 unprompted / 1.0 prompted), so eval stays
        on-distribution.

        Args:
            inputs (str | Path | list[str]): A record FASTA path, a bare sequence string, or a list
                of sequences / Records (bare sequences are treated as unprompted IDRs).
            batch_size (int): Records per batch.
            prompted_prob (float | None): Probability of the prompted variant; defaults from fim_mode.

        Returns:
            FidelityResult: The fidelity metrics.
        """
        from torch.utils.data import DataLoader  # noqa: PLC0415

        from idiom.data.dataset import RecordDataset, make_collate  # noqa: PLC0415
        from idiom.data.io import to_records  # noqa: PLC0415
        from idiom.sae.eval.fidelity import compute_fidelity  # noqa: PLC0415

        if prompted_prob is None:
            prompted_prob = 0.0 if self.fim_mode == UNPROMPTED else 1.0

        ds = RecordDataset(to_records(inputs), self.tok, max_len=self.model.cfg.max_seq_len,
                           prompted_prob=prompted_prob)
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=make_collate(self.tok.pad_id))
        return compute_fidelity(self.model, self.sae, self.layer, dl, pad_id=self.tok.pad_id,
                                tokenizer=self.tok, region=self.region, device=self.device)


def _idr_header(accession: str, seq: str) -> str:
    """Build a header for a generated sequence by appending _IDR_1-len (the whole seq is the IDR).

    This makes generated FASTAs valid record FASTAs that read_records can parse; read_records splits
    on the last _IDR_, so accessions that themselves contain underscores are fine.

    Args:
        accession (str): The record accession to prefix.
        seq (str): The generated sequence (its length sets the span).

    Returns:
        str: The FASTA header.
    """
    return f"{accession}_IDR_1-{len(seq)}"


def _write_fasta(records: list[tuple[str, str]], path) -> Path:
    path = Path(path)
    with path.open("w") as f:
        for header, seq in records:
            f.write(f">{header}\n{seq}\n")
    return path


def main(argv: list[str] | None = None) -> None:
    """Run the idiom_generate CLI: generate IDRs and write them to a FASTA."""
    import argparse

    p = argparse.ArgumentParser(description="Generate IDRs with IDiom (writes a FASTA).")
    p.add_argument("mode", choices=["unprompted", "prompted"],
                   help="unprompted = de novo (no flanks); prompted = in-filled in flanking context")
    p.add_argument("--model", required=True, help="HF repo id (e.g. jxliu2/idiom-300M) or local dir")
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

    model = IDiom.from_pretrained(args.model)
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
