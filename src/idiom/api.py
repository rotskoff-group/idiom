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
from idiom.model.extract import embed_fasta
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
    def load(cls, name_or_path: str | Path, *, device="auto") -> "IDiom":
        """Format-agnostic load: a local training ``.ckpt`` file, a released dir, or an HF repo id."""
        if Path(name_or_path).is_file():  # a Lightning .ckpt
            return cls.from_lightning_checkpoint(name_or_path, device=device)
        return cls.from_pretrained(name_or_path, device=device)

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path, *, device="auto") -> "IDiom":
        from safetensors.torch import load_model  # noqa: PLC0415

        d = _resolve(name_or_path)
        cfg = ModelConfig(**json.loads((d / CONFIG_FILE).read_text()))
        model = IDiomTransformer(cfg)
        load_model(model, str(d / WEIGHTS_FILE))  # handles the tied embedding
        return cls(model, device=resolve_device(device))

    def save_pretrained(self, out_dir: str | Path, *, model_card: str | None = None) -> Path:
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
        """Save in released form and upload to the HF Hub as ``repo_id``; returns the repo URL.

        Creates the repo if missing (private by default), then uploads ``config.json`` +
        ``model.safetensors`` (+ a ``README.md`` model card if ``model_card`` is given) to the repo
        root, so the result loads directly via :meth:`from_pretrained`. ``token`` falls back to the
        cached login / ``HF_TOKEN``.
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
        """Wrap a training ``.ckpt`` (e.g. to then ``save_pretrained`` a release); arch read from it."""
        from idiom.model.io import load_pretrained  # noqa: PLC0415

        dev = resolve_device(device)
        return cls(load_pretrained(ckpt_path, device=dev), device=dev)

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
                  max_oversample: int = 20, **kw) -> list[str]:
        seed = kw.pop("seed", None)

        def _batch(k: int, s: int | None) -> list[str]:
            gen = torch.Generator(device=self.device).manual_seed(s) if s is not None else None
            prompts = torch.tensor(self.tok.encode(prompt), device=self.device).unsqueeze(0).repeat(k, 1)
            out = generate(self.model, prompts, tokenizer=self.tok, generator=gen, **kw)
            return [self._decode_idr(row) for row in out]

        if length_range is None:
            return _batch(n, seed)

        # oversample: keep drawing batches of n and length-filtering until n fall in [lo, hi]
        # (inclusive), capped at n * max_oversample total draws so an unreachable range can't hang.
        lo, hi = length_range
        kept: list[str] = []
        drawn, rounds, cap = 0, 0, n * max(1, max_oversample)
        while len(kept) < n and drawn < cap:
            s = None if seed is None else seed + rounds  # vary the seed per batch
            kept.extend(x for x in _batch(n, s) if x and lo <= len(x) <= hi)
            drawn += n
            rounds += 1
        if len(kept) < n:
            import warnings  # noqa: PLC0415
            warnings.warn(f"generate: only {len(kept)}/{n} sequences fell in length {length_range} "
                          f"after {drawn} draws (max_oversample={max_oversample}); returning those.")
        return kept[:n]

    def generate_idp(self, n: int = 100, *, max_new_tokens: int = 1000, temperature: float = 1.0,
                     top_k: int | None = None, top_p: float | None = None, seed: int | None = None,
                     length_range: tuple[int, int] | None = None, max_oversample: int = 20) -> list[str]:
        """De-novo IDPs (prompt ``132``). Returns ``n`` IDR residue strings.

        ``length_range=(lo, hi)``: oversample — re-generate and length-filter until ``n`` sequences
        have length in ``[lo, hi]`` (inclusive), capped at ``n * max_oversample`` total draws (warns
        and returns fewer if the cap is hit).
        """
        kw = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                  length_range=length_range, max_oversample=max_oversample)
        if seed is not None:
            kw["seed"] = seed
        return self._generate(fim_prompt(), n, **kw)

    def generate_idr(self, seq: str, idr_start: int, idr_end: int, n: int = 100, **kw) -> list[str]:
        """IDRs conditioned on flanks. ``idr_start/idr_end`` are 0-based, half-open (``seq[start:end]``).

        Accepts ``length_range=(lo, hi)`` / ``max_oversample`` (see :meth:`generate_idp`) to oversample
        until ``n`` generated IDRs fall in the length range.
        """
        return self._generate(fim_prompt(seq, idr_start, idr_end), n, **kw)

    # --- FASTA-first wrappers ---
    def generate_idp_fasta(self, out_fasta, n: int = 100, *, prefix: str = "idiom_idp", **kw) -> Path:
        seqs = self.generate_idp(n, **kw)
        # the whole generated sequence is the IDR -> header carries the span `_IDR_1-len` so the
        # output is a valid record FASTA (read_records-parseable). See _idr_header.
        return _write_fasta([(_idr_header(f"{prefix}_{i}", s), s) for i, s in enumerate(seqs) if s], out_fasta)

    def generate_idr_fasta(self, in_fasta, out_fasta, n: int = 100, *, return_full: bool = False,
                           marker: str = "idiom_idr", **kw) -> Path:
        """Generate ``n`` IDRs per input record (each conditioned on that record's flanks).

        Each output record is tagged ``{source_accession}_{marker}_gen{i}`` (``marker`` defaults to
        ``idiom_idr``, symmetric with the ``idiom_idp`` prefix on de-novo IDP output) so generated
        records are distinguishable by mode at a glance; the source accession is preserved in front.

        ``return_full=False`` (default) writes the generated IDR alone with header span ``_IDR_1-len``
        — the original behaviour, so existing analysis code keeps working. ``return_full=True`` splices
        each IDR back into its flanks and writes the whole protein, with the header span pointing at
        the IDR region (``_IDR_{idr_start+1}-{idr_start+len}``, 1-indexed inclusive).
        """
        rows = []
        for r in read_records(in_fasta):
            for i, s in enumerate(self.generate_idr(r.full_seq, r.idr_start, r.idr_end, n, **kw)):
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
    def embed(self, fasta, layers: list[int], *, pool: str = "mean"):
        """Residual-stream embeddings (D14). See ``idiom.model.extract.embed_fasta``."""
        return embed_fasta(self.model, fasta, layers, pool=pool, tokenizer=self.tok, device=self.device)


class IDiomSAE:
    """Public API for a sparse autoencoder trained on an :class:`IDiom` residual stream.

    An SAE is only meaningful together with its **host model** and the **layer** it reads, so this
    wrapper bundles ``(IDiom, layer, SparseCoder)``. The release dir records the host model + layer,
    so an SAE is self-describing about where it mounts — ``idiom_sae`` writes exactly this format.

        from idiom import IDiom, IDiomSAE
        sae   = IDiomSAE.from_pretrained("jxliu2/idiom-medium-sae-L8")   # host auto-loaded
        feats = sae.encode(fasta)                              # feature activations
        seqs  = sae.steer_generate(feature=1234, strength=0.5, n=100)
        fid   = sae.fidelity(records_fasta)
    """

    def __init__(self, sae, model: IDiom, layer: int, *, host_model: str | None = None,
                 region: str = "all", fim_mode: str = "idr"):
        self.sae = sae.eval().to(model.device)
        self.host = model
        self.layer = int(layer)
        self.host_model = host_model
        # the distribution this SAE was trained on (residual-stream slice + prompt format); both are
        # applied automatically everywhere downstream so the SAE stays on-distribution.
        self.region = region
        self.fim_mode = fim_mode

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
        """Load a released SAE dir (``sae_config.json`` + ``sae.safetensors``).

        ``model`` is the host :class:`IDiom`; if omitted, it is loaded from the ``host_model`` recorded
        in the SAE config (HF repo id or local dir).
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
                   region=cfg.get("region", "all"), fim_mode=cfg.get("fim_mode", "idr"))

    def save_pretrained(self, out_dir, *, host_model: str | None = None) -> Path:
        """Write the release dir (``sae_config.json`` + ``sae.safetensors``); ``host_model`` (repo id
        / path) is recorded so the SAE can later self-load its host."""
        from idiom.sae.io import save_sae  # noqa: PLC0415

        return save_sae(self.sae, out_dir, host_model=host_model or self.host_model,
                        layer=self.layer, region=self.region, fim_mode=self.fim_mode)

    # --- feature activations ---
    @torch.no_grad()
    def encode(self, fasta, *, pool: str = "mean", region: str | None = None):
        """SAE feature activations for each record's residues.

        ``region`` (``all`` | ``idr`` | ``non_idr``) defaults to the SAE's training region.
        ``pool="none"`` → ``(acts[N_res, num_latents], index)`` per residue (index rows carry
        ``accession``/``source_pos``/``is_idr``). ``pool="mean"`` → ``(acts[N_seq, num_latents],
        accessions)`` averaged over each record's residues within ``region``.
        """
        region = region or self.region
        emb = embed_fasta(self.model, fasta, [self.layer], pool="none", tokenizer=self.tok,
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
    def build_feature_dataset(self, fasta, out_dir, *, batch_size: int = 16) -> Path:
        """Write the offline per-residue feature-activation dataset (for the feature viewer)."""
        from idiom.data.io import read_records  # noqa: PLC0415
        from idiom.sae.features.build_feature_dataset import build_feature_dataset as _bfd  # noqa: PLC0415

        return _bfd(self.model, self.sae, read_records(fasta), self.layer, out_dir,
                    tokenizer=self.tok, device=self.device, batch_size=batch_size,
                    region=self.region, fim_mode=self.fim_mode)

    # --- steering ---
    @torch.no_grad()
    def steer_generate(self, feature, strength, *, n: int = 100, mode: str = "add_direction",
                       normalize: bool = False, relative: bool = False,
                       prompt: str | None = None, max_new_tokens: int = 1000, temperature: float = 1.0,
                       top_k: int | None = None, top_p: float | None = None, seed: int | None = None) -> list[str]:
        """Generate IDRs with ``feature`` steered on the SAE's layer. Returns residue strings.

        ``normalize=True`` (add_direction only): the push is ``strength * unit(sum of decoder rows)``,
        so ``strength`` is the magnitude in residual-norm units and the number of features sets only
        the direction (not the magnitude). ``relative=True``: push is ``strength * ||x_pos|| *
        unit(sum of decoder rows)`` -- ``strength`` is a dimensionless fraction of the local residual
        norm (self-adapting; overrides ``normalize``)."""
        from idiom.sae.steering import SteeringSpec, steer_generation  # noqa: PLC0415

        spec = SteeringSpec(layer=self.layer, feature_idx=feature, strength=strength, mode=mode,
                            normalize=normalize, relative=relative)
        gen = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        prompt_tokens = self.tok.encode(prompt) if prompt else None
        out = steer_generation(
            self.model, self.sae, spec, prompt_tokens=prompt_tokens, n_samples=n,
            max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
            tokenizer=self.tok, region=self.region, generator=gen,
        )
        return [self.host._decode_idr(row) for row in out]

    # --- fidelity ---
    @torch.no_grad()
    def fidelity(self, fasta, *, batch_size: int = 16, fim_idr_prob: float | None = None):
        """Substitution-loss fidelity (``loss_clean``/``loss_sae``/``loss_ablate``,
        ``pct_loss_recovered``) over a record FASTA. ``fim_idr_prob`` defaults to match the SAE's
        training prompt format (0.0 idp / 1.0 idr), so eval stays on-distribution."""
        from torch.utils.data import DataLoader  # noqa: PLC0415

        from idiom.data.dataset import RecordDataset, make_collate  # noqa: PLC0415
        from idiom.data.io import read_records  # noqa: PLC0415
        from idiom.sae.fidelity import compute_fidelity  # noqa: PLC0415

        if fim_idr_prob is None:
            fim_idr_prob = 0.0 if self.fim_mode == "idp" else 1.0

        ds = RecordDataset(read_records(fasta), self.tok, max_len=self.model.cfg.max_seq_len,
                           fim_idr_prob=fim_idr_prob)
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=make_collate(self.tok.pad_id))
        return compute_fidelity(self.model, self.sae, self.layer, dl, pad_id=self.tok.pad_id,
                                tokenizer=self.tok, region=self.region, device=self.device)


def _idr_header(accession: str, seq: str) -> str:
    """Header for a generated sequence: append ``_IDR_1-len`` (the whole sequence is the IDR).

    Makes generated FASTAs valid record FASTAs that ``read_records`` can parse. ``read_records``
    splits on the last ``_IDR_``, so accessions that themselves contain underscores are fine.
    """
    return f"{accession}_IDR_1-{len(seq)}"


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
    p.add_argument("--return-full", action="store_true",
                   help="idr mode: splice each IDR back into its flanks and write the whole protein")
    p.add_argument("--max-new-tokens", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--min-len", type=int, default=None, help="oversample until len >= this (inclusive)")
    p.add_argument("--max-len", type=int, default=None, help="oversample until len <= this (inclusive)")
    p.add_argument("--max-oversample", type=int, default=20,
                   help="cap on total draws as a multiple of n when a length range is set")
    args = p.parse_args(argv)

    model = IDiom.from_pretrained(args.model)
    kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
              top_k=args.top_k, top_p=args.top_p)
    if args.seed is not None:
        kw["seed"] = args.seed
    if args.min_len is not None or args.max_len is not None:
        kw["length_range"] = (args.min_len or 1, args.max_len or 10**9)
        kw["max_oversample"] = args.max_oversample

    if args.mode == "idp":
        model.generate_idp_fasta(args.out, n=args.n, **kw)
    else:
        if not args.fasta:
            p.error("idr mode requires --fasta")
        model.generate_idr_fasta(args.fasta, args.out, n=args.n, return_full=args.return_full, **kw)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
