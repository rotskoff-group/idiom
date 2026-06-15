# AGENTS.md — IDiom (v2 refactor)

Orientation for agents working in this repo. The detailed decision log (D1–D20) and phase plan
live in **`REFACTOR_PLAN.md`** — read it for specifics; this file is the durable conventions,
gotchas, and "where things are." As of 2026-06-14 the refactor is code-complete (P0–P5 + public
API), ~70 CPU tests green; training/RL/SAE runs and the heavy curation drivers are operator/GPU.

## What this is
`IDiom` is an autoregressive (decoder-only) protein language model for **intrinsically disordered
regions (IDRs)**, trained on AFDB-curated IDRs with a **fill-in-the-middle (FIM)** objective so it
generates IDRs conditioned on flanking context (or de-novo IDPs). This refactor **merges the old
`idiom` (generator) + `idiomatics` (SAE interpretability) into one clean, public-ready package**,
adds KV-cache inference, on-the-fly tokenization (no token h5), and retrains at larger scale
(12L de-risk → 24L → 36L) and 1024-token context.

## Repo layout
- **`src/idiom/`** — the shipped wheel (public API). Subpkgs: `data` (tokenizer, FIM, dataset,
  io, datamodule), `model` (RMSNorm/SwiGLU/RoPE/QK-norm + KV cache, sampling, activations, io,
  export), `train` (Lightning pretrain/SFT + `train/grpo`), `sae` (streaming SAE + fidelity +
  steering + feature datasets), `api.py` (`IDiom.from_pretrained`/generate/embed).
- **`data_pipeline/`**, **`analysis/`**, **`rewards/`** — clone-only (NOT in the wheel),
  importable in tests via pytest `pythonpath = ["src", "."]`. Curation, paper figures, and example
  GRPO rewards respectively.
- `tests/` (CPU-only), `bash/` (example Slurm scripts that spell out every Hydra override),
  `REFACTOR_PLAN.md`, `README.md`.

## Environment, compute, tests
- Python via **`uv`**: run anything as `uv run python ...` / `uv run pytest -q`. Venv at `.venv`.
- **Route compute through Slurm** (shared GPU box): `salloc -G 1 -c 4 --mem 16GB`, then
  `srun --pty bash`; drop `-G 1` for CPU. Plain `srun` (no `-G`) still exposes the H100 but does
  not account it — fine for quick profiling, not for real jobs. CPU paths must run anytime.
- Tests are CPU-only and fast; keep them green. Config: flat/fat **Hydra** YAMLs (keep Hydra).
- Full repo backup: `/data2/scratch/jxliu2/tmp/idiom_backup_2026-06-14`. Heavy artifacts are
  gitignored / HF-hosted; old `models/` (27G) + `datasets/` (177G) live in
  `/data2/scratch/group_scratch/idr_plm/2026-06-14_refactor/`.

## Core conventions (do not drift from these)
- **Tokenizer** (`data/tokenizer.py`): char-level, fixed 27-token alphabet — 20 AAs (`ACDEFGHIKLMNPQRSTVWY`,
  ids 0–19), 3 FIM markers `1/2/3` (ids 20–22), controls `PAD/START/STOP/MASK` (23–26).
  `residue_mask(ids) = ids < 20`. Non-canonical residues are **hard-dropped** (D15) — enforced once,
  in `data/io.py`, so curation/training/inference behave identically.
- **FIM** (`data/fim.py`): markers are `1`=prefix, `3`=suffix, `2`=middle(IDR). Full variant =
  `1{prefix}3{suffix}2{IDR}`; de-novo variant = `132{IDR}`. Chosen per-sample at random (this
  replaces the old stored ×2 duplication).
- **Record / FASTA convention** (`data/io.py`, "Option A"): one FASTA convention everywhere —
  each entry is a **full protein** whose header ends `_IDR_{x}-{y}` (**1-indexed, inclusive**)
  marking the IDR span; the line is the `full_seq`. Parsed to `Record(accession, full_seq,
  idr_start, idr_end)` with **0-indexed half-open** coords (`idr = full_seq[start:end]`). No token
  h5 / no precompute — FIM assembly + tokenization happen on the fly in `data/dataset.py`.
- **Length budget:** a `full` example occupies `len(full_seq) + 4` positions (3 FIM markers +
  START/STOP boundary; input/target are the same-length teacher-forcing shift). So
  `FIM_OVERHEAD = 4` and the protein cap = **`max_len − 4` = 1020** for a 1024-token model.
  `1024` is the model's context; the residue cap is derived.

## Model
RMSNorm + SwiGLU + QK-norm + RoPE + no-bias + **tied embeddings**, SDPA attention with a KV cache
(`is_causal = q_len == kv_len`). Sequence-only (no structural tokens). GPT-2 family sizes,
`head_dim 64`: **12L/d768/12h**, **24L/d1024/16h**, **36L/d1280/20h**. KV-cache verified
(cached == uncached); ~3–5× decode speedup. `extract_activations(..., drop_markers=True)` aligns
residual-stream vectors to residue positions for the SAE / downstream use.

## Post-training (GRPO) & SAE
- GRPO must stay **legacy-faithful**: **quadratic** length/entropy penalties
  (`-((x-target)/(target*width))²`), not Gaussian. Custom rewards: users `@register_reward` in
  `rewards/` and pass `reward.module=...` (see `rewards/example_rewards.py`).
- SAE: top-k `SparseCoder` (EleutherAI `sparsify`), fully **streaming** `ActivationStore` (no h5),
  plus fidelity + steering. Prior findings (see memory): edits must be confined to **residue
  positions**; feature steering strength `α≈0.5` is the usable knob.

## Data
- **Canonical data home:** `/data2/scratch/group_scratch/idr_plm/2026-06-14_idiom_data/`
  (→ future HF dataset `jxliu2/idiom-datasets`). Code stays pure (no data in repo). Same filesystem
  (dev 64768) as the scratch sources, so `mv` between them is instant.
- **⚠️ `…/0000_dump/sshfs/4080/*` is a LIVE sshfs mount** to `rotskoff.stanford.edu` — never `rm`
  there (hits the remote, frees no local disk); `umount` instead.
- **Verified AFDB lineage** (matches paper SI; all counts measured):
  AFDB v4 (1.02M proteomes) → extract IDRs (Tesei: window-15 pLDDT, folded>80/disordered<70,
  IDR 30–4096) → **110.46M** raw IDRs (`AFDB_v4_idr_alldata/` 64 parts; `…_seqs.fasta`) →
  mmseqs linclust 90% → **53.38M** reps (`clustering_90/AFDB_IDR_90_reps.fasta` + `mapping.tsv`) →
  keep IDRs of representative proteins → **curated master `clustering_90/AFDB_IDR_90_alldata.h5`**
  = **73.04M** IDRs over 53.38M proteins (1.37 IDRs/protein). v1 then applied ≤512-len + drop
  ~1/3 fully-low-pLDDT → ~**37M** FIM training set (the legacy `…_FIM_512.h5`, intentionally NOT
  in the v2 dir).

## v2 curation pipeline (`data_pipeline/`, see `RUNBOOK.md`)
On-the-fly store; **no FIM materialization**. Stages (pure cores CPU-tested; mmseqs/h5 drivers are
operator-run via Slurm):
1. **extract** (`extract.extract_idrs`) — done; produced the master.
2. **cluster** 90% + `find_fastas_in_h5.py` — done; produced the 73M master.
3. **filter** (`filter_length_plddt.py`, `python -m data_pipeline.filter_length_plddt`) — master h5 → record FASTA, applying
   `passes_length` (≤1020) and `is_fully_low_plddt` (= `not extract.has_folded_segment`, the
   **aggressive** no-folded-segment criterion: drops proteins with no folded *segment*, stricter
   than `max(plddt)<80`).
4. **DisProt dedup** (`dedup.py` + `dedup.bash`) — drop record IDRs ≥50% id to any DisProt IDR,
   **IDR-vs-IDR** (ESM-2 params `--min-seq-id 0.5 -c 0.8 -s 7`).
   - **4b. [FUTURE] TM/SignalP/coiled-coil filter** — DeepTMHMM + SignalP + coiled-coil removal,
     slots in here (after dedup, before split). Not yet implemented.
5. **split** (`split.py`) — plain **random, record-level** 99/0.5/0.5 → `train/val/test.fasta` for
   `RecordDataModule`.

`_legacy_reference/` holds verbatim v1 scripts (`make_AFDB_FIM.py`, `find_fastas_in_h5.py`, …) —
reference only; the two filters were baked into v1 `make_AFDB_FIM.py` (no standalone pLDDT script
ever existed).

## Manuscript & figures
- Figure scripts: `analysis/figures/` (run with `PYTHONPATH=/data2/scratch/jxliu2/idiom`). Use
  `_style.use_style()` (loads `idiom.mplstyle`, font **Liberation Sans** installed in `~/.fonts`)
  and `save_fig(fig, name, subdir="si_figs/<topic>")` → `$IDIOM_FIG_DIR` = the manuscript `figs/`
  dir. **The filename IS the LaTeX `\includegraphics` path.** SI figs under
  `si_figs/{dataset,disorder,biophysics,curves,sae}/`.
- Manuscript repo: `/data2/scratch/jxliu2/papers/overleaf/IDiom-manuscript-v1` (own git → GitHub
  `jxliu2/IDiom-manuscript-v1`; SI under `\appendix`). **After ANY `.tex` change, recompile:**
  `PATH=/data2/scratch/jxliu2/tmp/bin tectonic --synctex --keep-logs main.tex`. Keep paper-text
  additions **minimal** for now. Proactively suggest plots at natural moments.

## Distribution
HF org **`jxliu2`**. Models → `jxliu2/idiom-*` as safetensors + config.json via
`IDiom.save_pretrained` / `from_lightning_checkpoint`. **FASTA-first** user interface; "simple is
better for every aspect." Public API exposed via `IDiom` + `idiom_generate` CLI.

## Gotchas
- Deps that may be missing in a fresh venv: `loguru`, `safetensors`, `h5py` — `uv pip install` as
  needed (usually already in `uv.lock`).
- `tectonic` is not on PATH by default → `/data2/scratch/jxliu2/tmp/bin/tectonic`.
- The curated master moved into `clustering_90/` (was at `pretraining/AFDB/`);
  `filter_length_plddt.py`'s `DEFAULT_MASTER` points at the current path.
- `accession_ids` in the master are `{base}_{s}-{e}` (e.g. `D2CY91-F1_0-149`); `idr_end` is
  0-based **inclusive**, so the 1-based-inclusive header span is `(idr_start+1, idr_end+1)`.
- Persistent agent memory lives at
  `/home/jxliu2/.claude/projects/-data2-scratch-jxliu2/memory/` (see `idiom-data-directory`,
  `idiom-v2-refactor-status`, `idiom-paper-figures`).
