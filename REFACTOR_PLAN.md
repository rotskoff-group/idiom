# IDiom v2 — Refactor Plan

**Status:** in progress · **Branch:** `refactor/v2` (off `feature/latent-extraction`) · **Owner:** jxliu2

This is a *living document*. Every refactor step updates it: tick the checklists, append to
the Progress Log, and revise decisions as they're made.

---

## 1. Objective

Merge `idiom` (generative model + training) and `idiomatics` (SAE interpretability) into a
single, clean, public-ready package under **`idiom`**, then re-run the whole program from
scratch on a larger model:

- retrain IDiom at **24L / d1024 (~350M, GPT-2 medium)** first, **1024** max protein length;
- re-do post-training (GRPO/ProtGPS), SAE training, and all analysis on the new model;
- ship a minimal, well-exposed public API + hosted checkpoints.

Because **everything is retrained, there is no backward-compatibility burden** — no old
checkpoints or h5 layouts must remain readable. We make clean breaks freely.

## 2. Locked decisions

| # | Decision | Choice |
|---|----------|--------|
| D1 | Repo / branch | Combine under `idiom`; long-lived branch `refactor/v2` (becomes `main`/`v2` at release). idiomatics merged by **copying** source into the new layout. |
| D2 | Training corpus | **Raw records + FIM on the fly.** Store `(accession, full_seq, idr_start, idr_end[, plddt])`; dataset does FIM-transform + char-tokenize + START/STOP + shift per batch. `full`/`132` variants = runtime augmentation. No precompute, no token h5. |
| D3 | SAE activations | **Fully streaming, regenerated per run** — an `ActivationStore` runs the frozen model live during SAE training; residue-only positions; no activation h5. |
| D4 | Training framework | **Keep PyTorch Lightning** (pretrain / GRPO / SAE). |
| D5 | Model size | Config-driven **GPT-2 family** (head_dim 64): **12L/d768/12h (small — test) → 24L/d1024/16h (medium — primary) → 36L/d1280/20h (large)**. Factories `idiom_12l/24l/36l`. |
| D20 | Repo structure & artifacts | **Three buckets, one repo:** `src/idiom/` (lean shipped library — `model/data/train/sae/utils/configs`; `analysis/` folded into `sae/`), `data_pipeline/` (pretraining corpus build: extract/cluster/split/filters — repo-only, not shipped), `analysis/` (downstream paper analysis + figures + eval — repo-only). Boundary test: *does a `pip install idiom` user need it to run the model?* Heavy deps behind a `[paper]` extra. **Large artifacts (`models/`, `datasets/`) are gitignored, regenerable, never in git or the wheel** — distributed via **HF Hub** (`jxliu2/idiom-*` models/SAEs + `jxliu2/idiom-datasets`); users get them via `from_pretrained` (auto-download) or `hf download --local-dir`. Runs happen in dated `group_scratch/.../YYYY-MM-DD_x/` dirs, not git. |
| D19 | Keep legacy code | During the refactor, **keep the legacy `idiom.nn` / `idiom.scripts` / `rewards/` (and the separate `idiomatics` repo) in place for reference** — don't delete. New code reimplements faithfully (esp. RL: quadratic length/entropy penalties + `-scale*(raw-target)^2` shaping, matching the known-good runs). Prune only once v2 is validated. |
| D18 | SFT | Supervised fine-tuning shares the **same** `LitAutoregressive` module as pretraining. Differences live in config/data only: **completion-only loss mask** (loss on the IDR + STOP, via `RecordDataset(completion_only=True)`) and **warm-start** from a pretrained ckpt (`init_from`). One entrypoint (`idiom_train`), two flat configs (`pretrain.yaml` / `sft.yaml`). |
| D17 | Model architecture | **RMSNorm + SwiGLU + QK-norm (RMSNorm) + RoPE + bias-off + tied embeddings**, SDPA attention with a KV cache. Modernizes the legacy ESM-style block's LayerNorm→RMSNorm; keeps SwiGLU/QK-norm/no-bias; swaps rel-pos-bias→RoPE; drops structural tokens (D8). Causal via SDPA `is_causal` (right-pad + causal ⇒ no pad mask needed). |
| D6 | Config system | **Hydra, flat style.** A few *fat* self-contained YAMLs (`pretrain.yaml`, `grpo.yaml`, `sae.yaml`, `curate.yaml`, `generate.yaml`) — params nested inside one file per task, **no** cross-file `defaults` group composition. Hydra still gives CLI overrides + multirun sweeps. |
| D7 | Positional encoding | **RoPE only**; drop the relative-position-bias (RPE) path (RoPE KV-caches & extrapolates to 1024 cleanly). |
| D8 | Structural tokens | **Removed.** Model is sequence-only (`structural_tokens` were all-PAD, `sequence_id` was just the mask). |
| D9 | CPU testability | Everything must run **CPU-only on demand** (shared GPU box). CPU-first device resolution + tiny-config CPU tests in CI. |
| D10 | Public weights format | Release as **safetensors** + `config.json` on the HF Hub (org **`jxliu2`** for now); Lightning `.ckpt` stays internal-only. Follow standard HF conventions (`from_pretrained`/`push_to_hub`, model cards) and expose standard generation options (temperature, top-k, top-p, max_new_tokens, seed). |
| D11 | Primary user interface | **FASTA-first.** Users drive generation/analysis with FASTA in and FASTA out wherever possible (headers carry `_IDR_x-y` like today); the Python API is the thin layer beneath. |
| D12 | Guiding bias | **Simple > clever, everywhere.** Fewer files, fewer abstractions, fewer knobs. When in doubt, choose the more obvious implementation. |
| D13 | Testing / compute | Run **everything through the Slurm scheduler** — `srun -c 4 --mem 16GB -t … <cmd>` for CPU, add `-G 1` for GPU; interactive via `salloc … && srun --pty bash`. Never compute on the login node. Scratchpad for throwaway outputs: `/data2/scratch/group_scratch/idr_plm/2026-06-14_refactor`. |
| D16 | IDR coordinates | **Header = 1-based inclusive** (`_IDR_x-y`, the biology/UniProt/PDB standard and existing public convention — what users read off a database). **Internal `Record` = 0-based half-open** (`idr = full_seq[start:end]`, Python-slice native; length `end-start`, no `+1`s). Convert once, in `parse_idr_header`. Standard convention in each place. |
| D15 | Non-canonical residues | **Drop the whole sequence.** Any incoming FASTA sequence containing a residue outside the 20 canonical AAs (e.g. `X B Z U O *`, lowercase) is dropped at ingestion — one policy shared by curation, training records, and inference/extract FASTAs — with a logged count of how many were dropped. No UNK token, no substitution; `Tokenizer.encode` hard-fails on a stray non-canonical char so nothing slips through silently. Enforced in the single FASTA reader (`data/io.py`). |
| D14 | Activation extraction | **One hook-based core, two consumers.** A single layer-activation extractor (FASTA in) feeds both the SAE training stream (ephemeral, D3) *and* a user-facing **save-to-disk** path, so exported embeddings are identical to what the SAE trains on. User export is **FASTA-first**, multi-layer, with `--pool mean\|none` (per-sequence vs per-residue), written as **safetensors/npy + a small index** (not h5), ESM-`extract.py`-style UX. Both consumers can **drop the FIM markers (1/3/2) + control tokens** (residue-only — default for SAE), and every kept row carries **alignment metadata** (seq id, residue identity, source position in `full_seq`; IDR-vs-flank derivable from the coords) so activation vectors map 1:1 to their residues. |

## 3. Target layout

```
idiom/
  REFACTOR_PLAN.md
  pyproject.toml            # single env; hatchling + hydra + lightning
  src/idiom/
    __init__.py             # public API surface (from_pretrained, generate, ...)
    data/                   # tokenizer (on-the-fly), FIM transform, datasets, curation/
    model/                  # IDiomTransformer, rope, attention+KV-cache, sampling
    train/                  # pretrain LightningModule, grpo/, schedulers
    rewards/                # protgps + custom  (migrates from repo-root rewards/)
    sae/                    # sparse_coder, lit_sae, ActivationStore, fidelity, steering
    analysis/               # interpretability + figures/
    configs/                # consolidated hydra configs
    utils/                  # device, io, logging
  tests/                    # CPU-only pytest
  legacy/  (transient)      # nn/, scripts/ kept until migrated, then deleted
```

Legacy mapping (deleted as each is ported): `nn/` → `model/`; `scripts/transformer/{train,inference,extract_activations,precompute}` → `train/` + `model/` + `sae/` + (precompute removed); `scripts/data/*` → `data/curation/`.

**Flat config style (D6).** One fat self-contained YAML per task — no `defaults:` group
composition. Hydra still gives CLI overrides (`model.n_layers=36`) and multirun
(`-m sae.k=32,64`). Rule of thumb: **a Hydra group dir only when a slot has ≥2 genuinely
interchangeable options**; otherwise it's a section in the fat file. (The legacy
`cfgs/{model,training,data,inference}/transformer.yaml` had one option each — pure
indirection.) Example `configs/pretrain.yaml`:

```yaml
seed: 0
device: auto                 # auto | cpu | cuda  (CPU-first; IDIOM_DEVICE overrides)
data:   {records: /path/records, max_len: 1024, fim_full_prob: 0.5, batch_size: 64, num_workers: 8, val_frac: 0.005}
model:  {n_layers: 24, d_model: 1024, n_heads: 16, max_seq_len: 1024, rope_base: 10000}
optim:  {lr: 3.5e-4, warmup_steps: 3000, max_steps: 250000, weight_decay: 0.0}
trainer:{devices: 1, precision: bf16-mixed, accumulate_grad_batches: 1, val_check_interval: 25000}
log:    {wandb: false, project: idiom, out_dir: ${oc.env:IDIOM_OUT,./runs}/pretrain}
```

## 4. Public API & exposure (D1, D11, public-ready)

Goal: a user can `pip install idiom` and do useful things in ~5 lines, **primarily through
FASTA files**. Keep the surface small and obvious; hide Hydra/Lightning behind it.

**FASTA-first (the primary path).** In a FASTA, out a FASTA:

```bash
# de-novo IDPs -> FASTA
idiom generate idp  --n 1000 --out idps.fasta
# context-prompted IDRs: input headers end with _IDR_x-y (1-indexed), as today
idiom generate idr  --fasta proteins.fasta --n 1000 --out idrs.fasta
# extract residual-stream activations for downstream use (same core the SAE trains on)
idiom extract --fasta proteins.fasta --layers 8 12 --pool mean --out emb/   # per-seq vectors
idiom extract --fasta proteins.fasta --layers 8    --pool none --out emb/   # per-residue
```

**Python API (the thin layer beneath):**

```python
from idiom import IDiom, SparseCoder
model = IDiom.from_pretrained("jxliu2/idiom-medium")            # downloads from HF
model.generate_idp(n=100, max_new_tokens=256, temperature=1.0)  # standard HF-style opts
model.generate_idr(sequence, idr_start, idr_end, n=100)         # context-prompted
sae = SparseCoder.from_pretrained("jxliu2/idiom-sae-L8")
```

- `IDiom`: `from_pretrained`, `generate_idp`, `generate_idr`, FASTA helpers, `.forward(...,
  return_hidden_states=)`, `.embed(fasta, layers=[...], pool=...)` (the user-facing wrapper
  over the shared extraction core, D14), KV-cached sampling underneath; standard generation
  kwargs (temperature/top_k/top_p/max_new_tokens/seed).
- `Tokenizer`: fixed alphabet (no data-derived vocab); `encode`/`decode`/`fim` helpers.
- `SparseCoder`: `from_pretrained`, `encode`/`decode`, steering hooks.
- Keep low-level training (Lightning/Hydra) importable but *not* required for inference.

## 5. Checkpoint hosting (public access)

- **Host on the HF Hub** under **`jxliu2`** (for now), matching the current
  `jxliu2/idiom` + `jxliu2/idiom-datasets` pattern.
- **Release format = safetensors + `config.json`** (framework-agnostic, safe, smaller than
  `.ckpt`). Provide `scripts/convert_ckpt_to_safetensors.py` (Lightning `.ckpt` → release).
- One repo per artifact class, tagged by size/layer:
  - `jxliu2/idiom-medium` (24L), later `-large` (36L) — `config.json`, `model.safetensors`.
  - `jxliu2/idiom-rl` — per-compartment GRPO checkpoints (subfolders).
  - `jxliu2/idiom-sae` — SAEs by layer/k/expansion.
  - `jxliu2/idiom-datasets` — curated record shards (raw FIM inputs), pointer-only.
- `from_pretrained` resolves a name → HF download via `huggingface_hub` (already a dep),
  caches locally; also accepts a local path. No giant files in git.
- Standard HF conventions throughout (`from_pretrained`/`push_to_hub`, a `MODEL_CARD.md`
  per release: arch, data, license, intended use).

## 6. Eliminating h5

| current h5 use | v2 |
|---|---|
| precompute token shards | **gone** — tokenize on the fly (D2) |
| curated FIM corpus | raw record shards (FASTA/parquet) (D2) |
| SAE activation shards | **gone** — streaming `ActivationStore` (D3) |
| feature dataset | parquet / on-demand |
| AFDB extraction master | stays binary (legit big-array case) |

## 7. Phase plan

Legend: ☐ todo · ◐ in progress · ☑ done

### P0 — Scaffold & merge  ☑
- ☑ Create `refactor/v2` branch off `feature/latent-extraction`.
- ☑ Write this plan (living doc).
- ☑ Create package skeleton (`data/model/train/sae/analysis`).
- ☑ CPU-first device utility + CPU-only pytest harness.
- ☑ Copy idiomatics **core** into `sae/` + `analysis/`, imports rewired `idiomatics.`→`idiom.`:
  `sae/{sparse_coder,lit_sae,fidelity}`, `sae/steering/{hooks,steer}`,
  `analysis/feature_activations`, and `idiom_interface/tokens.py`→`data/tokens.py`.
  `idiom_interface/model.py` (the loader bridge) **not** copied — replaced by `IDiom.from_pretrained` in P2.
- ☑ Merge dependency sets into one `pyproject` (+ `sparrow` uv source).
- ☑ Tests for the SAE port (forward/encode-decode + residue mask), 5/5 green via `srun`.

**Deferred (intentionally, to keep P0 a clean, green step):**
- `rewards/` migration → **P4** (with the GRPO port); moving it now would break the legacy
  reward registry. `protgps` stays vendored at repo root for now.
- idiomatics Hydra **scripts** (train_sae, build_feature_dataset, annotate_features,
  compute_fidelity, steer_generation) and **configs** → ported + flattened (D6) in P5/P6.
- `data/{activation_dataset,datamodule}` + `build_feature_dataset` h5 path → replaced by the
  streaming `ActivationStore` in **P5** (not copied).
- Streamlit/plotly **feature viewer** → **deferred to P6** (still needed): re-add
  `visualization/feature_viewer.py` + `streamlit`/`plotly` deps when analysis is ported, or
  pull earlier if useful.
- **Dangling-by-design:** `sae/fidelity.py` & `sae/steering/steer.py` lazy-import the legacy
  `idiom.nn` sampler / `IdiomModel`; importing `idiom.sae` does not trigger these. Rewired in P2.
- Docstring `:mod:`/`:class:` cross-refs still say `idiomatics`/`idiom_interface` — cleaned in
  the P2 model port (cosmetic, no runtime effect).

### P1 — Data (on-the-fly, no token h5)  ◐ (runtime path done; curation drivers operator-run)
- ☑ `data/tokenizer.py`: fixed-alphabet char tokenizer, deterministic id map, `encode/decode`,
  `is_residue`/`is_fim` predicates + `residue_mask` (the residue-only selector for extraction).
- ☑ `data/fim.py`: `fim_full` / `fim_132` + `residue_source_positions` (alignment map for
  dropping markers). Shared by train/infer/RL.
- ☑ `Tokenizer.is_canonical` + hard-failing `encode` — the non-canonical **drop** policy (D15).
- ☑ `data/io.py`: single FASTA reader (`read_fasta`/`read_records`/`parse_idr_header`) that
  drops non-canonical sequences (D15) with a logged count; one `_IDR_x-y` header convention
  shared by record store + inference; `Record` = Option-A `(accession, full_seq, start, end)`.
- ☑ `data/dataset.py`: `RecordDataset` (map-style) → FIM(`full`/`132` augmentation) + tokenize
  + START/shift; `record_to_example` (shared transform), `make_collate` (pad), `max_protein_len`
  (= max_len − 4 for the 3 markers + START). Length-bucketing + sharded/streaming variant for
  the full 37M corpus = follow-up (reuses `record_to_example`).
- ☑ Tests: tokenizer/FIM round-trip, residue-mask, marker-drop alignment, FASTA drop policy,
  header parsing, shift correctness, length filter, collate padding (18/18 CPU green).
- ☑ Lightning `RecordDataModule` over per-split record FASTAs (train/val/test); wraps
  `RecordDataset` + `make_collate`. Tested: builds splits, yields padded START-prefixed batches.
- ◐ `data/curation/`: pure segmentation core `extract.extract_idrs` (half-open spans) **done +
  tested**; `RUNBOOK.md` specifies the offline stages (mmseqs linclust 90/80, split+leakage,
  record-FASTA write, length cap ≤ max_len−4). **No FIM/precompute materialization** (Option A).
  Heavy AFDB-h5 / mmseqs / split drivers are **operator-run** (not in CI) — implement when
  re-curating at ≤1024.

### P2 — Model + KV cache  ☑
- ☑ `model/config.py`: flat `ModelConfig` + `idiom_12l/24l/36l` factories (D5).
- ☑ `model/{rope,norms}.py`: Llama-style offset-aware RoPE + RMSNorm (D17).
- ☑ `model/attention.py`: MHA + QK-norm + RoPE + `KVCache` (prefill + single-token decode), SDPA.
- ☑ `model/transformer.py`: `IDiomTransformer` (SwiGLU pre-norm block, tied head), sequence-only,
  `return_hidden_states` (residual stream per block). RPE path dropped; structural tokens gone.
- ☑ `model/sampling.py`: KV-cached `generate` (START-prefill + incremental decode), `temperature`
  (0=greedy) / `top_k` / `top_p`, per-seq STOP, seeded; `fim_prompt` (`132` / flank-conditioned).
- ☑ `model/activations.py`: `extract_activations` (D14) — residual stream at given layer(s),
  `drop_markers` (residue-only, ids < 20), with `(seq_idx, pos_idx, token_id)` alignment per row;
  shared by SAE stream + user export.
- ☑ Tests: RMSNorm/RoPE, shapes + tied head, **KV-cache == full-forward logits**, **greedy gen ==
  uncached greedy**, seeded sampling, extractor selection/marker-drop/alignment/multi-layer
  (33/33 CPU green).

### ⛳ De-risk gate — train the 12L (GPT-2 small) on new code & confirm sane loss before scaling.  ☐

### P3 — Pretraining + SFT  ◐ (code done; training runs are operator/GPU)
- ☑ `train/lit_autoregressive.py`: `LitAutoregressive` — masked next-token CE, AdamW, **shared by
  pretrain + SFT** (D18). `train/schedulers.py`: warmup-cosine (drops `pl_bolts`).
- ☑ Data emits a per-token **loss mask**: pretraining = all tokens; SFT = IDR completion only
  (`RecordDataset(completion_only=True)`); collate/datamodule thread it through.
- ☑ `train/train.py` Hydra entrypoint (`idiom_train`) — `build`/`run`; SFT = `init_from` warm-start
  + completion-only data. Flat `configs/{pretrain,sft}.yaml` (D6).
- ☑ Tests: masked loss, SFT completion mask, warmup-cosine curve, ckpt warm-start round-trip,
  `build()` wiring, 2-step `Trainer.fit` smoke (39/39 CPU green).
- ☐ **Operator/GPU:** train 12L de-risk → 24L medium → 36L; loss/perplexity + scaling figure.

### P4 — Post-training (GRPO)  ◐ (code done; RL runs are operator/GPU)
- ☑ `train/grpo/core.py`: pure GRPO math — `sequence_logprobs`, `group_advantages`,
  DAPO `grpo_loss` (PPO-clip + Schulman KL). `rewards.py`: registry + fraction/length/entropy
  (`f(idr)->float`); ProtGPS is operator-wired (vendored model).
- ☑ `train/grpo/lit_grpo.py`: `LitGRPO` — online rollouts via the **KV-cached sampler**, frozen
  reference, group-normalized advantages, composite reward. `init_from_checkpoint` warm-start.
- ☑ `train/grpo/data.py`: on-the-fly prompts (`denovo "132"` / per-record flank); no RL h5.
- ☑ `train/grpo/train_grpo.py` (`idiom_grpo`) + flat `configs/grpo.yaml` (sweep-tuned
  length/entropy defaults). Composite reward builder.
- ☑ Tests: rewards, advantages, loss grad, logprobs, full generate→reward→loss step + backprop,
  reward composition, build wiring (47/47 CPU green).
- ☑ **KV-cache profiling (H100):** 12L 3.3× / 24L 4.8× faster generation vs uncached (256 tok,
  B=8); grows with length — the RL-rollout speedup.
- ☐ **Operator/GPU:** per-compartment ProtGPS runs.

### P5 — SAE (streaming)  ☑
- ☑ `SparseCoder`/`LitSAE` (P0 merge) trained on the new model.
- ☑ `sae/activation_store.py`: streaming shuffling buffer over records (on `model/activations.py`),
  residue-only, regenerated each epoch — **no h5**. `model/io.py::load_pretrained` (ckpt loader).
- ☑ `idiom_sae` entrypoint + flat `configs/sae.yaml` (frozen model → store → LitSAE → `ae.pt`).
- ☑ `idiom_extract` + `model/export.py::embed_fasta` (D14): FASTA-first, multi-layer,
  `--pool mean|none`, residue-aligned, written as `.npy` + `index.csv`. Same extractor as the store.
- ☑ Rewired `sae/fidelity.py` + `sae/steering/` onto `IDiomTransformer` (hooks on `model.blocks`,
  `Tokenizer.residue_mask`, steered generation via the KV-cached `generate`). No legacy `idiom.nn`
  refs in `idiom.sae` code. Tested (fidelity runs; steering edits only residues; steer-gen runs).
- ☑ Streamlit feature viewer restored: `analysis/feature_viewer.py` (+ `streamlit` dep).
- ☑ `analysis/build_feature_dataset.py` (`idiom_feature_dataset`): encode residues → top-k SAE
  features + FIM strings, written as a **`.npy` + `.json` directory (no h5)**; `pos_idx` aligns
  to the FIM string. `FeatureDataset` reader switched to that layout; viewer/annotation consume it.
- (Per-layer SAE sweep `layer × k × expansion` is covered by `sae_sweep.bash` / Hydra multirun.)

### P6 — Analysis & figures  ☐
- ☐ Move `idr-plm-figures/figure_scripts` → `analysis/figures/`, parameterized (no hardcoded paths).
- ☐ Interpretability modules: feature ratios, biophysics/SLiM P/R/F1, annotation+HDBSCAN, steering eval.
- ☐ Shared `metrics/`: sparrow, ELM-SLiM, mmseqs novelty, disorder preds.
- ☐ Restore the Streamlit/plotly **feature viewer** (`visualization/feature_viewer.py`) + deps.

## 8. Testing & compute strategy (D9, D13)

- **Always through the scheduler** (shared box — never compute on the login node):
  - CPU: `srun -c 4 --mem 16GB -t 00:30:00 env IDIOM_DEVICE=cpu python -m pytest -q`
  - GPU: add `-G 1` (e.g. `srun -G 1 -c 4 --mem 16GB -t … <cmd>`)
  - interactive: `salloc -G 1 -c 4 --mem 16GB -t 10:00:00` then `srun --pty bash`
- **Scratchpad** for throwaway outputs / quick experiments:
  `/data2/scratch/group_scratch/idr_plm/2026-06-14_refactor`.
- **CPU-first**: `idiom.utils.device.resolve_device()` returns CPU unless CUDA is present and
  not disabled; force CPU anywhere with `IDIOM_DEVICE=cpu`.
- **Tiny configs**: every component has a CPU-runnable tiny config (e.g. 2L/d32) so a full
  train/generate/SAE smoke pass runs on CPU in seconds.
- **`tests/` is CPU-only** and runs in CI; GPU paths are opt-in.
- Each phase lands with tests before the next begins ("implementation correct in steps").

## 9. Open items
- Exact id map for the fixed alphabet (document in `data/tokenizer.py`).
- Length-bucketing vs simple padding at 1024 (perf; decide in P1).
- Whether to keep Hydra structured configs or thin them further (revisit after P3).
- HF org repo names (confirm `rotskoff-group/...`).

## 10. Progress log
- **2026-06-14** — P0 started. Created `refactor/v2`; wrote this plan; scaffolded
  `data/model/train/sae/analysis` packages; added CPU-first `utils/device.py` and a CPU-only
  pytest smoke harness (3/3 pass via `srun`). Decisions D1–D10 locked.
- **2026-06-14** — Added decisions D11–D13: FASTA-first interface, simplicity bias,
  scheduler-only compute. Switched config strategy to flat fat YAMLs (D6). HF org set to
  `jxliu2`. Verified scheduler test protocol (`srun` CPU, no `-G`) works non-interactively.
- **2026-06-14** — **P0 complete.** Merged idiomatics SAE/analysis core into `idiom.sae` /
  `idiom.analysis` / `idiom.data.tokens` with imports rewired; dropped the `idiom_interface`
  loader; unioned dependencies into one `pyproject` (+sparrow). 5/5 CPU tests green via `srun`.
  rewards/scripts/configs/ActivationStore deferred to their phases (see P0 deferred list).
- **2026-06-14** — Data schema decided: **Option A** (store `full_seq` + `(idr_start,idr_end)`;
  prefix/idr/suffix sliced at load time). Added **D14**: one hook-based activation-extraction
  core feeding both the SAE stream and a FASTA-first user export (`idiom extract` / `.embed`,
  multi-layer, `--pool mean|none`, safetensors/npy). Restored feature viewer to P6 (not dropped).
- **2026-06-14** — **P1 started.** Wrote `data/tokenizer.py` (fixed 27-token alphabet —
  residues 0..19, FIM 20..22, controls 23..26; `is_residue`/`residue_mask` for marker-drop)
  and `data/fim.py` (`fim_full`/`fim_132` + `residue_source_positions` alignment map). Tests
  cover encode/decode, residue masking, FIM correctness, and marker-drop→residue alignment
  (full & 132). 11/11 CPU tests green via `srun`. Remaining P1: record dataset, DataModule, curation.
- **2026-06-14** — Added **D15** (non-canonical residues → **drop whole sequence**):
  `Tokenizer.is_canonical` + hard-failing `encode`; one shared policy for curation, training,
  and inference/extract, enforced in the upcoming `data/io.py` FASTA reader. 12/12 CPU tests green.
- **2026-06-14** — `data/io.py` (FASTA reader + drop policy + `_IDR_x-y` parsing → `Record`)
  and `data/dataset.py` (`RecordDataset` + `record_to_example` + `make_collate`) done; map-style
  for now, streaming/sharding deferred. 18/18 CPU tests green via `srun`. (Installed `loguru`
  into the venv; full `uv sync` of the heavy deps — sparrow/umap/anthropic — deferred until
  their phases need them.)
- **2026-06-14** — Added **D16** (coordinate convention): header **1-based inclusive**, internal
  `Record` **0-based half-open**. Refactored `fim.py` / `io.py` (and tests) to half-open internals
  (dropped the `+1`s; `idr = full_seq[start:end]`). 18/18 CPU tests green.
- **2026-06-14** — `data/datamodule.py` (`RecordDataModule`, per-split FASTAs) done + tested.
  Curation: `extract.extract_idrs` (pLDDT→IDR, half-open) done + tested; `curation/RUNBOOK.md`
  specifies the offline stages (no FIM materialization, ≤max_len−4 cap). Heavy drivers left
  operator-run (not executed, per request). **Runtime data path complete; 22/22 CPU tests green.**
- **2026-06-14** — **P2 model core landed.** Locked **D17** (RMSNorm + SwiGLU + QK-norm + RoPE +
  no-bias + tied; modernizes legacy LayerNorm→RMSNorm) and **D5** sizes (12L→24L→36L). Wrote
  `model/{config,rope,norms,attention,transformer}.py` (sequence-only, KV cache, SDPA causal).
  Critical **KV-cache == full-forward** equivalence test passes. 26/26 CPU green. Remaining P2:
  `sampling.py` (KV-cached generate) + `activations.py` (D14 extractor).
- **2026-06-14** — **P2 complete.** `model/sampling.py` (KV-cached `generate`, temp/top-k/top-p,
  STOP, seeded) + `data/fim.py::fim_prompt`; `model/activations.py` (D14 extractor: residual
  stream, residue-only marker-drop, per-row alignment). Tests: greedy-gen == uncached greedy,
  seeded sampling, extractor selection/alignment/multi-layer. **33/33 CPU green.** Switched the
  12L config to GPT-2 small (d768/12h) — sizes are now the exact GPT-2 family (D5).
- **2026-06-14** — **P3 code done + SFT (D18).** `LitAutoregressive` (masked CE, AdamW) shared by
  pretrain + SFT; `warmup_cosine` (drops pl_bolts); data emits a loss mask (all tokens vs IDR-only
  completion); `idiom_train` Hydra entrypoint + flat `pretrain.yaml`/`sft.yaml`. Tests: masked
  loss, SFT mask, scheduler, ckpt warm-start, build wiring, fit smoke. **39/39 CPU green.** Actual
  training runs are operator/GPU.
- **2026-06-14** — **P4 GRPO code done.** Clean reimpl: `grpo/core.py` (logprobs, group
  advantages, DAPO loss), `grpo/rewards.py` (registry, `f(idr)->float`), `grpo/lit_grpo.py`
  (`LitGRPO`, rollouts via KV-cached sampler, frozen ref), `grpo/data.py` (on-the-fly prompts),
  `idiom_grpo` entrypoint + `grpo.yaml`. **47/47 CPU green.** KV-cache profiling on H100:
  **3.3× (12L) / 4.8× (24L)** vs uncached generation. ProtGPS reward + RL runs are operator/GPU.
- **2026-06-14** — **P5 streaming SAE + export.** `sae/activation_store.py` (shuffling buffer over
  records, residue-only, no h5) + `model/io.py::load_pretrained`; `idiom_sae` entrypoint + `sae.yaml`.
  `model/export.py::embed_fasta` + `idiom_extract` (D14, FASTA→`.npy`+index, mean/per-residue,
  residue-aligned). **55/55 CPU green.** Remaining P5: fidelity/steering rewire to the new model.
- **2026-06-14** — Rewired `sae/fidelity.py` + `sae/steering/` onto `IDiomTransformer` (no legacy
  `idiom.nn` refs left in `idiom.sae`); restored the Streamlit `feature_viewer`. **Corrected GRPO
  rewards to the legacy quadratic** length/entropy penalties + `-scale*(raw-target)^2` shaping
  (D19 — match the known-good RL). Added `D19` (keep legacy for reference). **58/58 CPU green.**
  Wrote example SLURM scripts (pretrain/sft/grpo/sae/sae_sweep/extract, every Hydra param explicit)
  to `…/2026-06-14_refactor/bash_scripts`.
- **2026-06-14** — **P5 complete.** `analysis/build_feature_dataset.py` (`idiom_feature_dataset`):
  residues → top-k SAE features + FIM strings as a `.npy`+`.json` directory (no h5); `FeatureDataset`
  reader switched to it (mmap), `pos_idx` residue-aligned. Viewer/annotation now have their input.
  **61/61 CPU green.** Next: P6 (analysis & figures).
- **2026-06-14** — **Repo reorg (D20).** Folded `src/idiom/analysis/` → `src/idiom/sae/`
  (feature_activations/build_feature_dataset/feature_viewer). Moved `src/idiom/data/curation/`
  → top-level **`data_pipeline/`** (pretraining corpus build). Created top-level **`analysis/`**
  (paper bucket). pyproject: repointed `idiom_feature_dataset`, pytest `pythonpath=[src,.]`,
  wheel ships only `src/idiom`. Artifacts (`models/`/`datasets/`) → gitignored, HF-hosted.
  61/61 CPU green.
- **2026-06-14** — Moved `models/` (27G) + `datasets/` (177G) out of the repo to scratch
  safekeeping; removed their `.gitignore` blocks (repo is now pure code; artifacts → HF).
- **2026-06-14** — Scaffolded P6 **figure plumbing**: `analysis/figures/{idiom.mplstyle,_style.py}`
  (`use_style` + `save_fig` writing to `$IDIOM_FIG_DIR` = the overleaf `figs/`, filename = LaTeX
  `\includegraphics` path) + `bash/figures.bash`. Figure scripts live in `analysis/figures/`,
  output into the manuscript repo. 64/64 CPU green.
- **2026-06-14** — **Legacy removed** (backed up to `tmp/idiom_backup_2026-06-14` + git history).
  Deleted `src/idiom/nn/`, `src/idiom/scripts/`, legacy `utils/{data_utils,misc,sampler,token}.py`,
  root `entrypoints/`, `rewards/custom_rewards/` (kept `rewards/protgps/`). Dropped 7 stale
  `transformer_*`/`make_*` console-scripts and the `lightning-bolts` + `torch-geometric` deps.
  Scrubbed stale `idiom.nn`/`h5py`/`pl_bolts` docstring refs. Repo is now v2-only; 64/64 CPU green.
