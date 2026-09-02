# IDiom

IDiom is an autoregressive transformer for generating, designing, and studying intrinsically disordered protein regions (IDRs). Trained on 54M IDRs curated from the AlphaFold Database with a fill-in-the-middle objective, IDiom can generate IDRs **unprompted** (de novo, with no flanking context) as well as **prompted** (conditioned on their flanking context). The model can also be post-trained with reinforcement learning to optimize for custom reward functions.

This work additionally presents IDiomSAE, sparse TopK autoencoders trained on the residual stream of IDiom. IDiomSAE enables us to mechanistically interpret the features learned by IDiom, as well as to causally steer the model during generation.

The associated preprint is: [Generative design of intrinsically disordered protein regions with IDiom](https://doi.org/10.64898/2026.04.10.717777)

![IDiom](assets/github_fig.png)

The repository supports three things equally: **generation**, **SAE interpretability and steering**, and **training / RL post-training**.

## Install

```bash
git clone https://github.com/rotskoff-group/idiom.git
cd idiom
uv sync          # creates .venv, installs dependencies and the idiom package (with its CLIs)
```

Then either activate the environment (`source .venv/bin/activate`) or prefix commands with
`uv run` (e.g. `uv run idiom_generate ...`). Python >= 3.10.

## Sequence conventions (read this first)

IDiom is a fill-in-the-middle model, so **every input carries an IDR span**. Getting this wrong
does not raise — it silently produces off-distribution output.

- **FASTA headers must end with `_IDR_x-y`** (1-based, inclusive) marking the IDR within the
  sequence, e.g. `>P06748_IDR_119-242`. For a fully disordered sequence, the span covers the whole
  sequence (`_IDR_1-<len>`).
- **Python coordinates are 0-based, half-open** (`idr = seq[idr_start:idr_end]`), matching Python
  slicing. The 1-based header form is converted on read.
- **Unprompted vs prompted**: unprompted generation uses the FIM prompt `132` (no flanks); prompted
  generation conditions on the flanks as `1{prefix}3{suffix}2`. Older releases called these
  IDP / context-IDR; the old names remain as deprecated aliases.
- Sequences must use the **20 canonical amino acids**. Non-canonical entries in a FASTA are dropped
  (with a logged count); a non-canonical sequence passed explicitly raises.

Functions that take sequences (`embed`, `encode`, `fidelity`, `build_feature_dataset`) accept a
FASTA path, a single sequence string, or a list of sequences. A bare sequence string is treated as
an unprompted IDR (the whole sequence is the IDR).

## 1. Generation and embeddings

```python
from idiom import IDiom

model = IDiom.from_pretrained("jxliu2/idiom-300M")   # HF repo id or a local directory

# Unprompted IDRs (de novo, no flanking context)
idrs = model.generate_unprompted(n=100, temperature=1.0)

# ... within a target length range (oversamples and length-filters)
idrs = model.generate_unprompted(n=100, length_range=(80, 120))

# Prompted IDRs, conditioned on flanking context (0-based, half-open coords)
idrs = model.generate_prompted(protein_seq, idr_start, idr_end, n=100)

# Residual-stream embeddings: a FASTA, one sequence, or a list of sequences
values, index = model.embed(["MKKLVA...", "GSGSQP..."], layers=[18], pool="none")[18]
```

FASTA-first wrappers (`generate_unprompted_fasta`, `generate_prompted_fasta`) write valid record
FASTAs, and the same is available from the command line:

```bash
idiom_generate unprompted --model jxliu2/idiom-300M --n 1000 --out idrs.fasta
idiom_generate prompted   --model jxliu2/idiom-300M --fasta proteins.fasta --n 1000 --out idrs.fasta
idiom_extract --ckpt model.ckpt --fasta proteins.fasta --layers 18 --out embeddings/
```

## 2. Interpretability and steering (IDiomSAE)

`IDiomSAE` bundles a trained SAE with its **host model** and the **layer** it reads, so it always
runs on the distribution it was trained on (its `region` and `fim_mode` are applied automatically).

```python
from idiom import IDiomSAE

sae = IDiomSAE.from_pretrained("jxliu2/idiomsae-300M-L18-k32")   # host model auto-loaded

# Per-sequence (or per-residue) feature activations
feats, accessions = sae.encode("proteins.fasta", pool="mean")
feats, index      = sae.encode("proteins.fasta", pool="none")   # per-residue rows

# Causal steering: push generation along a feature direction
seqs = sae.steer_generate(feature=1234, strength=0.5, n=100)
seqs = sae.steer_generate(feature=[12, 44], strength=0.3, relative=True, preserve_norm=True)

# How much of the model's behaviour the SAE preserves (Gao "loss recovered")
fid = sae.fidelity("records.fasta")
print(fid.pct_loss_recovered)
```

Steering modes: `add_direction` (default), `clamp`, and `ablate`. `normalize` makes `strength` a
magnitude in residual-norm units; `relative` makes it a fraction of the local residual norm; and
`preserve_norm` rotates toward the feature at constant residual norm instead of inflating it.

Offline feature analysis — build a per-residue feature-activation dataset, then browse it:

```bash
idiom_feature_dataset --sae jxliu2/idiomsae-300M-L18-k32 --fasta records.fasta --out features/
streamlit run src/idiom/sae/features/feature_viewer.py -- --features features/
```

Train your own SAE on any layer (streaming activations, no cached activations on disk):

```bash
idiom_sae model_ckpt=/path/model.ckpt data.fasta=/path/records.fasta layer=18 sae.k=32
```

Reconstruction/sparsity metrics (FVU, explained variance, mean L0, dead fraction, per-feature
firing frequency) are available as library primitives in `idiom.sae.eval`.

## 3. Training and RL post-training

All training entrypoints are Hydra CLIs; configs are flat YAMLs in `src/idiom/configs/`, overridden
on the command line.

**Pretraining and SFT** share one LightningModule — the only difference is the loss mask
(pretraining trains on every token; SFT trains only on the IDR completion):

```bash
idiom_build_store --fasta corpus.fasta            # memory-mapped record store (recommended at scale)
idiom_train data.train_fasta=corpus.fasta model.n_layers=24 model.d_model=1024
idiom_train --config-name sft init_from=/path/base.ckpt data.train_fasta=sft.fasta
```

**GRPO post-training** optimizes a reward over generated IDRs. The reward is a **weighted sum of
terms** — entropy and length guardrails, an RL-SAE feature-code reward, and any number of external
reward models — configured in `src/idiom/configs/grpo.yaml`:

```yaml
reward:
  entropy: {enabled: true,  weight: 1.0, target_entropy: 3.68, width: 0.2}   # naturalness guardrail
  length:  {enabled: true,  weight: 1.0, target_length: 100, width: 1.0}
  rl_sae:  {enabled: false, weight: 1.0, signature: nucleolus}
  external: []
```

`total = Σ weightᵢ · termᵢ`, and that total drives the GRPO advantages. Toggle a term with
`enabled`, scale it with `weight`.

**RL toward SAE features (RL-SAE).** The `rl_sae` term rewards a model for reproducing a target's
interpretable SAE feature code, scored through a frozen IDiom base + SAE as a fixed lens — so a
reward gain requires encoding the real code, not merely satisfying a classifier. The reward model is
IDiom itself, so it needs no third-party dependency and runs straight after `uv sync`:

```bash
idiom_grpo init_from=/path/base.ckpt reward.rl_sae.enabled=true reward.rl_sae.signature=nucleolus
```

Signatures ship in `rewards/signatures/` for the released SAE (cases `top30` and `private30`, select
with `IDIOM_SAEREWARD_CASE`). **Build a signature from your own sequences** with
`examples/05_feature_enrichment.py` and point `IDIOM_SAEREWARD_FEATURES` at it.

**Bring your own reward model.** Each `external` term is either a simple in-process Python function,
or a command that runs a reward model in its own environment (for one whose dependencies conflict
with IDiom's — a different python, torch, or CUDA). For the in-process case, copy
`rewards/builtin_rewards.py`. For the subprocess case there is no install step — let uv build and
cache the environment on demand, so the config is all you write:

```yaml
# design IDRs with a radius of gyration near 25 A, scored by sparrow in its own environment
reward.external:
  - {enabled: true, weight: 0.5, target: 25, width: 3,
     cmd: "uv run --isolated --no-project --with 'sparrow @ git+https://github.com/idptools/sparrow.git' python rewards/scorers/sparrow.py --property radius_of_gyration"}
```

```bash
# point uv's cache at scratch (it is several GB), then verify before spending a GPU allocation
export UV_CACHE_DIR=/scratch/you/uv-cache
python -m idiom.train.grpo.external \
  --cmd "uv run --isolated --no-project --with 'sparrow @ git+https://github.com/idptools/sparrow.git' python rewards/scorers/sparrow.py --property radius_of_gyration" \
  --target 25 --width 3
```

uv builds the environment once at startup (about 30s for sparrow, which needs a C compiler; every
run after is a cache hit); pin `@<commit>` for a reproducible build. The scorer is a fifteen-line
program that reads `{"sequences": [...]}` from stdin and writes `{"scores": [...]}` to stdout,
importing nothing from IDiom — so the same `cmd` form covers an on-demand uv env, a pre-built venv, a
conda env, or a container. Because the command lives in the config, several external rewards, each
its own environment and target, combine in one run. [`rewards/README.md`](rewards/README.md) has the
details, with [sparrow](https://github.com/idptools/sparrow) (biophysics: Rᵧ, asphericity, charge
patterning) as the worked example.

## Command-line reference

| Command | Does |
|---------|------|
| `idiom_generate` | generate unprompted/prompted IDRs to a FASTA |
| `idiom_extract` | export residual-stream embeddings from a FASTA |
| `idiom_train` | pretrain (and SFT via `--config-name sft`) |
| `idiom_grpo` | GRPO / RL post-training against a reward |
| `idiom_sae` | train a top-k SAE on a layer (streaming activations) |
| `idiom_feature_dataset` | build the per-residue SAE feature dataset |
| `idiom_build_store` | build a memory-mapped record store from a record FASTA |

## Models and data (HuggingFace)

Weights load by repo id or local path — `IDiom.from_pretrained` and `IDiomSAE.from_pretrained`
accept either.

### Models

| Repo | Params | Architecture |
|------|--------|--------------|
| [`jxliu2/idiom-300M`](https://huggingface.co/jxliu2/idiom-300M) | 302M | 24 layers, d_model 1024 |
| [`jxliu2/idiom-85M`](https://huggingface.co/jxliu2/idiom-85M) | 85M | 12 layers, d_model 768 |
| [`jxliu2/idiom-20M`](https://huggingface.co/jxliu2/idiom-20M) | 18.9M | 6 layers, d_model 512 |

### Sparse autoencoders

| Repo | Host model | Configuration |
|------|-----------|---------------|
| [`jxliu2/idiomsae-300M-L18-k32`](https://huggingface.co/jxliu2/idiomsae-300M-L18-k32) | `idiom-300M`, layer 18 | 16,384 latents (expansion 16), TopK k=32, IDR residues, unprompted |

A released SAE records its host model, so `IDiomSAE.from_pretrained` loads the pair in one call.

### Data

[`jxliu2/idiom-data`](https://huggingface.co/datasets/jxliu2/idiom-data) holds plain FASTA files for
direct download (it is not a `datasets`-loadable dataset):

| Path | Contents |
|------|----------|
| `training_sequences/` | the curated IDR corpus IDiom was pretrained on — `train.fasta` (~53.6M records, 23.6 GB), `validation.fasta` (270,775), `test.fasta` (270,777) |
| `reference_sequences/` | reference and held-out evaluation sets *(coming soon)* |
| `generated_sequences/` | sequences generated by released models *(coming soon)* |

```bash
hf download jxliu2/idiom-data --repo-type dataset --include "training_sequences/*"
```

The corpus is **54,155,136 IDR records** curated from the AlphaFold Database (pLDDT segmentation,
90% clustering, length <= 1020, dedup, DisProt holdout removal, SignalP-6 filtering). It is released
under CC BY 4.0, inherited from AlphaFold DB / UniProt; the code in this repository is MIT.

## Repository layout

| Path | Role |
|------|------|
| `src/idiom/` | the library: `data` (tokenizer/FIM/dataset), `model` (transformer + KV cache + sampling), `train` (pretrain/SFT/GRPO), `sae` (SAEs + steering + features + eval), `utils`, public `IDiom`/`IDiomSAE` API |
| `rewards/` | GRPO reward definitions: RL-SAE, custom in-process rewards, and `scorers/` (external models) |
| `examples/` | short runnable scripts: generation, embeddings, SAE features, custom rewards, feature enrichment |
| `assets/` | static assets (figures for docs) |
| `tests/` | unit/integration tests for the library |

## Citation

```bibtex
@article{liu2026idiom,
  author = {Liu, Jason and Ibarraran, Sebastian and Hu, Frank and Park, Abigail and Dunn, Alexander and Rotskoff, Grant},
  title = {Generative design of intrinsically disordered protein regions with {IDiom}},
  journal = {bioRxiv},
  year = {2026},
  doi = {10.64898/2026.04.10.717777},
  URL = {https://doi.org/10.64898/2026.04.10.717777},
}
```

## License

MIT — see [LICENSE](LICENSE).
