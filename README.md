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

model = IDiom.from_pretrained("jxliu2/idiom-24l")   # HF repo id or a local directory

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
idiom_generate unprompted --model jxliu2/idiom-24l --n 1000 --out idrs.fasta
idiom_generate prompted   --model jxliu2/idiom-24l --fasta proteins.fasta --n 1000 --out idrs.fasta
idiom_extract --ckpt model.ckpt --fasta proteins.fasta --layers 18 --out embeddings/
```

## 2. Interpretability and steering (IDiomSAE)

`IDiomSAE` bundles a trained SAE with its **host model** and the **layer** it reads, so it always
runs on the distribution it was trained on (its `region` and `fim_mode` are applied automatically).

```python
from idiom import IDiomSAE

sae = IDiomSAE.from_pretrained("jxliu2/idiom-sae-24l-L18")   # host model auto-loaded

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
idiom_feature_dataset --sae jxliu2/idiom-sae-24l-L18 --fasta records.fasta --out features/
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

**GRPO post-training** optimizes any reward over generated IDRs. A reward is just
`f(idr: str) -> float` registered by name, so adding your own is a few lines:

```python
from idiom.train.grpo.rewards import register_reward

@register_reward("aromatic_fraction")
def aromatic_fraction(idr: str) -> float:
    return sum(idr.count(a) for a in "FWY") / len(idr) if idr else 0.0
```

```bash
idiom_grpo init_from=/path/base.ckpt \
  reward.module=rewards/example_rewards.py reward.name=aromatic_fraction
```

The composite reward adds optional **length** and **entropy** shaping terms (the entropy term is the
naturalness guardrail against low-complexity reward hacking), plus an optional `monitor` reward that
is logged but not optimized. See `src/idiom/configs/grpo.yaml`.

**Group rewards** score each completion *relative to its GRPO group* rather than on its own —
register with `@register_group_reward` (`f(idrs, group_size) -> list[float]`) and select via
`reward.group=`.

**RL toward SAE features (RL-SAE).** `rewards/sae_feature_reward.py` rewards a model for reproducing
a target's interpretable SAE feature code, using a frozen base + SAE as a fixed lens — so a reward
gain requires encoding the real code, not just satisfying a classifier:

```bash
# ProtGPS classifier + lambda * (fraction of the target's features that fire)
idiom_grpo init_from=... reward.module=rewards/sae_feature_reward.py \
  reward.name=protgps_feat_nucleolus

# population coverage of the feature signature across each GRPO group
idiom_grpo init_from=... reward.module=rewards/sae_feature_reward.py \
  reward.group=sae_coverage_nucleolus
```

`sae_only_<target>` optimizes the feature code alone (no classifier in the loop). Compartment
localization rewards (`protgps_<compartment>`, and the selectivity variants `protgps_sel_*` /
`protgps_anchor_*`) live in `rewards/protgps_reward.py`. Reward hyperparameters are read from
environment variables (`IDIOM_SAEREWARD_LAMBDA`, `IDIOM_SAEREWARD_SAE`, ...) since the reward
contract is `f(idr) -> float`.

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

Weights are loaded by repo id or local path — `IDiom.from_pretrained` and
`IDiomSAE.from_pretrained` accept either.

| Repo | Contents |
|------|----------|
| `jxliu2/idiom-24l`, `jxliu2/idiom-12l`, `jxliu2/idiom-6l` | base models |
| `jxliu2/idiom-sae-24l-L18` | sparse autoencoders (one repo per host model + layer) |
| `jxliu2/idiom-24l-rl-<target>` | RL post-trained checkpoints |
| `jxliu2/idiom-datasets` | curated IDR corpus, eval sets, generated sequences, feature datasets |

*(Uploads in progress.)* A released SAE records its host model, so `IDiomSAE.from_pretrained`
loads the pair with one call.

## Repository layout

| Path | Role |
|------|------|
| `src/idiom/` | the library: `data` (tokenizer/FIM/dataset), `model` (transformer + KV cache + sampling), `train` (pretrain/SFT/GRPO), `sae` (SAEs + steering + features + eval), `utils`, public `IDiom`/`IDiomSAE` API |
| `rewards/` | GRPO reward definitions + the vendored ProtGPS reward model |
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
