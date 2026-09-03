# IDiom

IDiom is an autoregressive transformer for generating, designing, and studying intrinsically disordered protein regions (IDRs). Trained on 54M IDRs curated from the AlphaFold Database with a fill-in-the-middle objective, IDiom can generate IDRs **unprompted** (de novo, with no flanking context) as well as **prompted** (conditioned on their flanking context). The model can also be post-trained with reinforcement learning to optimize for custom reward functions.

This work additionally presents IDiomSAE, sparse TopK autoencoders trained on the residual stream of IDiom. IDiomSAE enables us to mechanistically interpret the features learned by IDiom, as well as to causally steer the model during generation.

The associated preprint is: [Generative design of intrinsically disordered protein regions with IDiom](https://doi.org/10.64898/2026.04.10.717777)

![IDiom](assets/github_fig.png)

The repository supports three things equally: **generation**, **SAE interpretability and steering**, and **training / RL post-training**.

## Install

Install the package to use IDiom — generation, embeddings, SAEs, and all the training entrypoints
work from this alone:

```bash
pip install git+https://github.com/rotskoff-group/idiom.git      # or: uv pip install git+...
```

Clone instead if you want the cookbook (Slurm templates, example data, walkthrough scripts), the
tests, or to modify IDiom itself:

```bash
git clone https://github.com/rotskoff-group/idiom.git
cd idiom
uv sync          # creates .venv, installs dependencies and the idiom package (with its CLIs)
```

Then either activate the environment (`source .venv/bin/activate`) or prefix commands with
`uv run` (e.g. `uv run idiom_generate ...`). Python >= 3.10.

**A note on torch.** IDiom requires `torch>=2.4` but does not pin a build, because the right one
depends on your driver. A clone gets torch 2.4.0+cu121 from `uv.lock`; a plain `pip install`
resolves the newest compatible torch, which on an older driver fails at first use with *"The NVIDIA
driver on your system is too old"*. If that happens, install the build matching your CUDA from
[pytorch.org](https://pytorch.org/get-started/locally/) first, then install IDiom.

## Sequence conventions (read this first)

IDiom is a fill-in-the-middle model, so **every input carries an IDR span**. Getting this wrong
does not raise — it silently produces off-distribution output.

- **FASTA headers must end with `_IDR_x-y`** (1-based, inclusive) marking the IDR within the
  sequence, e.g. `>P06748_IDR_119-242`. For a fully disordered sequence, the span covers the whole
  sequence (`_IDR_1-<len>`).
- **Python coordinates are 0-based, half-open** (`idr = seq[idr_start:idr_end]`), matching Python
  slicing. The 1-based header form is converted on read.
- **Unprompted vs prompted** is a property of the *prompt*, not of the model: unprompted generation
  uses the FIM prompt `132` (no flanks), prompted generation conditions on the flanks as
  `1{prefix}3{suffix}2`. One model does both.
- Sequences must use the **20 canonical amino acids**. Non-canonical entries in a FASTA are dropped
  (with a logged count); a non-canonical sequence passed explicitly raises, so an input you named
  yourself is never silently discarded.

Three things in this repo are spelled "IDR" and they are *not* the same axis — worth reading once,
because mixing them up is the easiest way to get quiet nonsense:

| | Values | Means |
|---|---|---|
| the header span | `_IDR_x-y` | which residues of a protein are disordered |
| `fim_mode` / prompting | `prompted` / `unprompted` | whether the prompt carries the flanks |
| SAE `region` | `all` / `idr` / `non_idr` | which residues an SAE reads and edits |

Functions that take sequences (`embed`, `encode`, `build_feature_dataset`) accept a
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

## 3. Training and RL post-training

All training entrypoints are Hydra CLIs; configs are flat YAMLs in `src/idiom/configs/`, overridden
on the command line.

**Pretraining and SFT** share one LightningModule — the only difference is the loss mask
(pretraining trains on every token; SFT trains only on the IDR completion):

```bash
idiom_build_store --fasta corpus.fasta            # memory-mapped record store (recommended at scale)
idiom_train data.train_fasta=corpus.fasta model.n_layers=24 model.d_model=1024
idiom_train --config-name sft init_from=jxliu2/idiom-300M data.train_fasta=sft.fasta   # init_from: HF repo id, released dir, or .ckpt
```

**GRPO post-training** optimizes a reward over generated IDRs. A reward term is a **reward** and its
**shaping**: the reward reports one raw value in its own units — bits, residues, angstroms, a
fraction — and the shaping says what a good value is. The total is the weighted sum of the shaped
rewards, configured in `src/idiom/configs/grpo.yaml`:

```yaml
reward:
  terms:
    - {reward: entropy, weight: 1.0, shaping: {type: quadratic, target: 3.65, width: 0.2}}  # bits
    - {reward: length,  weight: 1.0, shaping: {type: quadratic, target: 100,  width: 1.0}}  # residues
```

`total = Σ weightᵢ · shapingᵢ(rewardᵢ)`, and that total drives the GRPO advantages. The split is what
keeps the config small: any reward can be aimed at a target or used raw without being rewritten,
and the same one can appear twice under different labels. Shaping is
`quadratic` (0 at the target, −1 one `width` out, unbounded below), `gaussian` (the bounded version,
for when several targets have to coexist), or omitted, which uses the raw value — what the RL-SAE
term does, its raw reward already being a fraction in [0, 1].

Two ways to take a term out, and they differ: `weight: 0` keeps it running and logged (watch a
quantity without optimizing it), while `enabled: false` skips it entirely — nothing imported, no
subprocess, no per-step cost. The shipped `grpo.yaml` uses the latter to carry a menu of ready-made
terms (RL-SAE, an in-process reward, an external scorer) that cost nothing until you switch one on:

```bash
idiom_grpo init_from=jxliu2/idiom-300M reward.terms.2.enabled=true   # term 2 is the RL-SAE one
```

Every enabled term is logged twice — `<label>` is what it contributed to the objective,
`<label>_raw` its raw reward — so a length term reads 98 residues alongside its penalty. A
malformed term fails when the config is parsed, before the model loads; a disabled one is not
validated at all, which is what lets the menu name things this environment cannot import.

**RL toward SAE features (RL-SAE).** The `sae_only_<signature>` rewards score a model on
reproducing a target's interpretable SAE feature code, scored through a frozen IDiom base + SAE as a
fixed lens — so a reward gain requires encoding the real code, not merely satisfying a classifier.
The reward model is IDiom itself, so it needs no third-party dependency and runs straight after
`uv sync`. The raw reward is already a fraction in [0, 1] and needs no shaping; `module` imports the
lens on demand, which is what keeps it out of runs that leave the term off. It ships as term 2 of
the default config, switched off:

```bash
idiom_grpo init_from=jxliu2/idiom-300M \
  reward.terms.2.enabled=true reward.terms.2.reward=sae_only_nucleolus
```

Signatures ship in `idiom/rewards/rl_sae_targets/` for the released SAE (cases `top30` and `private30`, select
with `IDIOM_SAEREWARD_CASE`). **Build a signature from your own sequences** with
`cookbook/scripts/feature_enrichment.py` and point `IDIOM_SAEREWARD_FEATURES` at it.

**Bring your own reward model.** A term names either a registered in-process reward or a command
that runs a reward model in its own environment (for one whose dependencies conflict with IDiom's —
a different python, torch, or CUDA). The reward content ships inside the package (`idiom/rewards/`): `custom_rewards.py` (copy-me
in-process rewards), `external_rewards/` (external programs), and `rl_sae_targets/` (SAE
signatures). Configs address it with the `${idiom_rewards:...}` resolver, so the paths work from any
working directory and in any install. They are reference examples: to write your own, put a
file anywhere and name it in the term (see **Extending IDiom**), rather than editing these.

*In-process* — register an `f(idr) -> float` in a module and name it (copy the pattern from `idiom/rewards/custom_rewards.py`):

```yaml
reward.terms:
  - {reward: net_charge_fraction, module: "${idiom_rewards:custom_rewards.py}", weight: 1.0,
     shaping: {type: gaussian, target: 0.25, width: 0.5}}
```

*A command in its own environment* — no install step. The scorer declares its own dependencies in a
[PEP 723](https://peps.python.org/pep-0723/) header, so uv builds and caches that environment on
demand and the config only names the script:

```yaml
# design IDRs with a radius of gyration near 25 A, scored by sparrow in its own environment
reward.terms:
  - {cmd: "uv run --script ${idiom_rewards:external_rewards/sparrow.py} --property radius_of_gyration",
     label: rg, weight: 0.5, shaping: {type: quadratic, target: 25, width: 0.2}}
```

Six scorers ship in `idiom/rewards/external_rewards/`, each self-contained — the PEP 723 header is the
whole environment, and any model weights are fetched on first use, so a fresh clone needs no setup
step:

| Scorer | Reward | Environment and weights | Cost per 32 sequences |
|---|---|---|---|
| [`sparrow.py`](src/idiom/rewards/external_rewards/sparrow.py) | single-chain biophysics: radius of gyration, asphericity, scaling exponent, FCR, kappa | [sparrow](https://github.com/idptools/sparrow) from git; no weights | 2.8 s (CPU) |
| [`finches.py`](src/idiom/rewards/external_rewards/finches.py) | epsilon interaction parameter, self or against a `--partner` sequence | [finches](https://github.com/idptools/finches) from git; forcefield parameters ship in the package | 0.1 s (CPU) |
| [`protgps.py`](src/idiom/rewards/external_rewards/protgps.py) | condensate compartment probability (ESM-2 classifier) | [ProtGPS](https://github.com/pgmikhael/protgps) on python 3.8 / torch 2.0; 166 MB of weights from Zenodo (CC BY 4.0) | 1.6 s (CPU) |
| [`paddle.py`](src/idiom/rewards/external_rewards/paddle.py) | transcriptional activation strength, max-Z over 53-residue windows | [PADDLE](https://github.com/asanborn/PADDLE) on TensorFlow; 36 MB of models cloned from GitHub (Apache-2.0) | 3 s (CPU) |
| [`starling.py`](src/idiom/rewards/external_rewards/starling.py) | ensemble radius of gyration or end-to-end distance, from a generated conformational ensemble | [STARLING](https://github.com/idptools/starling) from PyPI; 1.5 GB of weights auto-downloaded | 9 s (GPU) |
| [`pspred.py`](src/idiom/rewards/external_rewards/pspred.py) | phase-separation thermodynamics: transfer free energy in kT, or saturation concentration in mg/mL | [PSpred](https://github.com/KULL-Centre/_2024_buelow_PSpred) scripts and models (3.7 MB) fetched from GitHub | 4.7 s (CPU) |

They cover five different notions of "good": single-chain biophysics, interaction chemistry,
phase-separation thermodynamics, a learned classifier, and an experimental activation assay. A cold machine spends about 2 GB and a few minutes on the first
run of each; everything after is a cache hit.

`cmd` also takes a list of arguments (`[python, /path/my scorer.py, --flag, value]`) when shell
quoting gets in the way. The scorer returns a **raw reward** and stops there; shaping it stays on the
IDiom side, so the objective is retuned without touching that environment. Because each command
lives in the config, several external rewards — each its own environment and target — combine in one
run. Point uv's cache at scratch and verify a command before spending a GPU allocation:

```bash
export UV_CACHE_DIR=/scratch/you/uv-cache   # several GB; keep it off your home directory
uv run python -m idiom.train.grpo.reward.external \
  --cmd "uv run --script ${idiom_rewards:external_rewards/sparrow.py} --property radius_of_gyration" \
  --shaping quadratic --target 25 --width 0.2
```

uv builds the environment once at the startup handshake (~30s for sparrow, which needs a C compiler;
every run after is a cache hit); pin `@<commit>` in the script's header for a reproducible build.

**Writing a scorer.** A scorer is a standalone program — copy `idiom/rewards/external_rewards/sparrow.py` — that
speaks newline-delimited JSON on stdin/stdout, one exchange per GRPO step, importing nothing from
IDiom:

```
->  {"sequences": ["ACDEF...", "GHIKL..."]}
<-  {"scores": [24.8, 31.2]}          # or {"error": "..."}
```

Return one finite score per sequence, in order (a count mismatch is rejected, so misaligned rewards
can't silently corrupt training); flush after each response; and load the model once at import (the
process is reused for the whole run).

**stdout is the protocol.** Many model libraries print on import or first load — ProtGPS writes
`Using ESM hidden layers 6`, STARLING writes `Using DDIM sampler`, TensorFlow announces itself — and
a single stray line there is read as a malformed response, so the run dies at the handshake with
`scorer wrote a non-JSON line`. Take the real stdout for yourself and send everything else to stderr,
which the parent forwards to its log with the term's label:

```python
_PROTOCOL_STDOUT = sys.stdout
sys.stdout = sys.stderr              # library chatter goes to the log, not the protocol
...
print(json.dumps(response), file=_PROTOCOL_STDOUT, flush=True)
```

Two more traps worth knowing. A scorer named after the package it wraps (`finches.py` importing
`finches`) shadows that package, because python puts the script's own directory first on `sys.path`
— drop it before importing. And a scorer's environment resolves its own torch, which can be newer
than the host CUDA driver (`The NVIDIA driver on your system is too old`); pin the build that matches
your driver in the PEP 723 header. Since it
imports nothing from IDiom, the same `cmd` form covers an on-demand uv env, a pre-built venv
(`/path/venv/bin/python …`), a conda env (`conda run -n env python …`), or a container
(`docker run -i …`). The worked example is
[`idiom/rewards/external_rewards/sparrow.py`](src/idiom/rewards/external_rewards/sparrow.py), which wraps
[sparrow](https://github.com/idptools/sparrow) for biophysics (radius of gyration, asphericity,
scaling exponent, charge patterning); strip its `value()` down to your own model and the rest of the
file is the protocol boilerplate you keep.

## Extending IDiom

Everything below works from a plain `pip install` — your code lives in your project, not in this
repository, and the shipped versions are reference examples to copy rather than files to edit.

**Your own reward.** Register `f(idr) -> float` in a file anywhere and name that file in the term:

```yaml
reward.terms:
  - {reward: my_reward, module: /path/to/my_rewards.py, weight: 1.0}
```

**Your own reward model**, in its own environment — write a program that speaks the scorer protocol
(see *Writing a scorer* below) and name the command:

```yaml
reward.terms:
  - {cmd: "uv run --script /path/to/my_scorer.py", label: mine, weight: 1.0}
```

**Your own config.** Keep it in your project and inherit the shipped one, rather than forking it:

```yaml
# my_grpo.yaml
defaults:
  - grpo
  - _self_

init_from: jxliu2/idiom-300M
reward:
  terms:
    - {reward: my_reward, module: /path/to/my_rewards.py, weight: 1.0}
    - {reward: entropy, module: "${idiom_rewards:custom_rewards.py}", weight: 1.0,
       shaping: {type: quadratic, target: 3.65, width: 0.2}}
```

```bash
idiom_grpo --config-dir . --config-name my_grpo
```

`${idiom_rewards:...}` resolves to the shipped reward files wherever IDiom is installed. **Quote it
inside a flow mapping** (`{...}` on one line) — unquoted, YAML reads the `{` as a nested mapping and
fails before Hydra sees it.

**A new training method.** `src/idiom/train/` holds one self-contained package per method — `autoreg`
(pretraining and SFT) and `grpo` — each with its own LightningModule, config, and entrypoint. A new
method (DPO, say) is a new sibling package plus a config and a `[project.scripts]` entry, not a
change to an existing one.

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
| `src/idiom/` | the library: `data` (tokenizer/FIM/dataset), `model` (transformer + KV cache + sampling), `train/` one package per training method (`autoreg` = pretraining and SFT, `grpo` = RL post-training), `sae/` (`model`, `train`, `steer`, `features`, `eval`), `rewards/` (shipped reward content), `utils`, public `IDiom`/`IDiomSAE` API |

| `cookbook/` | `scripts/` (generation and embeddings, SAE features + steering, enrichment + logos), `slurm/` training scripts (pretrain, SFT, GRPO, SAE), and small input sets in `example_data/` (ProtGPS + AD/RD IDRs) |
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
