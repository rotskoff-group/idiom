# IDiom

IDiom is an autoregressive transformer for generating, designing, and studying intrinsically disordered protein regions (IDRs). Trained on 54M IDRs curated from the AlphaFold Database with a fill-in-the-middle objective, IDiom generates IDRs **unprompted** (de novo) or **prompted** (conditioned on flanking context), and can be post-trained with reinforcement learning against custom rewards.

This work additionally presents IDiomSAE, sparse TopK autoencoders trained on IDiom's residual stream, for mechanistically interpreting and causally steering the model.

Preprint: [Generative design of intrinsically disordered protein regions with IDiom](https://doi.org/10.64898/2026.04.10.717777)

![IDiom](assets/github_fig.png)

Start with the **[cookbook](cookbook/)** — runnable scripts, indexed by what you want to do.

## What post-training does

500 unprompted generations each. Both RL runs are 3000 GRPO steps from `idiom-300M` with the shipped
config, changing only which reward term is on.

| | base | + RL toward SAE features | + RL toward radius of gyration |
|---|---|---|---|
| reward | — | `sae_only_nucleolus` | sparrow Rg → 25 Å |
| R<sub>S</sub>(nucleolus) | 0.036 | **0.990** | 0.011 |
| radius of gyration (Å) | 25.8 | 28.2 | **25.0** |
| composition entropy (bits) | 3.66 | 3.65 | 3.66 |
| predicted disorder | 0.67 | 0.93 | 0.93 |

R<sub>S</sub> is the fraction of a target's SAE feature signature firing in a generated IDR; SFT on
the same set reaches 0.163. The bottom rows are the guardrail terms holding the sequences on the IDR
distribution while the objective moves.

## Install

```bash
git clone https://github.com/rotskoff-group/idiom.git
cd idiom
uv sync      # .venv with the locked torch build, the package, and its CLIs
```

Then `source .venv/bin/activate`, or prefix commands with `uv run`. Python >= 3.10.

**IDiom is used from a clone** — [`rewards/`](rewards/) and [`cookbook/`](cookbook/) are repository
material a run points at, not library code. `uv add --editable /path/to/idiom` (your own uv project)
and `pip install -e .` (conda/venv) also work, but resolve torch themselves; only `uv sync` is
reproducible. A non-editable `pip install git+...` has no `rewards/` for the config to find.

## Sequence conventions (read this first)

IDiom is a fill-in-the-middle model, so **every input carries an IDR span**. Getting this wrong does
not raise — it silently produces off-distribution output.

- **FASTA headers end with `_IDR_x-y`** (1-based, inclusive), e.g. `>P06748_IDR_119-242`. A fully
  disordered sequence uses `_IDR_1-<len>`.
- **Python coordinates are 0-based, half-open** (`idr = seq[idr_start:idr_end]`).
- Only the **20 canonical amino acids**. Non-canonical FASTA entries are dropped with a logged
  count; a non-canonical sequence passed explicitly raises.

Three things are spelled "IDR" and are *not* the same axis:

| | Values | Means |
|---|---|---|
| the header span | `_IDR_x-y` | which residues of a protein are disordered |
| `fim_mode` / prompting | `prompted` / `unprompted` | whether the prompt carries the flanks |
| SAE `region` | `all` / `idr` / `non_idr` | which residues an SAE reads and edits |

`embed`, `encode` and `build_feature_dataset` all accept a FASTA path, one sequence, or a list. A
bare sequence string is treated as an unprompted IDR.

## Generation and embeddings

```python
from idiom import IDiom

model = IDiom.from_pretrained("jxliu2/idiom-300M")   # HF repo id or a local directory

idrs = model.generate_unprompted(n=100, temperature=1.0)
idrs = model.generate_unprompted(n=100, length_range=(80, 120))
idrs = model.generate_prompted(protein_seq, idr_start, idr_end, n=100)

values, index = model.embed(["MKKLVA...", "GSGSQP..."], layers=[18], pool="none")[18]
```

`generate_unprompted_fasta` / `generate_prompted_fasta` write record FASTAs directly, as do the CLIs:

```bash
idiom_generate unprompted --model jxliu2/idiom-300M --n 1000 --out idrs.fasta
idiom_extract --ckpt model.ckpt --fasta proteins.fasta --layers 18 --out embeddings/
```

Walkthrough: [`cookbook/scripts/generate_and_embed.py`](cookbook/scripts/generate_and_embed.py).

## Interpretability and steering

`IDiomSAE` bundles an SAE with its host model and layer, so it always runs on the distribution it
was trained on.

```python
from idiom import IDiomSAE

sae = IDiomSAE.from_pretrained("jxliu2/idiomsae-300M-L18-k32")   # host model auto-loaded

feats, accessions = sae.encode("proteins.fasta", pool="mean")    # pool="none" for per-residue
seqs = sae.steer_generate(feature=1234, strength=0.5, n=100)
```

Steering modes are `add_direction` (default), `clamp`, and `ablate`; `normalize`, `relative` and
`preserve_norm` control how `strength` is interpreted.

```bash
idiom_feature_dataset --sae jxliu2/idiomsae-300M-L18-k32 --fasta records.fasta --out features/
streamlit run src/idiom/sae/features/feature_viewer.py -- --features features/
idiom_sae model_ckpt=/path/model.ckpt data.fasta=/path/records.fasta layer=18 sae.k=32
```

Walkthroughs: [`sae_features.py`](cookbook/scripts/sae_features.py) (read and steer),
[`feature_enrichment.py`](cookbook/scripts/feature_enrichment.py) (what your own set shares).

## Training and RL post-training

Hydra CLIs over flat YAMLs in `src/idiom/configs/`. For real runs start from the
[cookbook's training scripts](cookbook/slurm/), which spell out every config value.

```bash
idiom_build_store --fasta corpus.fasta          # memory-mapped record store, recommended at scale
idiom_train data.train_fasta=corpus.fasta model.n_layers=24 model.d_model=1024
idiom_train --config-name sft init_from=jxliu2/idiom-300M data.train_fasta=sft.fasta
```

GRPO optimizes a list of reward **terms**, each pairing a reward with the shaping that says what a
good value is: `total = Σ weightᵢ · shapingᵢ(rewardᵢ)`. The shipped config carries the entropy and
length guardrails plus a menu of further terms — SAE feature codes, in-process rewards, six external
reward models — switched off until you want one:

```bash
idiom_grpo init_from=jxliu2/idiom-300M \
  reward.terms.2.enabled=true reward.terms.2.reward=sae_only_nucleolus
```

**Everything about rewards — the menu, picking targets and weights, adding your own — is in
[`rewards/README.md`](rewards/README.md).**

Keep your own configs in your own project rather than forking the shipped one:

```yaml
# my_grpo.yaml — idiom_grpo --config-dir . --config-name my_grpo
defaults: [grpo, _self_]
init_from: jxliu2/idiom-300M
reward:
  terms:
    - {reward: my_reward, module: /path/to/my_rewards.py, weight: 1.0}
```

`src/idiom/train/` holds one self-contained package per method (`autoreg` = pretraining and SFT,
`grpo`), so a new method is a new sibling package plus a config and a `[project.scripts]` entry.

## Command-line reference

| Command | Does |
|---------|------|
| `idiom_generate` | generate unprompted/prompted IDRs to a FASTA |
| `idiom_extract` | export residual-stream embeddings from a FASTA |
| `idiom_train` | pretrain (and SFT via `--config-name sft`) |
| `idiom_grpo` | GRPO / RL post-training against a reward |
| `idiom_sae` | train a top-k SAE on a layer |
| `idiom_feature_dataset` | build the per-residue SAE feature dataset |
| `idiom_build_store` | build a memory-mapped record store from a record FASTA |

## Models and data

| Repo | Params | Architecture |
|------|--------|--------------|
| [`jxliu2/idiom-300M`](https://huggingface.co/jxliu2/idiom-300M) | 302M | 24 layers, d_model 1024 |
| [`jxliu2/idiom-85M`](https://huggingface.co/jxliu2/idiom-85M) | 85M | 12 layers, d_model 768 |
| [`jxliu2/idiom-20M`](https://huggingface.co/jxliu2/idiom-20M) | 18.9M | 6 layers, d_model 512 |
| [`jxliu2/idiomsae-300M-L18-k32`](https://huggingface.co/jxliu2/idiomsae-300M-L18-k32) | — | SAE on `idiom-300M` layer 18: 16,384 latents, TopK k=32, IDR residues, unprompted |

[`jxliu2/idiom-data`](https://huggingface.co/datasets/jxliu2/idiom-data) holds plain FASTAs:
`training_sequences/` is the pretraining corpus (`train.fasta` ~53.6M records / 23.6 GB, plus
validation and test at ~271k each); reference and generated sets are coming.

```bash
hf download jxliu2/idiom-data --repo-type dataset --include "training_sequences/*"
```

The corpus is **54,155,136 IDR records** from the AlphaFold Database (pLDDT segmentation, 90%
clustering, length <= 1020, dedup, DisProt holdout removal, SignalP-6 filtering), released CC BY 4.0
inherited from AlphaFold DB / UniProt. The code here is MIT.

## Repository layout

| Path | Role |
|------|------|
| `src/idiom/` | the library: `data`, `model`, `train/` (`autoreg`, `grpo`), `sae/` (`model`, `train`, `steer`, `features`), `configs/`, `utils`, and the `IDiom`/`IDiomSAE` API in `api.py` |
| `rewards/` | reward content you edit ([README](rewards/README.md)) |
| `cookbook/` | walkthrough scripts, training scripts, example data ([README](cookbook/README.md)) |
| `tests/`, `assets/` | tests; figures for docs |

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
