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

Into an environment you already have — the library, the CLIs, the configs, and the shipped reward
scorers, with torch resolved against whatever is already there:

```bash
pip install git+https://github.com/rotskoff-group/idiom.git
```

Or as a clone, which additionally gives you [`cookbook/`](cookbook/) and a locked, reproducible
torch build:

```bash
git clone https://github.com/rotskoff-group/idiom.git
cd idiom
uv sync      # .venv with the locked torch build, the package, and its CLIs
```

Then `source .venv/bin/activate`, or prefix commands with `uv run`. Python >= 3.10.

Both installs are complete for training, generation, interpretability and RL — nothing in the
package reaches outside itself. The clone adds the runnable examples and example data, and pins
torch; `pip install` leaves your existing torch alone, which is what you want when IDiom is going in
beside code you already have.

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

`generate_unprompted_fasta` / `generate_prompted_fasta` write record FASTAs directly. Passing
`return_full=True` to the prompted one splices each generated IDR back between its flanks and writes
the **whole protein** with a corrected span — how you redesign the IDR of an existing protein, rather
than collecting IDRs on their own. The same is available from the command line:

```bash
idiom_generate unprompted --model jxliu2/idiom-300M --n 1000 --out idrs.fasta
idiom_extract --ckpt model.ckpt --fasta proteins.fasta --layers 18 --out embeddings/
```

Score sequences under the model's own fill-in-the-middle objective — to check a fine-tune against
held-out data, or to rank designs by how IDR-like the model finds them:

```python
from idiom.utils.perplexity import perplexity

perplexity(model.model, "heldout.fasta", device=model.device)   # {"nll", "perplexity", "n_tokens"}
```

Walkthrough: [`cookbook/scripts/python/generate_and_embed.py`](cookbook/scripts/python/generate_and_embed.py).

## Interpretability and steering

`IDiomSAE` bundles an SAE with its host model and layer, so it always runs on the distribution it
was trained on.

```python
from idiom import IDiomSAE

sae = IDiomSAE.from_pretrained("jxliu2/idiomsae-300M-L18-k32")   # host model auto-loaded

feats, accessions = sae.encode("proteins.fasta", pool="mean")    # pool="none" for per-residue
seqs = sae.steer_generate(feature=1234, strength=0.5, n=100)
```

`encode` also takes `region=` to select which residues are read (`all`, `idr`, `non_idr`), defaulting
to what the SAE was trained on. The released SAE was trained **unprompted on IDR residues**, so
`region="idr"` is its only valid value; asking for another raises rather than returning something
meaningless.

Steering modes are `add_direction` (default), `clamp`, and `ablate`; `normalize`, `relative` and
`preserve_norm` control how `strength` is interpreted.

```bash
idiom_feature_dataset --sae jxliu2/idiomsae-300M-L18-k32 --fasta records.fasta --out features/
streamlit run src/idiom/sae/features/feature_viewer.py -- --features features/
idiom_sae model_ckpt=/path/model.ckpt data.fasta=/path/records.fasta layer=18 sae.k=32
```

The same dataset reads from Python — rank features by how often they fire, then pull the sequences
that drive one:

```python
from idiom.sae.features import FeatureDataset

fd = FeatureDataset("features/")
ids, freq, mean_act = fd.feature_ranking()
seq_ids, scores = fd.top_sequences(ids[0], n=20)
```

Walkthroughs: [`sae_features.py`](cookbook/scripts/python/sae_features.py) (read and steer),
[`feature_enrichment.py`](cookbook/scripts/python/feature_enrichment.py) (what your own set shares).

## Training and RL post-training

Hydra CLIs over flat YAMLs in `src/idiom/configs/`. For real runs start from the
[cookbook's training scripts](cookbook/scripts/bash/), which spell out every config value.

```bash
idiom_build_store --fasta corpus.fasta          # memory-mapped record store, recommended at scale
idiom_train data.train_fasta=corpus.fasta model.n_layers=24 model.d_model=1024
idiom_train --config-name sft init_from=jxliu2/idiom-300M data.train_fasta=sft.fasta
```

The `model:` block above is the released 300M architecture; 85M is 12L/768/12 and 20M is 6L/512/8
(`idiom_20m()`, `idiom_85m()`, `idiom_300m()` in `idiom.model` return the same as `ModelConfig`s).

GRPO optimizes a list of reward **terms**, each pairing a reward with the shaping that says what a
good value is: `total = Σ weightᵢ · shapingᵢ(rewardᵢ)`. The config carries only the entropy and
length guardrails; **what a run optimizes is chosen at launch**, by name, from a shipped menu of
tuned terms — SAE feature codes, six external reward models, and the built-in composition rewards:

```bash
idiom_grpo init_from=jxliu2/idiom-300M reward.add=[sae]
```

Names compose and stay addressable as the menu grows, which list positions do not:

```bash
idiom_grpo init_from=jxliu2/idiom-300M reward.add=[sae,rg] \
  reward.presets.rg.shaping.target=30
```

Your own reward needs no entry in the menu — pass the whole term:

```bash
idiom_grpo init_from=jxliu2/idiom-300M \
  reward.add='[{reward: "mypackage.scoring:score_idr", weight: 1.0}]'
```

The guardrails and the other shipped rewards are built into the library, so a term names one and
needs nothing else. Your own reaches a run one of three ways:

```yaml
- {reward: fraction_aromatic, module: /path/to/my_rewards.py, weight: 1.0}  # registered by name
- {reward: "mypackage.scoring:score_idr", weight: 1.0}                      # any importable callable
- {cmd: "uv run --script /path/to/my_scorer.py", label: mine, weight: 1.0}  # its own environment
```

The second is the one to reach for when IDiom is installed beside code that can already score a
sequence: no decorator, no edit to that code, nothing copied into this repo. The third is for a
reward model whose dependencies cannot coexist with IDiom's — six such scorers ship, each carrying
its own environment in a [PEP 723](https://peps.python.org/pep-0723/) header.

**Everything about rewards — the scorers, picking targets and weights, adding your own — is in
the `rewards/` section of [`cookbook/README.md`](cookbook/README.md).**

### After training

Training writes a Lightning `.ckpt`, which `from_pretrained` cannot read — use `IDiom.load`, which
takes a `.ckpt`, a released directory, or a Hub repo id. `save_pretrained` converts one into the
released `config.json` + `model.safetensors` pair, and `push_to_hub` uploads it with a model card:

```python
model = IDiom.load("runs/grpo/checkpoints/last.ckpt")
model.generate_unprompted(n=100)
model.save_pretrained("my-idiom-nucleolus")
model.push_to_hub("me/my-idiom-nucleolus", private=True)
```

`IDiomSAE` has the same `save_pretrained` and `push_to_hub`, recording its host model so the pair
reloads in one call. Anywhere a model is named — `init_from`, `idiom_generate --model`, `IDiom.load`
— all three forms are accepted.

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

| `cookbook/` | walkthroughs, training scripts, reward templates, example data ([README](cookbook/README.md)) — clone only |
| `tests/`, `assets/` | tests; figures for docs |

Everything under `src/` ships in the wheel; `cookbook/` does not.

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
