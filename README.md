# IDiom

[Preprint](https://doi.org/10.64898/2026.04.10.717777) &sdot; [Models](https://huggingface.co/jxliu2) &sdot; [Data](https://huggingface.co/datasets/jxliu2/idiom-data) &sdot; [Cookbook](cookbook/)

IDiom is an autoregressive transformer for generating, designing, and studying intrinsically
disordered protein regions (IDRs). Trained on 54M IDRs from the AlphaFold Database with a
fill-in-the-middle objective, it generates IDRs **unprompted** (de novo) or **prompted** (conditioned
on flanking context), and can be post-trained with reinforcement learning against custom rewards.

We additionally release IDiomSAE, sparse top-k autoencoders trained on IDiom's residual stream, for
interpreting and steering the model.

![IDiom](assets/github_fig.png)

## Models

| Model Name | Parameters | Architecture | HuggingFace Link |
|---|---|---|---|
| idiom-300M | 302M | 24 layers, d_model 1024 | [jxliu2/idiom-300M](https://huggingface.co/jxliu2/idiom-300M) |
| idiom-85M | 85M | 12 layers, d_model 768 | [jxliu2/idiom-85M](https://huggingface.co/jxliu2/idiom-85M) |
| idiom-20M | 18.9M | 6 layers, d_model 512 | [jxliu2/idiom-20M](https://huggingface.co/jxliu2/idiom-20M) |
| idiomsae-300M-L18-k32 | — | SAE on `idiom-300M` layer 18, 16,384 latents, k=32 | [jxliu2/idiomsae-300M-L18-k32](https://huggingface.co/jxliu2/idiomsae-300M-L18-k32) |

Weights download on first use. Inference runs on CPU; a GPU is recommended.

## Installation

```bash
pip install git+https://github.com/rotskoff-group/idiom.git
```

Or clone, which additionally gives you [`cookbook/`](cookbook/) and a locked torch build:

```bash
git clone https://github.com/rotskoff-group/idiom.git
cd idiom
uv sync      # .venv with the locked torch build, the package, and its CLIs
```

Then `source .venv/bin/activate`, or prefix commands with `uv run`. Python >= 3.10.

## Sequence conventions

IDiom is a fill-in-the-middle model, so **every input carries an IDR span**. Getting this wrong does
not raise — it silently produces off-distribution output.

- **FASTA headers end with `_IDR_x-y`**, 1-based inclusive, e.g. `>P06748_IDR_119-242`. A fully
  disordered sequence uses `_IDR_1-<len>`. A bare sequence string is treated as an unprompted IDR.
- **Python coordinates are 0-based, half-open**: `idr = seq[idr_start:idr_end]`.
- Only the **20 canonical amino acids**. Non-canonical FASTA entries are dropped with a logged count;
  a non-canonical sequence passed explicitly raises.

## Interactive Usage

```python
from idiom import IDiom

model = IDiom.from_pretrained("jxliu2/idiom-300M")   # HF repo id, local directory, or .ckpt

idrs = model.generate_unprompted(n=100, temperature=1.0)
idrs = model.generate_unprompted(n=100, length_range=(80, 120))
idrs = model.generate_prompted(protein_seq, idr_start, idr_end, n=100)

values, index = model.embed(["MKKLVA...", "GSGSQP..."], layers=[18], pool="none")[18]
```

`generate_unprompted_fasta` / `generate_prompted_fasta` write record FASTAs directly. Passing
`return_full=True` to the prompted one splices each generated IDR back between its flanks and writes
the whole protein with a corrected span — how you redesign the IDR of an existing protein.

Score sequences under the model's own fill-in-the-middle objective:

```python
from idiom.utils.perplexity import perplexity

perplexity(model.model, "heldout.fasta", device=model.device)   # {"nll", "perplexity", "n_tokens"}
```

Tutorial: [`generate_and_embed.ipynb`](cookbook/notebooks/generate_and_embed.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/generate_and_embed.ipynb)

## Sparse Autoencoders

`IDiomSAE` bundles an SAE with its host model and layer, so it always runs on the distribution it was
trained on.

```python
from idiom import IDiomSAE

sae = IDiomSAE.from_pretrained("jxliu2/idiomsae-300M-L18-k32")   # host model auto-loaded

feats, accessions = sae.encode("proteins.fasta", pool="mean")    # pool="none" for per-residue
seqs = sae.steer_generate(feature=1234, strength=0.5, n=100)
```

`encode` takes `region=` (`all`, `idr`, `non_idr`), defaulting to what the SAE was trained on. The
released SAE was trained unprompted on IDR residues, so `region="idr"` is its only valid value.
Steering modes are `add_direction` (default), `clamp`, and `ablate`; `normalize`, `relative`, and
`preserve_norm` control how `strength` is interpreted.

Build a per-residue feature dataset, then rank features and pull the sequences that drive one:

```bash
idiom_feature_dataset --sae jxliu2/idiomsae-300M-L18-k32 --fasta records.fasta --out features/
streamlit run src/idiom/sae/features/feature_viewer.py -- --features features/
```

```python
from idiom.sae.features import FeatureDataset

fd = FeatureDataset("features/")
ids, freq, mean_act = fd.feature_ranking()
seq_ids, scores = fd.top_sequences(ids[0], n=20)
```

Tutorials: [`sae_features.ipynb`](cookbook/notebooks/sae_features.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/sae_features.ipynb),
[`feature_enrichment.ipynb`](cookbook/notebooks/feature_enrichment.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/feature_enrichment.ipynb)

## Training and Post-training

Hydra CLIs over flat YAMLs in `src/idiom/configs/`. For real runs start from the
[cookbook's training scripts](cookbook/scripts/), which spell out every config value.

```bash
idiom_build_store --fasta corpus.fasta          # memory-mapped record store, recommended at scale
idiom_train_autoreg data.train_fasta=corpus.fasta model.n_layers=24 model.d_model=1024
idiom_train_autoreg --config-name sft init_from=jxliu2/idiom-300M data.train_fasta=sft.fasta
idiom_train_sae model_ckpt=/path/model.ckpt data.fasta=/path/records.fasta layer=18 sae.k=32
```

GRPO optimizes a list of reward **terms**, each pairing a reward with the shaping that says what a
good value is: `total = Σ weightᵢ · shapingᵢ(rewardᵢ)`. The shipped config carries **no terms at
all** — no defaults, no presets — so a run names its whole objective and the launch line is what is
being optimized.

A term is four keys — `reward`, `shaping`, `weight`, `label` — and the reward and the shaping are
named the same way: a name, plus that thing's own arguments.

```bash
ENTROPY='{label: entropy, weight: 1.0, reward: entropy, shaping: {name: quadratic, target: 3.65, width: 0.2}}'
LENGTH='{label: length,  weight: 1.0, reward: length,  shaping: {name: quadratic, target: 100,  width: 1.0}}'

# an SAE feature signature (the RL-SAE result)
SAE='{label: sae, weight: 1.0, reward: {name: sae_signature, signature: nucleolus,
                                        features: signature.json}}'

# a reward model in its own environment
RG='{label: rg, weight: 0.5,
     reward: {name: scorer,
              cmd: "uv run --script cookbook/rewards/scorers/sparrow.py --property radius_of_gyration"},
     shaping: {name: quadratic, target: 25, width: 0.2}}'

# anything importable, with its own settings
MINE='{label: mine, weight: 1.0, reward: {name: "mypackage.scoring:make_scorer", cutoff: 0.3}}'

idiom_train_grpo init_from=jxliu2/idiom-300M reward.terms="[$ENTROPY, $LENGTH, $RG]"
```

`entropy`, `length`, `scorer` and `sae_signature` are the shipped reward names; `quadratic`,
`gaussian` and `identity` the shipped shaping. Anything else is a `module:function` path to a
factory of your own — no registration, no decorators. `entropy` and `length` are worth naming in
most objectives, since a target is otherwise satisfiable by a low-complexity tract or a degenerate
length, but they are terms like any other, so drop either one and it is gone.
`cookbook/scripts/grpo/` has a ready-to-submit script per objective.

Six external scorers ship in the cookbook — sparrow, finches, PSpred, ProtGPS, PADDLE, STARLING —
each a standalone program carrying its own environment in a
[PEP 723](https://peps.python.org/pep-0723/) header, so there is no install step. **Rewards are
documented in full in [`cookbook/rewards/README.md`](cookbook/rewards/README.md).**

Training writes a Lightning `.ckpt`. `save_pretrained` converts one into the released
`config.json` + `model.safetensors` pair, and `push_to_hub` uploads it with a model card:

```python
model = IDiom.load("runs/grpo/checkpoints/last.ckpt")
model.save_pretrained("my-idiom-nucleolus")
model.push_to_hub("me/my-idiom-nucleolus", private=True)
```

`IDiomSAE` has the same two methods, recording its host model so the pair reloads in one call.

## Command-line Tools

| Command | Does |
|---------|------|
| `idiom_generate` | generate unprompted/prompted IDRs to a FASTA |
| `idiom_extract` | export residual-stream embeddings from a FASTA |
| `idiom_train_autoreg` | pretrain, and SFT via `--config-name sft` |
| `idiom_train_grpo` | GRPO post-training against a reward |
| `idiom_train_sae` | train a top-k SAE on a layer |
| `idiom_feature_dataset` | build the per-residue SAE feature dataset |
| `idiom_build_store` | build a memory-mapped record store from a record FASTA |

```bash
idiom_generate unprompted --model jxliu2/idiom-300M --n 1000 --out idrs.fasta
idiom_extract --ckpt model.ckpt --fasta proteins.fasta --layers 18 --out embeddings/
```

## Data

[`jxliu2/idiom-data`](https://huggingface.co/datasets/jxliu2/idiom-data) holds plain FASTAs.
`training_sequences/` is the pretraining corpus — `train.fasta`, 53.6M records / 23.6 GB, plus
validation and test at ~271k each. `example_data/` holds the demo-sized sets the notebooks use.

```bash
hf download jxliu2/idiom-data --repo-type dataset --include "training_sequences/*"
```

The corpus is 54,155,136 IDR records from the AlphaFold Database (pLDDT segmentation, 90% clustering,
length <= 1020, dedup, DisProt holdout removal, SignalP-6 filtering).

## Contributing

Contributions are welcome — fork the repository, raise issues, contribute reward functions, and open
pull requests.

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

Code is released under the [MIT License](LICENSE). The pretraining corpus is CC BY 4.0, inherited
from AlphaFold DB / UniProt. Data redistributed in [`cookbook/example_data/`](cookbook/example_data/)
carries the licenses of its original sources.
