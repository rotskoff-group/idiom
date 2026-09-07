# IDiom

[Preprint](https://doi.org/10.64898/2026.04.10.717777) &sdot; [Models](https://huggingface.co/jxliu2) &sdot; [Data](https://huggingface.co/datasets/jxliu2/idiom-data) &sdot; [Cookbook](cookbook/)

IDiom is an autoregressive transformer for generating and studying intrinsically disordered protein
regions (IDRs), trained on 54M IDRs from the AlphaFold Database. It generates sequences de novo or
conditioned on flanking protein context, and supports fine-tuning with custom rewards.
IDiomSAE provides sparse autoencoders for interpreting and steering the model.

![IDiom](assets/github_fig.png)

## Installation

Python ≥3.10. Choose either setup.

### Install into an existing environment

Activate your Python environment, then install:

```bash
python -m pip install git+https://github.com/rotskoff-group/idiom.git
```

To access the cookbook files, also clone the repository:

```bash
git clone https://github.com/rotskoff-group/idiom.git
cd idiom
```

### Create an environment from the clone

With `uv` installed (`python -m pip install uv`):

```bash
git clone https://github.com/rotskoff-group/idiom.git
cd idiom
uv sync
source .venv/bin/activate
```

This installs the checkout with its locked dependencies. For either setup, see
[running cookbook scripts](cookbook/scripts/README.md#running-scripts) for paths and run settings.

## Quickstart

Generate ten IDRs:

```python
from idiom import IDiom

model = IDiom.from_pretrained("jxliu2/idiom-300M")
sequences = model.generate_unprompted(n=10)
print(sequences[0])
```

Weights download on first use. Inference supports CPU; a GPU is recommended.
Generation and SAE steering use batches of eight by default. Set `batch_size` (CLI: `--batch-size`)
to adjust memory use; reduce it for smaller devices. Seeded results are reproducible for a fixed batch size.
`from_pretrained` accepts a Hub model ID or a released directory. Use `IDiom.load` to also
accept a Lightning `.ckpt` file.

### Generation and embeddings

Generate IDRs within a length range and embed them:

```python
sequences = model.generate_unprompted(n=10, length_range=(80, 120))
values, index = model.embed(sequences, layers=[18], pool="mean")[18]
```

Generation caps `max_new_tokens` at the remaining model context, accounting for flanks and FIM
markers, and warns when reducing the requested budget. Prompts exceeding the context are rejected.
Length filtering may return fewer sequences if it reaches the sampling limit. Use `temperature=0`
for greedy generation; `seed` controls stochastic sampling. Invalid generation options raise `ValueError`.

Embedding layers are zero-based block indices. `pool="mean"` returns one vector per record,
averaged over its IDR residues; `pool="none"` returns per-residue rows. Each layer maps to
`(values, index)`: a NumPy array and matching metadata. Per-residue metadata includes `record_idx`,
`accession`, `source_pos` (0-based in the original protein), `residue`, and `is_idr`.

For prompted generation, supply a protein sequence and its 0-based, half-open IDR span:

```python
# Toy example: replace seq[3:9], keeping the flanks.
seq = "MEDSKVDNRPQACDEFG"
replacements = model.generate_prompted(seq, idr_start=3, idr_end=9, n=10)
```

To redesign proteins from an existing [record FASTA](#sequence-conventions):

```python
model.generate_prompted_fasta(
    "proteins.fasta", "redesigned.fasta", n=10, return_full=True
)
```

`return_full=True` inserts each generated IDR between its flanks and updates the FASTA span.
For de novo FASTA output, use `model.generate_unprompted_fasta("idrs.fasta", n=100)` or:

```bash
idiom_generate unprompted --model jxliu2/idiom-300M --n 100 --out idrs.fasta
```

### SAE features and steering

`IDiomSAE` loads the SAE with its recorded host model, layer, and prompt format:

```python
from idiom import IDiomSAE

sae = IDiomSAE.from_pretrained("jxliu2/idiomsae-300M-L18-k32")
features, accessions = sae.encode(sequences, pool="mean")
steered = sae.steer_generate(feature=1234, strength=0.5, n=10)
```

Use `pool="none"` for per-residue features. The released SAE uses unprompted IDRs and accepts
only `region="idr"` (its default). Steering supports `add_direction`, `clamp`, and `ablate`.
For ablation, `strength=0` leaves activations unchanged and `strength=1` removes the selected
features' decoder contributions. Mean encoding preserves separate records with repeated accessions.
See the [SAE notebook](cookbook/notebooks/sae_features.ipynb) to build feature datasets,
rank features, and inspect activation traces.

### Saving and publishing

Export a checkpoint as `config.json` and `model.safetensors`:

```python
model = IDiom.load("/path/to/model.ckpt")
model.save_pretrained("my-idiom")
```

After authenticating with Hugging Face, upload a release with:

```python
model.push_to_hub("your-account/my-idiom", private=True, model_card="# My IDiom model")
```

`IDiomSAE` also provides `save_pretrained` and `push_to_hub`, recording its host model for
reloading. SAE releases contain `sae_config.json` and `sae.safetensors`.

See the [generation notebook](cookbook/notebooks/generate_and_embed.ipynb) for more examples,
including perplexity scoring.

## Models

| Model | Parameters | Architecture |
|---|---|---|
| [idiom-300M](https://huggingface.co/jxliu2/idiom-300M) | 302M | 24 layers, width 1024 |
| [idiom-85M](https://huggingface.co/jxliu2/idiom-85M) | 85M | 12 layers, width 768 |
| [idiom-20M](https://huggingface.co/jxliu2/idiom-20M) | 18.9M | 6 layers, width 512 |
| [idiomsae-300M-L18-k32](https://huggingface.co/jxliu2/idiomsae-300M-L18-k32) | — | SAE on layer 18 of idiom-300M; 16,384 latents, k=32 |

## Sequence conventions

IDiom uses fill-in-the-middle formatting, which requires an IDR span. Incorrect spans can produce
off-distribution output even when the input is accepted.

- FASTA headers end with `_IDR_x-y`, using 1-based inclusive coordinates:
  `>P06748_IDR_119-242`. For a fully disordered sequence, use `_IDR_1-<length>`.
- Python coordinates are 0-based and half-open: `idr = seq[idr_start:idr_end]`.
  A bare sequence string is treated as an unprompted IDR.
  Prompted generation requires `0 <= idr_start < idr_end <= len(seq)`.
- Use the 20 canonical amino acids. Non-canonical FASTA entries are dropped with a logged count;
  explicitly supplied non-canonical sequences raise an error.

Record FASTA readers skip missing, malformed, or out-of-range spans with a logged count.
The enrichment workflow instead treats missing or unusable spans as whole-sequence IDRs;
check annotations when supplying full proteins.

## Training

The [cookbook](cookbook/) includes scripts for pretraining, supervised fine-tuning, SAE training,
and GRPO post-training. Training commands use [YAML configs](src/idiom/configs/) with command-line
overrides; add `--cfg job` to inspect a command's configuration without starting training.
The [reward guide](cookbook/rewards/) explains custom objectives and external scorers.

## Command-line tools

| Command | Purpose |
|---|---|
| `idiom_generate` | Generate unprompted or prompted IDRs to FASTA |
| `idiom_extract` | Export residual-stream embeddings |
| `idiom_train_autoreg` | Pretrain, or fine-tune with `--config-name sft` |
| `idiom_train_grpo` | Post-train with custom rewards |
| `idiom_train_sae` | Train a sparse autoencoder |
| `idiom_feature_dataset` | Build a per-residue SAE feature dataset |
| `idiom_feature_enrichment` | Compare IDR sets and export an enriched SAE feature signature |
| `idiom_build_store` | Build a memory-mapped record store from FASTA |

Export embeddings from a Hub model, released directory, or Lightning checkpoint:

```bash
idiom_extract --model jxliu2/idiom-300M --fasta proteins.fasta --layers 18 --pool mean --out embeddings
```

The older `--ckpt` flag remains an alias for `--model`.

## Data

[jxliu2/idiom-data](https://huggingface.co/datasets/jxliu2/idiom-data) contains the training FASTAs
and cookbook example data. The training split contains 53.6M records (23.6 GB); validation and test
contain approximately 271k records each.

```bash
hf download jxliu2/idiom-data --repo-type dataset --include "training_sequences/*"
```

See the [cookbook data notes](cookbook/example_data/) for demo datasets and provenance.

## Contributing

Issues, reward examples, and pull requests are welcome.

From a clone, install the development dependencies and run the checks:

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```

Optionally run `uv run pre-commit install` to enable the commit hooks.

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
from AlphaFold DB / UniProt. Data in [cookbook/example_data/](cookbook/example_data/) retains the
licenses of its original sources.
