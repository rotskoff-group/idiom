# IDiom

<p align="center">
  <a href="https://doi.org/10.64898/2026.04.10.717777">Preprint</a>
  |
  <a href="https://huggingface.co/collections/jxliu2/idiom">Models and Data</a>
  |
  <a href="cookbook/">Cookbook</a>
</p>

IDiom is an autoregressive protein language model trained on 54M intrinsically disordered protein regions (IDRs) curated from the AlphaFold Database. IDiom is trained in three model sizes, with 20M, 85M, and 300M parameters. This repository supports:

- **Generation** of standalone unprompted IDRs as well as IDRs conditioned on flanking protein context
- **Extraction** of sequence- and residue-level embeddings for IDRs
- **Post-training** via supervised fine-tuning and reinforcement learning with custom rewards
- **Model interpretability** and steering via IDiomSAE sparse autoencoders

![IDiom](assets/github_fig.png)

## Updates

- **2026-09-XX:** IDiom v1 release, new preprint.
- **2026-04-11:** IDiom v0 release, [Generative design of intrinsically disordered protein regions with IDiom](https://doi.org/10.64898/2026.04.10.717777), presented at ICML GenBio Workshop 2026.


## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
  - [Sequence generation](#sequence-generation)
  - [Extracting model embeddings](#extracting-model-embeddings)
  - [IDiomSAE](#idiomsae)
- [Sequence conventions](#sequence-conventions)
- [Cookbook: notebooks, scripts, and rewards](#cookbook-notebooks-post-training-and-rewards)
- [Models and Data](#models-and-data)
- [Citation](#citation)
- [License](#license)

<br>

## Installation

Please install the `v1` release directly from GitHub (requires Python ≥3.10):

```bash
pip install git+https://github.com/rotskoff-group/idiom.git@v1
```

To access the examples in `cookbook/`, please also clone the `v1` release:

```bash
git clone --branch v1 https://github.com/rotskoff-group/idiom.git
```

We welcome any contributions to this open source project. For development, clone the repository and install the package with its development dependencies (Python ≥3.10):

```bash
pip install uv
git clone https://github.com/rotskoff-group/idiom.git
cd idiom
uv sync --group dev
source .venv/bin/activate
```

If you have any questions please open an issue or email [jxliu2@stanford.edu](mailto:jxliu2@stanford.edu).

<br>

## Quickstart

Below, we provide several examples to get started with IDiom. More detailed examples and workflows are provided in `cookbook/`.

## Sequence generation

IDiom enables the generation of standalone unprompted IDRs, as well as IDRs conditioned on flanking protein context.

### Unprompted generation

Generate 10 unprompted IDRs:

```python
from idiom import IDiom

model = IDiom.from_pretrained("jxliu2/idiom-300M")
sequences = model.generate_unprompted(n=10)
print(sequences)
```

Weights download on first use. By default, device selection uses `IDIOM_DEVICE` if set,
otherwise CUDA when available or CPU. Pass `device="cpu"` or `device="cuda:0"` to
`from_pretrained` to select a device explicitly. Reduce `batch_size` if GPU memory is limited.
`from_pretrained` accepts a Hub model ID or a released directory; `IDiom.load`
also accepts a Lightning `.ckpt` file.

Generate 10 unprompted IDRs within a length range:

```python
from idiom import IDiom

model = IDiom.from_pretrained("jxliu2/idiom-300M")
sequences = model.generate_unprompted(n=10, length_range=(80, 120))
# Oversamples up to max_oversample * n draws; may return fewer than n sequences
print(sequences)
```

Sample with temperature and top-p sampling:

```python
from idiom import IDiom

model = IDiom.from_pretrained("jxliu2/idiom-300M")
sequences = model.generate_unprompted(n=10, temperature=0.8, top_p=0.9, seed=42)
print(sequences)
```

Lower temperatures concentrate sampling on more likely residues; `top_p=0.9` restricts
each step to the most likely residues whose cumulative probability reaches 90%.
Use `seed` for reproducibility with a fixed batch size. Length filtering
may return fewer sequences if it reaches the sampling limit. See the
[generation notes](cookbook/README.md#generation-and-analysis-details) for details.

For de novo FASTA output, use `model.generate_unprompted_fasta("idrs.fasta", n=10)` or:

```bash
idiom_generate unprompted --model jxliu2/idiom-300M --n 10 --out idrs.fasta
```

### Prompted generation

For prompted generation, supply a protein sequence and its IDR span. See [Sequence conventions](#sequence-conventions)
for residue position indexing conventions.

This example uses the flanking context around the IDR within residues 119–259 (1-based, inclusive) of human [NPM1 (UniProt P06748)](https://www.uniprot.org/uniprotkb/P06748/entry) as the prompt for generating IDRs:

```python
from idiom import IDiom

model = IDiom.from_pretrained("jxliu2/idiom-300M")
# Human NPM1 (UniProt P06748), full-length sequence
seq = (
    "MEDSMDMDMSPLRPQNYLFGCELKADKDYHFKVDNDENEHQLSLRTVSLGAGAKDELHIV"
    "EAEAMNYEGSPIKVTLATLKMSVQPTVSLGGFEITPPVVLRLKCGSGPVHISGQHLVAVE"
    "EDAESEDEEEEDVKLLSISGKRSAPGGGSKVPQKKVKLAADEDDDDDDEEDDDEDDDDDD"
    "FDDEEAEEKAPVKKSIRDTPAKNAQKSNQNGKDSKPSSTPRSKGQESFKKQEKTPKTPKG"
    "PSSVEDIKAKMQASIEKGGSLPKVEAKFINYVKNCFRMTDQEAIQDLWQWRKSL"
)
# Use flanking context around IDR residues 119–259 (1-based inclusive, following bio convention) as the prompt
regen_sequences = model.generate_prompted(seq, idr_start=118, idr_end=259, n=10) # Standard Python indexing here
print(regen_sequences) # Returns only the generated prompted IDRs
```

To generate prompted IDRs from an existing FASTA file containing protein sequences (see [Sequence conventions](#sequence-conventions) for FASTA file requirements), run this example from the cloned repository root to use [HP1α](cookbook/example_data/prompted_grpo/P45973.fasta) as an example sequence (IDR between residues 79–123):

```python
from idiom import IDiom

model = IDiom.from_pretrained("jxliu2/idiom-300M")
model.generate_prompted_fasta(
    "cookbook/example_data/prompted_grpo/P45973.fasta",
    "redesigned.fasta",
    n=10,
    return_full=True,
)
# redesigned.fasta is the output
```

`return_full=True` places each generated IDR between its original prompting flanks in the output FASTA, while `return_full=False` just outputs the prompted IDRs in the FASTA.

<br>

## Extracting model embeddings

Extract sequence-level embeddings from IDR sequences:

```python
from idiom import IDiom

model = IDiom.from_pretrained("jxliu2/idiom-300M")
idr_sequences = ["MSSGQSSQSPGSGQQQQSSG", "GSGSSQPSQGQSSGSSQQPN"]

values, index = model.embed(idr_sequences, layers=[18], pool="mean")[18]
# Use pool="none" for per-residue embeddings
# Embedding layers are 0-based Transformer block indices

print(values.shape) # (2, 1024) one IDR-averaged embedding per sequence
print(values) # Embedding vector
```

To export embeddings for a FASTA file of proteins with IDR regions marked, run the `idiom_extract` CLI:

```bash
idiom_extract --model jxliu2/idiom-300M \
    --fasta cookbook/example_data/disprot/disprot_len1020_idrs.fasta \
    --layers 18 --pool mean --out embeddings
```

See the [generation and embedding notebook](cookbook/notebooks/generate_and_embed.ipynb) for more detailed examples.

<br>

## IDiomSAE

We provide a TopK sparse autoencoder, IDiomSAE, trained on the residual stream of layer-18 of 24 in IDiom-300M. IDiomSAE has k = 32 and a latent dimension of z = 16,384.

```python
from idiom import IDiomSAE

sae = IDiomSAE.from_pretrained("jxliu2/idiomsae-300M-L18-k32")
idr_sequences = ["MSSGQSSQSPGSGQQQQSSG", "GSGSSQPSQGQSSGSSQQPN"]

features, accessions = sae.encode(idr_sequences, pool="mean")
# Use pool="none" for per-residue feature vectors

print(features.shape) # (2, 16384) one IDR-averaged feature vector per sequence
print(features) # SAE feature activations
```


To steer the generation of IDRs using SAE features, run:

```python
# Example feature ID 1234
steered = sae.steer_generate(feature=1234, strength=0.25, n=10)
```

This IDiomSAE only uses unprompted IDRs and only accepts `region="idr"` (its default). Use the [SAE notebook](cookbook/notebooks/sae_features.ipynb)
to inspect highly activating sequences and activation patterns, and use the
[enrichment notebook](cookbook/notebooks/feature_enrichment.ipynb) to identify features
enriched within a set of sequences.

<br>

## Sequence conventions

IDiom is trained using a fill-in-the-middle (FIM) format with one token per canonical amino acid and three positional marker tokens. The markers denote the N-terminal flank `1`, IDR `2`, and C-terminal flank `3`.

As an example, consider the example full protein sequence `MEDQSSGACDE` where `QSSG` is an IDR flanked by `MED` and `ACDE`. The sequence's FIM representation is `1MED3ACDE2QSSG`, and during prompted generation, the model receives `1MED3ACDE2` and generates an IDR conditioned on the flanks. During unprompted generation, the model only receives `132` and generates a de novo IDR without flanking context.

The positional markers are handled automatically by IDiom, and all use cases need only to supply amino acid sequences and IDR positions. **FASTA headers use 1-based, inclusive IDR residue positions**, following biological convention, and **Python methods use standard 0-based, end-exclusive IDR residue positions**. IDiom converts between these conventions automatically when reading and writing FASTA files.

In the example `MEDQSSGACDE` with IDR `QSSG`, we have:

| Interface | IDR `QSSG` in  `MEDQSSGACDE` |
|---|---|
| FASTA header | `>example_IDR_4-7` |
| Python args | `idr_start=3, idr_end=7` |
| Python slice | `seq[3:7]` |

<!-- All three identify the same four residues, `QSSG`. To convert a FASTA span `_IDR_x-y` to Python,
subtract one from the start only: `idr_start=x-1`, `idr_end=y` -->

To interface with IDiom, the first whitespace-delimited token in each FASTA header must end with `_IDR_x-y`, where `x` and `y` are the **1-based, inclusive** indices of the IDR in that record. A fully disordered sequence of `<length>` would for example have FASTA headers ending in `_IDR_1-<length>`.

FASTA readers skip records with missing or malformed IDR spans, out-of-range coordinates,
or sequences containing anything outside the 20 uppercase canonical amino acids, and log
the counts skipped. Bare sequence inputs to the Python API instead raise `ValueError`
for empty or noncanonical sequences.



For example, a FASTA record marking `QSSG` as the IDR is:

```fasta
>example_IDR_4-7
MEDQSSGACDE
```

<br>

## Cookbook: notebooks, post-training, and rewards

The cookbook in `cookbook/` provides detailed examples and workflows for using and post-training IDiom. Detailed information can be found in the [cookbook readme](cookbook/README.md).

- Notebooks in `cookbook/notebooks/`: generate IDRs, extract embeddings, and explore SAE features and enrichment, locally or in Colab.
- Scripts in `cookbook/scripts/`: run supervised fine-tuning and GRPO-based reinforcement learning with custom rewards, and run additional SAE workflows.
- Rewards in `cookbook/rewards/`: define custom reinforcement learning rewards and connect external scorers such as SPARROW, FINCHES, ProtGPS, PADDLE, STARLING, or custom code.

<br>

## Models and Data

IDiom models and data are hosted in our [Hugging Face collection](https://huggingface.co/collections/jxliu2/idiom).

| Model | Parameters | Architecture |
|---|---|---|
| [idiom-300M](https://huggingface.co/jxliu2/idiom-300M) | 302M | 24 layers, width 1024 |
| [idiom-85M](https://huggingface.co/jxliu2/idiom-85M) | 85M | 12 layers, width 768 |
| [idiom-20M](https://huggingface.co/jxliu2/idiom-20M) | 18.9M | 6 layers, width 512 |
| [idiomsae-300M-L18-k32](https://huggingface.co/jxliu2/idiomsae-300M-L18-k32) | — | SAE on layer 18 of idiom-300M; 16,384 latents, k=32 |

Training sequences, generated sequences, and cookbook example data are available in FASTA format
on Hugging Face at [jxliu2/idiom-data](https://huggingface.co/datasets/jxliu2/idiom-data).
The training split contains 53.6M records (23.6 GB), and the validation and test splits contain
271k records each (128 MB each).

To download these data:

```bash
# Download all training, validation, and test splits
hf download jxliu2/idiom-data --repo-type dataset --include "training_sequences/*"

# Download only the validation split
hf download jxliu2/idiom-data --repo-type dataset --include "training_sequences/validation.fasta"
```

<br>

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

<br>

## License

MIT license. The pretraining corpus is CC BY 4.0, from AFDB/UniProt. Data examples follow the licenses of their original sources.
