# IDiom

<p align="center">
  <a href="">Preprint</a>
  |
  <a href="https://huggingface.co/collections/jxliu2/idiom">Models</a>
  |
  <a href="https://huggingface.co/datasets/jxliu2/idiom-db">Data</a>
  |
  <a href="cookbook/">Cookbook</a>
</p>

IDiom is an autoregressive protein language model trained on IDiom-DB, a dataset of 54M intrinsically disordered protein regions (IDRs) curated from the AlphaFold Database. IDiom is trained in three model sizes, with 20M, 85M, and 300M parameters. This repository supports:

- **Generation** of standalone unprompted IDRs as well as IDRs conditioned on flanking protein context
- **Extraction** of sequence- and residue-level embeddings for IDRs
- **Post-training** via supervised fine-tuning and reinforcement learning with custom rewards
- **Model interpretability** and steering via IDiomSAE sparse autoencoders

![IDiom](assets/github_fig.png)

## Updates

- **2026-09-XX:** [IDiom v1.0.0 release](https://github.com/rotskoff-group/idiom/releases#release-v1.0.0), new preprint.
- **2026-04-11:** [IDiom v0.0.0 release](https://github.com/rotskoff-group/idiom/releases#release-v0.0.0), [Generative design of intrinsically disordered protein regions with IDiom](https://doi.org/10.64898/2026.04.10.717777), presented at the ICML GenBio Workshop 2026.


## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
  - [Sequence conventions](#sequence-conventions)
  - [Sequence generation](#sequence-generation)
  - [Extracting model embeddings](#extracting-model-embeddings)
  - [IDiomSAE](#idiomsae)
- [Cookbook: notebooks, post-training, and rewards](#cookbook-notebooks-post-training-and-rewards)
- [Models](#models)
- [Data](#data)
- [Citation](#citation)
- [License](#license)


## Installation

Please install the `v1` release directly from GitHub (Python ≥3.10):

```bash
pip install git+https://github.com/rotskoff-group/idiom.git@v1
```

To access the examples in `cookbook/`, please also clone the `v1` release:

```bash
git clone --branch v1 https://github.com/rotskoff-group/idiom.git
```

We welcome any contributions to this open source project. For development, clone the repository and install the package with its development dependencies:

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

Below, we provide several examples to get started with IDiom. More detailed examples and workflows are provided in the `cookbook/`.

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

For example, a FASTA record marking `QSSG` as the IDR is:

```fasta
>example_IDR_4-7
MEDQSSGACDE
```

<br>

## Sequence generation

IDiom enables the generation of standalone unprompted IDRs, as well as IDRs conditioned on flanking protein context. These flanking protein contexts are the protein residues preceding and following the IDR on its N-terminal and C-terminal sides. 

## Unprompted generation

Generate 10 unprompted IDRs:

```python
from idiom import IDiom

model = IDiom.from_pretrained("jxliu2/idiom-300M")
# The model weights will download from HuggingFace on the first use
sequences = model.generate_unprompted(n=10)
print(sequences)
```

Generate 10 unprompted IDRs within a length range. IDiom samples up to max_oversample * n candidates but may return fewer than n sequences within the requested length range:

```python
from idiom import IDiom

model = IDiom.from_pretrained("jxliu2/idiom-300M")
sequences = model.generate_unprompted(n=10, length_range=(80, 120), max_oversample=20)
print(sequences)
```

Sample with explicit temperature and top-p sampling (defaults when not specified: `temperature=1.0` and `top_p=None`, which disables top-p):

```python
from idiom import IDiom

model = IDiom.from_pretrained("jxliu2/idiom-300M")
sequences = model.generate_unprompted(n=10, temperature=0.8, top_p=0.9, seed=42)
print(sequences)
```

To have generated sequences directly output to a FASTA file, use `model.generate_unprompted_fasta("idrs.fasta", n=10)` or:

```bash
idiom_generate unprompted --model jxliu2/idiom-300M --n 10 --out idrs.fasta
```

## Prompted generation

For prompted generation, you must supply a protein sequence as well as the IDR span that you would like to re-generate. IDiom uses the specified IDR's preceding N-terminal residues and following C-terminal residues as the prompt for generating new IDRs. Please see [Sequence conventions](#sequence-conventions) for residue position indexing conventions.

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
regen_idrs = model.generate_prompted(seq, idr_start=118, idr_end=259, n=10)  # Standard Python indexing here
print(regen_idrs)  # Returns only the generated prompted IDRs
```

To generate prompted IDRs from an existing FASTA file containing protein sequences (see [Sequence conventions](#sequence-conventions) for FASTA file requirements), run this example from the cloned repository root to use [HP1α](cookbook/example_data/prompted_grpo/P45973.fasta) as an example sequence (IDR between residues 79–123):

```python
from idiom import IDiom

model = IDiom.from_pretrained("jxliu2/idiom-300M")
model.generate_prompted_fasta(
    "cookbook/example_data/prompted_grpo/P45973.fasta",
    "regen_hp1a.fasta",
    n=10,
    return_full=True,
)
# outputs regen_hp1a.fasta
```

When `return_full=True`, the output FASTA contains each generated IDR placed back within its full-length context, i.e. in bewteen its original prompting flanks. When `return_full=False`, the output FASTA contains just the prompt-generated IDRs. 

<!-- each generated IDR between its original prompting flanks in the output FASTA, while `return_full=False` just outputs the prompted IDRs in the FASTA. -->

<br>

## Extracting model embeddings

IDiom can extract embeddings from chosen model layers with mean pooling across the IDR `pool="mean"`, for every IDR residue `pool="none"`, and from only the final position `pool="last"`. **Embedding extraction returns only representations for IDR residues.** 

<!-- Embeddings can be extracted from the model X Y Z (mean pool, per residue, last) -->

### Embeddings of unprompted IDRs 

To extract embeddings without flanking context, pass in IDR sequences directly. Each sequence is treated as an unprompted IDR:

```python
from idiom import IDiom

model = IDiom.from_pretrained("jxliu2/idiom-300M")
idr_sequences = ["MSSGQSSQSPGSGQQQQSSG", "GSGSSQPSQGQSSGSSQQPN"]
# In this example, the entire sequence is the IDR
```

Then, extract the IDR embeddings:

```python
# Embedding layers are 0-based Transformer block indices
values, index = model.embed(idr_sequences, layers=[18], pool="none")[18]
print(values.shape)  # (40, 1024) one row per residue across both IDRs

pooled, _ = model.embed(idr_sequences, layers=[18], pool="mean")[18]
print(pooled.shape)  # (2, 1024) averaged over each IDR's residues

last, _ = model.embed(idr_sequences, layers=[18], pool="last")[18]
print(last.shape)  # (2, 1024) final residue representation for each IDR
```

### Embeddings of prompted IDRs 

To extract IDR embeddings with both flanks as prompting context, use the [`Record`](src/idiom/data/records.py) dataclass, which represents a single IDR data record. `Record`s take the full protein sequence and the indices of the IDR span. In this example, `QSSG` is the IDR, with `MED` and `ACDE` as its N- and C-terminal flanks:

```python
from idiom.data.records import Record

record = Record("protein1", full_seq="MEDQSSGACDE", idr_start=3, idr_end=7)  # IDR: QSSG
```

Then, extract the IDR embeddings: 

```python
values, index = model.embed(record, layers=[18], pool="none")[18]
print(values.shape)  # (4, 1024) embeddings for Q, S, S, G only
# Both flanks provide context, but their embeddings are not returned.

pooled, _ = model.embed(record, layers=[18], pool="mean")[18]
print(pooled.shape)  # (1, 1024) averaged over the four IDR residues

last, _ = model.embed(record, layers=[18], pool="last")[18]
print(last.shape)  # (1, 1024) representation of the final IDR residue, G
```

### Embeddings from FASTA files

To obtain IDR embeddings from a FASTA file, pass the FASTA path with [annotated IDR spans](#sequence-conventions) to `model.embed()`. Any present flanking residues are used as prompted context, and only IDR embeddings are returned.

```python
fasta = "cookbook/example_data/prompted_grpo/P45973.fasta"
values, index = model.embed(fasta, layers=[18], pool="mean")[18]
print(values.shape)  # (1, 1024); use pool="none" for (45, 1024)
```

To export embeddings from the command line:

```bash
idiom_extract --model jxliu2/idiom-300M \
    --fasta cookbook/example_data/prompted_grpo/P45973.fasta \
    --layers 18 --pool mean --out embeddings
```

This writes `embeddings/layer_18.npy` and `embeddings/layer_18_index.csv`.

<br>

## IDiomSAE

We provide a TopK sparse autoencoder, IDiomSAE, trained on the residual stream of layer-18 of 24 in IDiom-300M. IDiomSAE has k = 32 and a latent dimension of z = 16,384. We note that IDiomSAE was only trained on the IDiom activations of unprompted IDR residues. 

### Extracting SAE feature vectors 

To extract SAE feature vectors, pass in IDR sequences directly. Each residue's feature vector has dimension 16,384. 

```python
from idiom import IDiomSAE

sae = IDiomSAE.from_pretrained("jxliu2/idiomsae-300M-L18-k32")
idr_sequences = ["MSSGQSSQSPGSGQQQQSSG", "GSGSSQPSQGQSSGSSQQPN"]
```

Then, extract the IDR feature vectors:

```python
features, index = sae.encode(idr_sequences, pool="none")
print(features.shape)  # (40, 16384) one feature vector per IDR residue

pooled, accessions = sae.encode(idr_sequences, pool="mean")
print(pooled.shape)  # (2, 16384) averaged over each IDR's residues

peak, accessions = sae.encode(idr_sequences, pool="max")
print(peak.shape)  # (2, 16384) each feature's maximum activation in each IDR
```

Full-length protein sequences can also be passed in, but only the unprompted IDR will be used when extracting IDiom activations for IDiomSAE encoding: 

```python
from idiom.data.records import Record

record = Record("protein1", "MEDQSSGACDE", idr_start=3, idr_end=7)
features, index = sae.encode(record, pool="none")
print(features.shape)  # (4, 16384) features for QSSG
# Flanking context is not used for IDiom activation extraction when using sae.encode()
```

### Steering generation 

To steer the generation of IDRs using SAE features, run:

```python
# Example feature ID 1234 (of 16384)
steered = sae.steer_generate(feature=1234, strength=0.25, n=10)
```

To test whether each feature activates anywhere in each IDR:

```python
peak_features, accessions = sae.encode(idr_sequences, pool="max")
present = peak_features > 0  # [N_IDRs, num_latents] boolean feature presence
```

SAE features are returned only for IDR residues, with flanks excluded by this released model. Max-pooled values greater than zero indicate features active anywhere in an IDR.

For more complex SAE workflows, please see the notebooks for [inspecting SAE features](cookbook/notebooks/interpret_sae_features.ipynb), [finding enriched features](cookbook/notebooks/enriched_feature_signature.ipynb), and [using feature signatures as RL rewards](cookbook/notebooks/rl_with_sae_rewards.ipynb).

<br>


## Cookbook: notebooks, post-training, and rewards

The cookbook in `cookbook/` provides detailed examples and workflows for using and post-training IDiom. Detailed information can be found in the [cookbook readme](cookbook/README.md).

- [Notebooks](cookbook/notebooks/README.md): explore generation, prediction, embeddings, SAE analysis, and post-training.
- [Bash scripts](cookbook/scripts/README.md): run SFT, custom-reward GRPO, and SAE workflows.
- [Custom rewards](cookbook/rewards/README.md): define reinforcement learning rewards and connect external scorers.

<br>

## Models

IDiom models are hosted in our [Hugging Face collection](https://huggingface.co/collections/jxliu2/idiom).

| Model | Parameters | Architecture |
|---|---|---|
| [idiom-300M](https://huggingface.co/jxliu2/idiom-300M) | 302M | 24 layers, width 1024 |
| [idiom-85M](https://huggingface.co/jxliu2/idiom-85M) | 85M | 12 layers, width 768 |
| [idiom-20M](https://huggingface.co/jxliu2/idiom-20M) | 18.9M | 6 layers, width 512 |
| [idiomsae-300M-L18-k32](https://huggingface.co/jxliu2/idiomsae-300M-L18-k32) | — | SAE on layer 18 of idiom-300M with 16,384 latents, k=32 |

<br>

## Data

IDiom-DB is a dataset of 54M IDRs curated from the AlphaFold Database (curation details are provided in the manuscript). IDiom-DB can be found on Hugging Face at [jxliu2/idiom-db](https://huggingface.co/datasets/jxliu2/idiom-db). The training split contains 54M records (~24 GB), and the validation and test splits contain 270k records each (130 MB each).

Quick download instructions (for details see the Hugging Face repository):

```bash
# Download all training, validation, and test splits
hf download jxliu2/idiom-db --repo-type dataset --include "idiom-db/idiom-db-v1_*.fasta"

# Download only the validation split
hf download jxliu2/idiom-db --repo-type dataset --include "idiom-db/idiom-db-v1_validation.fasta"
```

<br>

## Citation

```bibtex
@article{,
  author = {},
  title = {},
  journal = {},
  year = {},
  doi = {},
  URL = {},
}
```

<br>

## License

MIT license. The pretraining corpus is CC BY 4.0, from AFDB/UniProt. Data examples follow the licenses of their original sources.
