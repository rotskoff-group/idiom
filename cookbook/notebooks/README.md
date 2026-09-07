# Notebooks

Run these notebooks locally or in Colab. Each installs IDiom if needed and downloads its input data.
In Colab, select **Runtime → Change runtime type → GPU**. Edit the parameter cell to use your own
model, SAE, or sequences.

Run cells from top to bottom. Model weights and data are cached after the first download.
The defaults use small batches. SAE analysis uses the full held-out validation split from
Hugging Face; set `MAX_RECORDS` to limit memory and runtime for a smaller run.
Generation saves FASTAs and embeddings, SAE analysis saves a feature dataset, and enrichment
saves a signature when features pass its filters. Each notebook prints its output paths;
use the Colab file browser to download files before the runtime is discarded.

Example FASTAs, sequence conventions, and provenance are described in
[example_data/](../example_data/). Set the notebook's input paths to use your own data.

| Notebook | Task | Colab |
|---|---|---|
| [`generate_and_embed.ipynb`](generate_and_embed.ipynb) | Generate IDRs, extract embeddings, and score perplexity | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/generate_and_embed.ipynb) |
| [`sae_features.ipynb`](sae_features.ipynb) | Build feature datasets and inspect SAE activations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/sae_features.ipynb) |
| [`feature_enrichment.ipynb`](feature_enrichment.ipynb) | Find enriched features and export a GRPO signature | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/feature_enrichment.ipynb) |

Use the signature exported by `feature_enrichment.ipynb` with
[sae_features.bash](../scripts/grpo/sae_features.bash) to train toward those features. Set
`FEATURES`, `SIGNATURE`, and `CASE` in the script to match the notebook's exported signature;
see [scripts/](../scripts/) for execution instructions.
