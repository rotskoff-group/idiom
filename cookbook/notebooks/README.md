# Notebooks

Run these notebooks locally or in Colab. Each installs IDiom if needed and downloads example data.
In Colab, select **Runtime → Change runtime type → GPU**. Edit the parameter cell to use your own
model, SAE, or sequences.

| Notebook | Task | Colab |
|---|---|---|
| [`generate_and_embed.ipynb`](generate_and_embed.ipynb) | Generate IDRs, redesign proteins, and extract embeddings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/generate_and_embed.ipynb) |
| [`sae_features.ipynb`](sae_features.ipynb) | Analyze SAE features and steer generation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/sae_features.ipynb) |
| [`feature_enrichment.ipynb`](feature_enrichment.ipynb) | Find enriched features and export a GRPO signature | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/feature_enrichment.ipynb) |

Use the signature exported by `feature_enrichment.ipynb` with
[sae_features.bash](../scripts/grpo/sae_features.bash) to train toward those features.
