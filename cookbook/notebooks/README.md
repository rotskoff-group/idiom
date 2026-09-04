# Notebooks

Three walkthroughs, in order. Each installs IDiom itself and pulls its inputs from
[`jxliu2/idiom-data`](https://huggingface.co/datasets/jxliu2/idiom-data), so no clone is needed —
click a badge and run. In Colab set **Runtime → Change runtime type → GPU** first; `DEVICE = "auto"`
falls back to CPU.

| Notebook | Covers | |
|---|---|---|
| [`generate_and_embed.ipynb`](generate_and_embed.ipynb) | unprompted and prompted generation, redesigning a real protein's IDR, residual-stream embeddings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/generate_and_embed.ipynb) |
| [`sae_features.ipynb`](sae_features.ipynb) | which SAE features fire on a sequence, and steering generation along one | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/sae_features.ipynb) |
| [`feature_enrichment.ipynb`](feature_enrichment.ipynb) | which features are enriched in your own set, their residue grammar, and the signature that turns them into an RL target | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/feature_enrichment.ipynb) |

Edit the parameter cell to point one at your own model, SAE, or sequences.
`feature_enrichment.ipynb` feeds [`../scripts/grpo/sae_features.bash`](../scripts/grpo/sae_features.bash):
the notebook writes a signature, the script post-trains a model to reproduce it.
