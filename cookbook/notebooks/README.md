# IDiom notebooks

| Notebook | Function | Colab |
|---|---|---|
| [Generate IDRs](generate_idrs.ipynb) | Generate standalone IDRs or replacements within protein flanks, using known or Metapredict-predicted IDR boundaries. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1.0.0/cookbook/notebooks/generate_idrs.ipynb) |
| [Extract embeddings](extract_embeddings.ipynb) | Extract pooled and per-residue model embeddings, plus SAE activations and feature-presence vectors. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1.0.0/cookbook/notebooks/extract_embeddings.ipynb) |
| [Enriched SAE features](enriched_sae_features.ipynb) | Compare positive and background IDR sets to identify enriched SAE features and export a signature for RL-SAE. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1.0.0/cookbook/notebooks/enriched_sae_features.ipynb) |
| [RL custom](rl_custom.ipynb) | Define a custom sequence reward, train with GRPO, and compare scores before and after training. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1.0.0/cookbook/notebooks/rl_custom.ipynb) |
| [RL SAE](rl_sae.ipynb) | Train with SAE feature signatures as rewards, optionally combine signatures, and compare feature coverage. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1.0.0/cookbook/notebooks/rl_sae.ipynb) |
| [SFT](sft.ipynb) | Fine-tune IDiom on an IDR sequence set and generate sequences from the adapted model. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1.0.0/cookbook/notebooks/sft.ipynb) |

See the [cookbook guide](../README.md#notebooks) for setup and saving results,
and [example data](../README.md#example-data) for the bundled inputs.
Input and parameter details are in each notebook.
