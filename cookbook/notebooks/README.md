# Notebooks

Run these notebooks locally or in Colab. Each installs IDiom if needed and downloads its input data.
In Colab, select **Runtime → Change runtime type → GPU**. Edit the parameter cell to use your own
model, SAE, or sequences.

Run cells from top to bottom. Model weights and data are cached after the first download.
See [example data](../example_data/) for input conventions and provenance.

| Notebook | Task | Colab |
|---|---|---|
| [`generate_and_embed.ipynb`](generate_and_embed.ipynb) | Generate IDRs, extract embeddings, and score perplexity | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/generate_and_embed.ipynb) |
| [`sae_features.ipynb`](sae_features.ipynb) | Build feature datasets and inspect SAE activations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/sae_features.ipynb) |
| [`feature_enrichment.ipynb`](feature_enrichment.ipynb) | Find enriched features and export a GRPO signature | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/feature_enrichment.ipynb) |

Use the signature exported by `feature_enrichment.ipynb` with
[sae_features.bash](../scripts/grpo/sae_features.bash) to train toward those features. Set
`FEATURES`, `SIGNATURE`, and `CASE` in the script to match the notebook's exported signature;
see [scripts/](../scripts/) for execution instructions.

## Run size and outputs

- **Generation:** defaults to ten sequences and saves FASTAs and embeddings.
- **SAE features:** defaults to the full validation split (approximately 271k records).
  Set `MAX_RECORDS=1000` for a smaller first run. `BATCH_SIZE` limits GPU memory per forward;
  `MAX_RECORDS` limits total feature-dataset size and encoding work.
- **Enrichment:** defaults to at most 128 positives and approximately 512 background records.
  Increase `MAX_POSITIVE` and `MAX_BACKGROUND` for a larger analysis. A signature is written only
  when features pass the filters; these demonstration defaults do not reproduce the released signatures.

Each notebook prints its output paths. Use a fresh `OUT_DIR` for a new analysis so old outputs
cannot be mistaken for current results. In Colab, download outputs before discarding the runtime.
