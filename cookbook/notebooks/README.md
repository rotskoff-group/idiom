# Notebooks

Run these notebooks locally or in Colab. Each installs IDiom if needed and downloads its input data.
In Colab, select **Runtime → Change runtime type → GPU**. Edit the parameter cell to use your own
model, SAE, or sequences.

Run cells from top to bottom. Model weights and data are cached after the first download.
See [example data](../example_data/) for input conventions and provenance.

| Notebook | Task | Colab |
|---|---|---|
| [`generate_and_embed.ipynb`](generate_and_embed.ipynb) | Generate IDRs, extract embeddings, and score perplexity | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/generate_and_embed.ipynb) |
| [`sae_features.ipynb`](sae_features.ipynb) | Build feature datasets and browse activation-shaded sequence examples | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/sae_features.ipynb) |
| [`feature_enrichment.ipynb`](feature_enrichment.ipynb) | Find enriched features and export a GRPO signature | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/feature_enrichment.ipynb) |

Use the signature exported by `feature_enrichment.ipynb` with
[sae_features.bash](../scripts/grpo/sae_features.bash) to train toward those features. Set
`FEATURES`, `SIGNATURE`, and `CASE` in the script to match the notebook's exported signature;
see [scripts/](../scripts/) for execution instructions.

## Run size and outputs

- **Generation:** shows ten complete sequences per generation example and saves FASTAs and embeddings.
- **SAE features:** defaults to the first 10,000 validation records with `BATCH_SIZE=32`.
  Lower `BATCH_SIZE` if GPU memory is limited; set `MAX_RECORDS=None` to process the full input.
  `BATCH_SIZE` controls the number of sequences per forward pass;
  `MAX_RECORDS` limits total feature-dataset size and encoding work. The inline gallery shows three
  top-activating sequences for four curated IDR-pattern features; edit `FEATURE_IDS`, `N_EXAMPLES`,
  or `FEATURE_ID`. Set `FEATURE_IDS=None` to show the top `N_FEATURES` by activation.
- **Enrichment:** uses NPC IDRs, with at most 128 positives and approximately 512 background records.
  Increase `MAX_POSITIVE` and `MAX_BACKGROUND` for a larger analysis. A signature is written only
  when features pass the filters; these demonstration defaults do not reproduce the released signatures.

Each notebook prints its output paths. Use a fresh `OUT_DIR` for a new analysis so old outputs
cannot be mistaken for current results. In Colab, download outputs before discarding the runtime.
