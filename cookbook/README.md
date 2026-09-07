# Cookbook

Examples for generation, interpretation, and training. Start with the
[installation instructions](../README.md#installation).

| Guide | Use it to |
|---|---|
| [Notebooks](notebooks/) | Generate and embed IDRs, inspect SAE features, or compare sequence sets locally or in Colab |
| [Scripts](scripts/) | Run generation, feature analysis, pretraining, SFT, SAE training, or GRPO jobs |
| [Rewards](rewards/) | Define objectives, combine shaping and weights, and use external predictors |
| [Example data](example_data/) | Find demo FASTAs, sequence conventions, and source citations |

For a first walkthrough, open [generate_and_embed.ipynb](notebooks/generate_and_embed.ipynb).
To design toward SAE features, use [feature_enrichment.ipynb](notebooks/feature_enrichment.ipynb)
to export a signature, then pass it to [grpo/sae_features.bash](scripts/grpo/sae_features.bash).
