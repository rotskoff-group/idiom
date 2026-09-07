# Cookbook

Examples for generation, interpretation, training, and reward definition.

Please first install the IDiom package: [installation](../README.md#installation).

# Notebooks

Interactive walkthroughs that run locally or in Colab:

- Generate IDRs, extract embeddings, and score perplexity.
- Build SAE feature datasets and inspect residue-level activations.
- Find enriched features against a background and export a signature.

See [notebooks/](notebooks/) for the walkthroughs and Colab links.

# Scripts

Editable Bash examples for running jobs on your own data:

- Generate unprompted or prompted IDRs and build feature datasets.
- Pretrain IDiom, fine-tune it, or train an SAE.
- Run GRPO post-training with a chosen reward objective.

See [scripts/](scripts/) for the script index, setup, and execution instructions.

# Rewards

Guides and templates for defining GRPO objectives:

- Combine raw rewards with shaping functions and weights.
- Write custom Python rewards or scorers with isolated dependencies.
- Use SAE feature signatures and external predictors such as FINCHES, ProtGPS, and sparrow.

See [rewards/](rewards/) for configuration, scorer examples, and troubleshooting.
