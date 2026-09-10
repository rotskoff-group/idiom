# IDiom Cookbook

In this cookbook, we provide Jupyter notebooks, custom rewards, and Bash scripts to demonstrate how to use IDiom.
<!--
show how to generate, analyze,
and train with IDiom. See the [installation instructions](../README.md#installation) for setup
and [example data](example_data/README.md) for demo inputs, sequence conventions, and sources. -->

## Notebooks

In [notebooks](notebooks/) we provide three examples for generating and embedding sequences as well as sparse autoencoder analysis. Notebooks can be run locally or in Colab.

| Notebook | Task | Colab |
|---|---|---|
| [`generate_and_embed.ipynb`](notebooks/generate_and_embed.ipynb) | Generate IDRs, extract embeddings, and score perplexity | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/generate_and_embed.ipynb) |
| [`sae_features.ipynb`](notebooks/sae_features.ipynb) | Build feature datasets and browse feature activation patterns | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/sae_features.ipynb) |
| [`feature_enrichment.ipynb`](notebooks/feature_enrichment.ipynb) | Find enriched features within a set of sequences | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/feature_enrichment.ipynb) |



## Bash scripts

In [scripts](scripts/) we provide Bash scripts to demonstrate various workflows with IDiom. A brief description of the example scripts is given below.

| Script | Task |
|---|---|
| [generate.bash](scripts/generate.bash) | Generate unprompted IDRs |
| [generate_prompted.bash](scripts/generate_prompted.bash) | Generate IDRs prompted with provided flanking contexts |
| [feature_dataset.bash](scripts/feature_dataset.bash) | Extract per-residue SAE features from sequences |
| [feature_enrichment.bash](scripts/feature_enrichment.bash) | Find enriched features and export top features |
| [sft.bash](scripts/sft.bash) | Supervised fine-tuning on a set of sequences |
| [train_sae.bash](scripts/train_sae.bash) | Train and export an SAE |
| [pretrain.bash](scripts/pretrain.bash) | Pretrain IDiom from scratch |
| [grpo/](scripts/grpo) | Train with custom rewards, SAE features, or external scorers |

GRPO examples cover the following rewards:

| Script | Reward |
|---|---|
| [sae_features.bash](scripts/grpo/sae_features.bash) | Reinforcement learning with sparse autoencoder features |
| [custom_reward.bash](scripts/grpo/custom_reward.bash) | Use a custom Python reward |
| [custom_scorer.bash](scripts/grpo/custom_scorer.bash) | Run a custom scorer as a subprocess in a separate environment |
| [sparrow.bash](scripts/grpo/sparrow.bash) | Target an IDR sequence property such as radius of gyration using [SPARROW](https://github.com/idptools/sparrow) |
| [finches.bash](scripts/grpo/finches.bash) | Match ProTalpha interaction strength with H1.0 CTD using [FINCHES](https://github.com/idptools/finches) |
| [protgps.bash](scripts/grpo/protgps.bash) | Increase compartment localization probability using [ProtGPS](https://github.com/pgmikhael/protgps) |
| [paddle.bash](scripts/grpo/paddle.bash) | Increase predicted transcriptional activation strength using [PADDLE](https://github.com/asanborn/PADDLE) |
| [starling.bash](scripts/grpo/starling.bash) | Target predicted ensemble dimensions using [STARLING](https://github.com/idptools/starling/) |
| [prompted_linker_rg.bash](scripts/grpo/prompted_linker_rg.bash) | Target linker dimensions within fixed flanks |
| [combined.bash](scripts/grpo/combined.bash) | Combine FINCHES and ProtGPS |


## Custom rewards


In [rewards](rewards/) we provide scripts to set up external scorers and custom rewards for reinforcement learning post-training.


| File | Provides |
|---|---|
| [custom_rewards.py](rewards/custom_rewards.py) | Custom reward and reward shaping |
| [custom_scorer.py](rewards/scorers/custom_scorer.py) | Template for a scorer in a separate environment |
| [sparrow.py](rewards/scorers/sparrow.py) | [SPARROW](https://github.com/idptools/sparrow): sequence properties, including radius of gyration |
| [finches.py](rewards/scorers/finches.py) | [FINCHES](https://github.com/idptools/finches): self- or partner-interaction epsilon |
| [protgps.py](rewards/scorers/protgps.py) | [ProtGPS](https://github.com/pgmikhael/protgps): subcellular compartment localization probabilities |
| [paddle.py](rewards/scorers/paddle.py) | [PADDLE](https://github.com/asanborn/PADDLE): predicted transcriptional activation strength |
| [starling.py](rewards/scorers/starling.py) | [STARLING](https://github.com/idptools/starling/): ensemble radius of gyration or end-to-end distance |


<!-- | [_scorer_protocol.py](rewards/scorers/_scorer_protocol.py) | Shared request/response helper for external scorers | -->


<!-- ## Example data

Demo inputs for the workflows above. See the [data guide](example_data/README.md) for
download instructions, sequence conventions, and sources.

| Directory | Contents |
|---|---|
| [protgps/](example_data/protgps) | IDRs from six subcellular compartments |
| [effector/](example_data/effector) | Activation and repression domain IDRs |
| [disprot/](example_data/disprot) | Held-out proteins with annotated IDRs and flanking context |
| [sae_features/](example_data/sae_features) | Example SAE signatures |
| [prompted_grpo/](example_data/prompted_grpo) | HP1α (P45973), residues 79–123 marked for redesign; from DisProt |
| [finches/](example_data/finches) | ProTalpha and H1.0 C-terminal reference constructs from FINCHES Fig. 5 | -->
