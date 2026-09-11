# Bash scripts

See the [cookbook](../README.md#bash-scripts) for details.

## IDR generation

| Script | Task |
|---|---|
| [generate_unprompted.bash](generation/generate_unprompted.bash) | Generate unprompted IDRs |
| [generate_prompted.bash](generation/generate_prompted.bash) | Generate IDRs prompted with provided flanking contexts |

## SAE analysis

| Script | Task |
|---|---|
| [build_feature_dataset.bash](sae/build_feature_dataset.bash) | Extract per-residue SAE features from sequences |
| [feature_enrichment.bash](sae/feature_enrichment.bash) | Find enriched features and export top features |
| [train_sae.bash](sae/train_sae.bash) | Train and export an SAE |

## Training

| Script | Task |
|---|---|
| [sft.bash](training/sft.bash) | Supervised fine-tuning on a set of sequences |
| [pretrain.bash](training/pretrain.bash) | Pretrain IDiom from scratch |
| [cookbook/scripts/training/grpo/](training/grpo/) | Reinforcement learning with custom rewards, SAE features, or external scorers |

### GRPO examples

Each `.bash` script loads reward terms from the matching `.yaml` file in
`cookbook/scripts/training/grpo/` (for example, `combined.bash` loads `combined.yaml`).
Edit rewards, scorer commands, shaping, targets, and weights in YAML; edit training settings
and runtime paths in Bash. The scripts export variables used by `${oc.env:...}` entries
in YAML and pass the loaded terms as a Hydra command-line override.

| Script | Reward |
|---|---|
| [sae_features.bash](training/grpo/sae_features.bash) | Reinforcement learning with sparse autoencoder features (RL-SAE) |
| [custom_reward.bash](training/grpo/custom_reward.bash) | Use a custom defined reward |
| [custom_scorer.bash](training/grpo/custom_scorer.bash) | Run a custom external scorer as a subprocess in a separate environment |
| [sparrow.bash](training/grpo/sparrow.bash) | Target an IDR sequence property such as radius of gyration using [SPARROW](https://github.com/idptools/sparrow) |
| [finches.bash](training/grpo/finches.bash) | Match ProTalpha interaction strength with H1.0 CTD using [FINCHES](https://github.com/idptools/finches) |
| [protgps.bash](training/grpo/protgps.bash) | Increase compartment localization probability using [ProtGPS](https://github.com/pgmikhael/protgps) |
| [paddle.bash](training/grpo/paddle.bash) | Increase predicted transcriptional activation strength using [PADDLE](https://github.com/asanborn/PADDLE) |
| [starling.bash](training/grpo/starling.bash) | Target predicted ensemble dimensions using [STARLING](https://github.com/idptools/starling/) |
| [prompted_linker_rg.bash](training/grpo/prompted_linker_rg.bash) | Target linker dimensions within fixed flanks |
| [combined.bash](training/grpo/combined.bash) | Combine FINCHES and ProtGPS |
