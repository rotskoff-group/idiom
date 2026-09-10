# Bash scripts

| Script | Task |
|---|---|
| [generate.bash](generate.bash) | Generate unprompted IDRs |
| [generate_prompted.bash](generate_prompted.bash) | Generate IDRs prompted with provided flanking contexts |
| [feature_dataset.bash](feature_dataset.bash) | Extract per-residue SAE features from sequences |
| [feature_enrichment.bash](feature_enrichment.bash) | Find enriched features and export top features |
| [sft.bash](sft.bash) | Supervised fine-tuning on a set of sequences |
| [train_sae.bash](train_sae.bash) | Train and export an SAE |
| [pretrain.bash](pretrain.bash) | Pretrain IDiom from scratch |
| [grpo/](grpo) | Train with custom rewards, SAE features, or external scorers |

GRPO examples cover the following rewards:

| Script | Reward |
|---|---|
| [sae_features.bash](grpo/sae_features.bash) | Reinforcement learning with sparse autoencoder features |
| [custom_reward.bash](grpo/custom_reward.bash) | Use a custom Python reward |
| [custom_scorer.bash](grpo/custom_scorer.bash) | Run a custom scorer as a subprocess in a separate environment |
| [sparrow.bash](grpo/sparrow.bash) | Target an IDR sequence property such as radius of gyration using [SPARROW](https://github.com/idptools/sparrow) |
| [finches.bash](grpo/finches.bash) | Match ProTalpha interaction strength with H1.0 CTD using [FINCHES](https://github.com/idptools/finches) |
| [protgps.bash](grpo/protgps.bash) | Increase compartment localization probability using [ProtGPS](https://github.com/pgmikhael/protgps) |
| [paddle.bash](grpo/paddle.bash) | Increase predicted transcriptional activation strength using [PADDLE](https://github.com/asanborn/PADDLE) |
| [starling.bash](grpo/starling.bash) | Target predicted ensemble dimensions using [STARLING](https://github.com/idptools/starling/) |
| [prompted_linker_rg.bash](grpo/prompted_linker_rg.bash) | Target linker dimensions within fixed flanks |
| [combined.bash](grpo/combined.bash) | Combine FINCHES and ProtGPS |

Example bash scripts using IDiom installed in your active Python environment. First follow the [installation instructions](../../README.md#installation).

## Running scripts

Activate your IDiom environment, edit `REPO`, `OUT`, inputs, and settings in the selected
script, then run it from the clone:

```bash
bash cookbook/scripts/sft.bash --cfg job  # Inspect training configuration
bash cookbook/scripts/sft.bash
```

Most training examples use one GPU; pretraining uses eight and STARLING uses two.
W&B logging defaults to offline. External scorers require `uv` (`python -m pip install uv`)
and download their dependencies and weights on first use. See the
[reward guide](../rewards/README.md) for GRPO examples and scorer setup.

## Checkpoints and data

- Pretraining and SFT automatically resume from `$OUT/checkpoints/last.ckpt` when present.
- GRPO saves a final checkpoint; set `trainer.checkpoint_every` for periodic checkpoints
  and `resume_from` to resume.
- SAE training exports `sae_config.json` and `sae.safetensors`; resuming requires a Lightning checkpoint.

FASTA training creates a `<fasta>.idiomstore/` sidecar on first use.
See [example data](../example_data/) for inputs and sequence conventions.

## Feature enrichment

Edit the positive FASTA and optional background in [feature_enrichment.bash](feature_enrichment.bash).
Use a fresh output directory. The run writes feature datasets, `enrichment.tsv`, `run.json`,
and `signature.json` if features pass the filters. Set `FEATURES`, `SIGNATURE`, and `CASE` in
[grpo/sae_features.bash](grpo/sae_features.bash) to train with that signature.
The [enrichment notebook](../notebooks/feature_enrichment.ipynb) provides interactive analysis.

## Prompted linker redesign

[grpo/prompted_linker_rg.bash](grpo/prompted_linker_rg.bash) uses HP1α residues 79–123 as a
linker redesign example. For another protein, set `FASTA` to a single full-length record
with an `_IDR_x-y` header, `TARGET_LENGTH` to `y - x + 1`, and `TARGET_RG` to your target.
Rewards measure the generated linker alone. Run the script, then generate full proteins:

```bash
idiom_generate prompted --model /path/to/final.ckpt \
    --fasta cookbook/example_data/prompted_grpo/P45973.fasta \
    --out redesigned.fasta --n 32 --return-full --max-new-tokens 96
```

`--return-full` preserves the flanks and updates the IDR coordinates.

## Generation and analysis details

Generation must fit the model context, including flanks and markers. Length filtering can
return fewer sequences than requested. Adjust `--batch-size` for memory use; seeded results
are reproducible for a fixed batch size. Consult each script and the command's `--help` for options.
