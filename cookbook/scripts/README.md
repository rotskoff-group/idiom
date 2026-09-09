# Scripts

Editable Bash examples for generation, feature extraction, and training. Start with the
[installation instructions](../../README.md#installation); each script uses your active environment.

## Choose a script

| Script | Task |
|---|---|
| [generate.bash](generate.bash) | Generate de novo IDRs to FASTA |
| [generate_prompted.bash](generate_prompted.bash) | Replace IDRs in flanking protein context |
| [feature_dataset.bash](feature_dataset.bash) | Build a per-residue SAE feature dataset |
| [feature_enrichment.bash](feature_enrichment.bash) | Compare IDR sets and export an enriched SAE feature signature |
| [sft.bash](sft.bash) | Fine-tune on a sequence set |
| [train_sae.bash](train_sae.bash) | Train and export an SAE |
| [pretrain.bash](pretrain.bash) | Pretrain IDiom from scratch |
| [grpo/](grpo/) | Post-train with feature signatures, custom rewards, or external scorers |
| [grpo/prompted_linker_rg.bash](grpo/prompted_linker_rg.bash) | Redesign a marked linker toward a target Rg in full-protein context |

See the [reward guide](../rewards/README.md#examples) for the GRPO objectives and scorer setup.
Demo inputs and their provenance are described in [example_data/](../example_data/).
The [SAE notebook](../notebooks/sae_features.ipynb) shows how to inspect a feature dataset.

## Running scripts

Scripts use IDiom from your active Python environment and the clone for cookbook files.
When installing and cloning
separately, use the same tag or commit to keep the examples matched to the installed API.

1. Activate the environment where IDiom is installed.
2. Edit `REPO`, `OUT`, input paths, and run settings in the selected script.
3. Run it from the clone:

```bash
bash cookbook/scripts/sft.bash
```

Training commands list every IDiom config parameter explicitly, including null values, reward-factory
arguments, and Hydra run/sweep paths. Defaults live in `src/idiom/configs/`; SFT reads its model
architecture from `init_from`. Hydra’s framework settings can be inspected with `--cfg hydra`,
and the composed training settings with `--cfg job`. These flags print configuration without training.

External-scorer examples also require `uv` (`python -m pip install uv`). Their
`uv run --script` commands install each scorer's dependencies separately; no `uv sync` is needed
for the pip workflow.

Most training examples use one GPU; pretraining uses eight, and the STARLING example uses two.
Check the selected script for resource estimates. W&B logging is offline by default; run
`wandb login` and change the script's `WANDB_MODE` to `online` for live logging.

## Feature enrichment

`idiom_feature_enrichment` accepts `--positive` and an optional local `--background` FASTA.
Omitting the background downloads `training_sequences/validation.fasta` from `jxliu2/idiom-data`.
Headers use `_IDR_x-y` spans; missing or unusable spans treat the whole sequence as the IDR.
The command uses all valid positives by default, excludes exact positive IDR matches from the
background, and samples an approximately length-matched background (default target: 10,000).
Use `--max-positive` and `--max-background` for smaller runs; building holds records and
activation arrays in host memory. A GPU is recommended.

Use a new or empty `--out` directory. Outputs are `fd_positive/`, `fd_background/`,
`enrichment.tsv` (all features, including selection flags), `run.json` (settings and counts),
and `signature.json` when features pass the filters. Untested features have `nan` FDR values.
The signature case defaults to `top<TOP_N>`; pass `--case` to override it.
Threshold options are `--fdr-alpha`, `--log2or-floor`, `--prev-pos-floor`, and
`--min-total-fire`; `--keep-boundary` disables the default boundary-feature filter.
Use the [enrichment notebook](../notebooks/feature_enrichment.ipynb) for interactive plots
and sequence logos. The demonstration settings do not reproduce the released signatures.

<details>
<summary>Schedulers and multiple GPUs</summary>

Submit a script with your scheduler, ensuring the job inherits the environment containing IDiom:

```bash
sbatch --gpus-per-node=1 --cpus-per-task=8 --time=12:00:00 \
    --wrap "bash $PWD/cookbook/scripts/grpo/sparrow.bash"
```

For single-node pretraining and SFT, launch the Bash script once: IDiom uses Lightning's local
process launcher with `trainer.devices` GPUs. The GRPO and SAE examples train on one GPU;
STARLING's second GPU runs its scorer.

For multi-node pretraining or SFT, launch the training command directly with `srun`, one task
per GPU. Set `+trainer.num_nodes` to the allocated node count and `trainer.devices` to the
GPUs **per node**. With more than one node, IDiom allows Lightning to detect the external
launcher and use its ranks. For example, save this as a batch script and submit with `sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=idiom-pretrain
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
set -euo pipefail

# Activate the same IDiom environment on every node; use shared paths for data and output
source /shared/path/to/idiom/.venv/bin/activate
export WANDB_MODE=offline
srun idiom_train_autoreg \
    data.train_fasta=/shared/path/to/train.fasta \
    data.val_fasta=/shared/path/to/validation.fasta \
    trainer.accelerator=gpu trainer.devices=4 +trainer.num_nodes=2 \
    out_dir=/shared/path/to/run hydra.run.dir=/shared/path/to/run/hydra
```

Adapt partition, time, and memory requests to your cluster. For SFT, add `--config-name sft`
and `init_from=<model>`. Global autoregressive batch size is
`data.batch_size × devices × nodes × accumulate_grad_batches` (1,024 in this pretraining example).
This multi-node recipe applies to autoregressive training; the GRPO and SAE cookbook examples
cover single-node runs.

</details>

## Checkpoints and data

Pretraining starts from scratch. SFT and GRPO load the model specified by `init_from`; SAE
training loads a frozen host model from `model_ckpt`. These accept a Hub model ID, a released
directory, or a Lightning checkpoint.

- Pretraining and SFT scripts automatically resume from `$OUT/checkpoints/last.ckpt` when present.
- GRPO saves a final-step checkpoint by default. Set `trainer.checkpoint_every` for periodic
  checkpoints and `last.ckpt`; set `resume_from` to resume a saved checkpoint.
- SAE training exports `sae_config.json` and `sae.safetensors` to `OUT` on completion.
  Its trainer uses Lightning's checkpoint defaults; `resume_from` accepts a Lightning checkpoint,
  not the exported SAE weights.

FASTA training builds a memory-mapped `<fasta>.idiomstore/` sidecar on first use. You can also
build it ahead of time with `idiom_build_store --fasta /path/to/corpus.fasta`.

## Generation and analysis details

Generation caps `max_new_tokens` at the remaining model context, accounting for flanks and FIM
markers, and warns when reducing the requested budget. Prompts exceeding the context are rejected.
Length filtering may return fewer sequences if it reaches the sampling limit. Use `temperature=0`
for greedy generation; `seed` controls stochastic sampling. Invalid generation options raise `ValueError`.

Generation and SAE steering use batches of eight by default. Set `batch_size` in Python
or `--batch-size` in the generation CLI to adjust memory use. Seeded results are
reproducible for a fixed batch size.

`idiom_extract --ckpt` remains an alias for `--model`. Per-residue embedding metadata
includes `record_idx`, `accession`, `source_pos` (0-based in the original protein),
`residue`, and `is_idr`. Mean SAE encoding preserves separate records with repeated accessions.

SAE steering supports `add_direction`, `clamp`, and `ablate`. For ablation, `strength=0`
leaves activations unchanged and `strength=1` removes the selected features’ decoder
contributions.

## GRPO disorder monitoring

GRPO logs `train/metapredict_disorder` every optimizer step using metapredict 3.0.2's V3
network on CPU. It averages per-residue disorder scores within each generated sequence,
then averages nonempty sequences across all gradient-accumulation microbatches and ranks.
This diagnostic is separate from the reward. Empty completions are excluded and reported as
`train/metapredict_empty_fraction`; an entirely empty step reports disorder 0 and empty fraction 1.
Set `grpo.track_disorder=false` to disable prediction. Metrics use the existing Lightning/W&B
logger (offline W&B runs still require syncing to appear online).
