# Cookbook

Examples for generation, interpretation, and training. Start with the [installation instructions](../README.md#installation).

| Task | Example |
|---|---|
| Generate sequences and extract embeddings | [generate_and_embed.ipynb](notebooks/generate_and_embed.ipynb) |
| Analyze SAE features and steer generation | [sae_features.ipynb](notebooks/sae_features.ipynb) |
| Find enriched features in a sequence set | [feature_enrichment.ipynb](notebooks/feature_enrichment.ipynb) |
| Train toward an SAE feature signature | [sae_features.bash](scripts/grpo/sae_features.bash) |
| Train with a custom reward | [custom_reward.bash](scripts/grpo/custom_reward.bash) |
| Train with an external scorer | [Reward examples](rewards/README.md#examples) |
| Fine-tune on a sequence set | [sft.bash](scripts/sft.bash) |
| Train an SAE | [train_sae.bash](scripts/train_sae.bash) |
| Pretrain IDiom | [pretrain.bash](scripts/pretrain.bash) |

[Usage reference](usage.md): FASTA output, perplexity, feature datasets, and model export.

## Running notebooks

Open a [notebook locally or in Colab](notebooks/). Each installs IDiom if needed and downloads
example data. Edit its parameter cell to use your own inputs. A GPU is recommended.

For SAE-guided GRPO, run `feature_enrichment.ipynb` first. Set `FEATURES`, `SIGNATURE`, and `CASE`
in `scripts/grpo/sae_features.bash` to match the notebook's exported signature.

## Running scripts

Scripts use IDiom from your active Python environment and the clone for cookbook files.
Either [installation workflow](../README.md#installation) works. When installing and cloning
separately, use the same tag or commit to keep the examples matched to the installed API.

1. Activate the environment where IDiom is installed.
2. Edit `REPO`, `OUT`, input paths, and run settings in the selected script.
3. Run it from the clone:

```bash
bash cookbook/scripts/sft.bash
```

External-scorer examples also require `uv` (`python -m pip install uv`). Their
`uv run --script` commands install each scorer's dependencies separately; no `uv sync` is needed
for the pip workflow.

Most training examples use one GPU; pretraining uses eight, and the STARLING example uses two.
Check the selected script for resource estimates. W&B logging is offline by default; run
`wandb login` and change the script's `WANDB_MODE` to `online` for live logging.

<details>
<summary>Advanced execution: schedulers, multiple GPUs, and checkpoints</summary>

Submit a script with your scheduler, ensuring the job inherits the environment containing IDiom:

```bash
sbatch --gpus-per-node=1 --cpus-per-task=8 --time=12:00:00 \
    --wrap "bash $PWD/cookbook/scripts/grpo/sparrow.bash"
```

On one node, Lightning launches processes according to `trainer.devices`. For multi-node runs,
use `srun` with one task per GPU and set `+trainer.num_nodes=<N>`. For autoregressive training,
global batch size is `data.batch_size × devices × nodes × accumulate_grad_batches`.

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

</details>

## Example data

Demo FASTAs and SAE signatures are available in [example_data/](example_data/) and on
[jxliu2/idiom-data](https://huggingface.co/datasets/jxliu2/idiom-data). Notebooks download their
inputs; Bash scripts use local paths.

| Directory | Contents |
|---|---|
| `protgps/` | IDRs from six subcellular compartments |
| `effector/` | Activation and repression domain IDRs |
| `disprot/` | Held-out proteins with annotated IDR spans and flanking context |
| `sae_features/` | Released SAE signatures as a format reference |

```bash
hf download jxliu2/idiom-data --repo-type dataset --include "example_data/*"
```

The ProtGPS and effector records mark the whole sequence as the IDR. DisProt records include
flanks; preserve their annotated spans for prompted generation.
See [sequence conventions](../README.md#sequence-conventions).

Sources: [ProtGPS](https://github.com/pgmikhael/protgps) (Kilgore et al.),
DelRosso et al., *Nature* 2023 (effector domains), and [DisProt](https://disprot.org/) (CC BY 4.0).
These are demonstration subsets; cite the original works and follow their licenses when reusing them.
