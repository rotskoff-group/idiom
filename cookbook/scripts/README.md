# Scripts

Editable Bash examples for generation, feature extraction, and training. Start with the
[installation instructions](../../README.md#installation); each script uses your active environment.

# Choose a script

| Script | Task |
|---|---|
| [generate.bash](generate.bash) | Generate de novo IDRs to FASTA |
| [generate_prompted.bash](generate_prompted.bash) | Replace IDRs in flanking protein context |
| [feature_dataset.bash](feature_dataset.bash) | Build a per-residue SAE feature dataset |
| [sft.bash](sft.bash) | Fine-tune on a sequence set |
| [train_sae.bash](train_sae.bash) | Train and export an SAE |
| [pretrain.bash](pretrain.bash) | Pretrain IDiom from scratch |
| [grpo/](grpo/) | Post-train with feature signatures, custom rewards, or external scorers |

See the [reward guide](../rewards/README.md#examples) for the GRPO objectives and scorer setup.
Demo inputs and their provenance are described in [example_data/](../example_data/).
The [SAE notebook](../notebooks/sae_features.ipynb) shows how to inspect a feature dataset.

# Running scripts

Scripts use IDiom from your active Python environment and the clone for cookbook files.
Either [installation workflow](../../README.md#installation) works. When installing and cloning
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
