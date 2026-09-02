# Slurm job templates

`sbatch` scripts for the training entrypoints on a Slurm cluster. They're **templates** — edit the
`#SBATCH` header for your site and fill in the data paths — but the launch pattern (Lightning + Slurm
DDP) is correct as written.

| Script | Job | GPUs |
|--------|-----|------|
| `pretrain.sbatch` | pretrain from scratch (`idiom_train`) | 8 (DDP) |
| `sft.sbatch` | fine-tune a released model (`idiom_train --config-name sft`) | 1 |
| `grpo.sbatch` | RL post-training (`idiom_grpo`) | 1 |
| `sae.sbatch` | train a top-k SAE (`idiom_sae`) | 1 |

## Submit

Run from the **repo root** (the scripts `cd "$SLURM_SUBMIT_DIR"` and `source .venv/bin/activate`,
which `uv sync` created):

```bash
sbatch examples/slurm/pretrain.sbatch
squeue --me                     # watch the queue
tail -f slurm-idiom-pretrain-*.out
```

## Before you submit, edit

- **`#SBATCH` header** — `--partition`, `--account` (delete if your site has none), `--time`, `--mem`.
- **Data paths** in the `srun` line (`data.train_fasta=…`, etc.).
- **`IDIOM_OUT`** — where checkpoints and Hydra run dirs go. Keep it on scratch, **not** in the repo
  (the configs default `IDIOM_OUT` to `./runs`, which would write into your checkout).
- **W&B** — the scripts set `WANDB_MODE=offline`; run `wandb login` (or set `WANDB_API_KEY`) and
  remove that line for live logging.
- **`UV_CACHE_DIR`** (GRPO with an external reward only) — point it at scratch; on-demand reward envs
  (e.g. sparrow) are several GB.

## The DDP rule (multi-GPU)

Lightning auto-detects Slurm from the job environment, and `srun` launches **one process per task**,
each bound to one GPU. So the geometry must line up:

```
--ntasks-per-node  ==  --gpus-per-node  ==  trainer.devices     (per node)
```

`pretrain.sbatch` uses 8 of each (matching `configs/pretrain.yaml`'s `devices: 8`); the single-GPU
jobs use 1. Always launch the training command with `srun` — that's what lets Lightning place one
rank per GPU. Per-GPU `batch_size` × `devices` × `accumulate_grad_batches` is the global batch.

**Multi-node:** raise `--nodes`, keep `--ntasks-per-node` at the per-node GPU count, and append
`+trainer.num_nodes=$SLURM_NNODES` to the `srun` command. Nothing else changes.

## Scaling the single-GPU jobs

`sft` / `grpo` / `sae` are single-GPU here for simplicity. To run any of them on N GPUs, set
`--ntasks-per-node=N --gpus-per-node=N` and add `trainer.devices=N` to the command — the same DDP
rule applies. (For big pretraining runs prefer `idiom_build_store` first so every rank memory-maps
one shared copy of the corpus instead of re-parsing the FASTA.)
