# Slurm job templates

`sbatch` scripts for the training entrypoints. Each script spells out **every config value as a Hydra
override** (so the run is fully reproducible from the script alone), self-logs its own contents into
the Slurm log, pins `out_dir` / `hydra.run.dir`, and auto-resumes from a rolling `last.ckpt`.

| Script | Job | GPUs |
|--------|-----|------|
| `pretrain.sbatch` | pretrain 24L from scratch (`idiom_train`) | 8 |
| `sft.sbatch` | fine-tune a released model (`idiom_train --config-name sft`) | 1 |
| `grpo.sbatch` | RL post-training toward an SAE signature (`idiom_grpo`) | 1 |
| `sae.sbatch` | train a top-k SAE (`idiom_sae`) | 1 |

## Submit

Run from the **repo root** (the scripts use `$SLURM_SUBMIT_DIR` as the repo, `cd` there, and
`source .venv/bin/activate`, which `uv sync` created). Create the log dir once:

```bash
mkdir -p slurm_out                 # #SBATCH --output writes here; Slurm won't create it
sbatch examples/slurm/pretrain.sbatch
squeue --me
tail -f slurm_out/slurm-*.out
```

## Before you submit, edit

- **`#SBATCH` header** — `--partition` (and `--account` if your site needs one), `--time`, and the
  memory/CPU directives for your cluster.
- **`OUT`** at the top of the script — where checkpoints and the Hydra run dir go. Keep it on scratch;
  the scripts pin `out_dir`/`hydra.run.dir` to it so nothing lands in the repo.
- **Data paths** — `data.train_fasta` / `data.val_fasta` (pretrain), `data.fasta` (SAE). The SFT/GRPO
  scripts default to the bundled `example_data`; point them at your own set for a real run.
- **W&B** — scripts export `WANDB_MODE=offline`; `wandb login` (or set `WANDB_API_KEY`) and submit with
  `WANDB_MODE=online sbatch …` for live logging.
- **`UV_CACHE_DIR`** (GRPO with an external reward only) — point it at scratch; on-demand reward envs
  (e.g. sparrow) are several GB.

## Multi-GPU: no `srun`

These follow the convention of **not** wrapping the command in `srun`: one Slurm task owns the node,
and Lightning launches one DDP process per GPU itself from `trainer.devices`. So on one node:

```
--gpus-per-node  ==  trainer.devices           # 8 for pretrain, 1 for the rest
--cpus-per-task   covers all DataLoader workers  # e.g. 32 = 4 workers/GPU x 8 GPUs
```

`data.batch_size` is **per GPU**; global batch = `batch_size × devices × accumulate_grad_batches`.
For **multi-node** DDP, Lightning's in-process launcher is single-node — launch with `srun` and
`--ntasks-per-node = gpus-per-node` instead, and add `+trainer.num_nodes=$SLURM_NNODES`.

## Auto-resume

Each training script checks `$OUT/checkpoints/last.ckpt` and, if present, adds `resume_from=…` to
continue (optimizer, global step, LR schedule, and RNG are all restored). A fresh run starts from
scratch; after a Slurm timeout, just re-submit the same script and it picks up where it left off.
(`grpo.sbatch` sets `trainer.checkpoint_every=500` so a `last.ckpt` exists to resume from; `sae.sbatch`
has no fixed `last.ckpt` — pass `resume_from=<ckpt>` by hand.)
