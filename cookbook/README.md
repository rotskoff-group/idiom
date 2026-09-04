# Cookbook

Runnable material, indexed by what you want to do. Inputs live in [`example_data/`](example_data/),
so nothing needs arguments to start.

This directory is **not** shipped in the wheel — clone the repository to get it. It runs against
either install: the clone's own `uv sync` venv, or an environment you ran
`pip install git+https://github.com/rotskoff-group/idiom.git` into.

| I want to... | run | needs |
|---|---|---|
| **generate IDRs** de novo or between flanks, and pull out embeddings | [`scripts/generate_and_embed.py`](scripts/generate_and_embed.py) | GPU |
| **see what an SAE feature is**, and steer generation along it | [`scripts/sae_features.py`](scripts/sae_features.py) | GPU |
| **find what my sequences share** — enriched features and their residue grammar | [`scripts/feature_enrichment.py`](scripts/feature_enrichment.py) | GPU |
| **write my own reward** | [`rewards/`](rewards/) | CPU |
| **design toward an SAE feature code** (the RL-SAE result) | [`slurm/grpo/sae_features.bash`](slurm/grpo/sae_features.bash) | 1 GPU, hours |
| **design toward my own reward** (in this interpreter) | [`slurm/grpo/my_reward.bash`](slurm/grpo/my_reward.bash) | 1 GPU, hours |
| **design toward a published reward model** (its own environment) | [`slurm/grpo/`](slurm/grpo/) — one script each | 1 GPU, hours |
| **specialize a model on my own set** | [`slurm/sft.bash`](slurm/sft.bash) | 1 GPU |
| **train an SAE on another layer** | [`slurm/sae.bash`](slurm/sae.bash) | 1 GPU |
| **pretrain from scratch** | [`slurm/pretrain.bash`](slurm/pretrain.bash) | 8 GPUs, days |

## Walkthroughs

Plain top-to-bottom scripts — no arguments, no `main()`. Edit the block of constants at the top to
point one at your own model, SAE, or sequences:

```bash
uv run cookbook/scripts/generate_and_embed.py
```

They read in that order, each ending by pointing at the next. `DEVICE = "auto"` falls back to CPU,
and figures are written as PNGs rather than shown, so they behave the same over SSH.

`feature_enrichment.py → slurm/grpo.bash` is the RL-SAE pipeline on your own sequences: the
walkthrough writes a signature of the features enriched in a set, and the training script
post-trains a model to reproduce that feature code.

```bash
IDIOM_SAEREWARD_FEATURES=enr/signature.json IDIOM_SAEREWARD_CASE=top30 \
  sbatch cookbook/slurm/grpo/sae_features.bash    # set SIGNATURE=<name> at the top
```

## `rewards/`

What GRPO optimizes, and how to point it at your own — the full guide is
[`rewards/README.md`](rewards/README.md).

| file | what it is |
|---|---|
| [`rewards/my_rewards.py`](rewards/my_rewards.py) | template: rewards that run in this process — copy it and edit it |
| [`rewards/scorers/`](rewards/scorers/) | seven reward models in their own environments, `my_scorer.py` being the minimal template |

The config carries only the guardrails (`entropy`, `length`); a run says what it optimizes at
launch, by name from the shipped menu:

```bash
idiom_grpo init_from=jxliu2/idiom-300M reward.add=[rg]                    # one
idiom_grpo init_from=jxliu2/idiom-300M reward.add=[sae,rg]      # several
idiom_grpo init_from=jxliu2/idiom-300M reward.add=[rg] reward.presets.rg.shaping.target=30
```

Yours needs no menu entry — pass the whole term to `reward.add`, in one of three forms:

```yaml
- {reward: fraction_charged, module: cookbook/rewards/my_rewards.py, weight: 1.0}  # registered by name
- {reward: "mypackage.scoring:score_idr", weight: 1.0}                             # any importable callable
- {cmd: "uv run --script cookbook/rewards/scorers/sparrow.py ...", label: rg, weight: 0.5}  # own environment
```

Each has a ready-to-submit script in [`slurm/grpo/`](slurm/grpo/) that passes it as `reward.add`.

Reach for the third only when the scorer's dependencies cannot coexist with IDiom's. Size any new
term against the guardrails first — one that starts ten times larger has made them invisible, and
the run will converge on something that is no longer an IDR.

## Training scripts

Each spells out every config value as a Hydra override, so a run is reproducible from the script
alone. They are `.bash` because they work either way — with a scheduler or without:

```bash
mkdir -p slurm_out && sbatch cookbook/slurm/pretrain.bash   # #SBATCH --output writes here
bash cookbook/slurm/sft.bash                                # no scheduler; #SBATCH lines are comments
```

Submit from the repo root (under `sbatch` the repo is `$SLURM_SUBMIT_DIR`).

**Edit before submitting:** the `#SBATCH` header for your cluster; `OUT` (keep it on scratch — the
scripts pin `out_dir`/`hydra.run.dir` there so nothing lands in the repo); the data paths, which
default to `example_data`; `UV_CACHE_DIR` if you use an external reward. W&B is offline by default —
`wandb login` and submit with `WANDB_MODE=online` for live logging.

Each script activates the clone's `.venv` if `uv sync` created one, and otherwise runs in whatever
environment is already active — so a `pip install git+...` into your own env needs no change.

**Multi-GPU: no `srun`.** One Slurm task owns the node and Lightning launches one process per GPU
from `trainer.devices`, so `--gpus-per-node == trainer.devices` and `--cpus-per-task` covers all
DataLoader workers. `data.batch_size` is per GPU; global batch is
`batch_size × devices × accumulate_grad_batches`. For multi-node, launch with `srun`,
`--ntasks-per-node = gpus-per-node`, and `+trainer.num_nodes=$SLURM_NNODES`.

**Resuming.** `pretrain.bash` and `sft.bash` pick up `$OUT/checkpoints/last.ckpt` automatically, so
re-submitting after a timeout continues (optimizer, step, schedule, RNG). GRPO and SAE keep only a
final checkpoint — pass `resume_from=<ckpt>` by hand, or set `trainer.checkpoint_every=<N>`.

Every script warm-starts from anything `model/io.load_model` accepts: a HF repo id, a released
directory, or a `.ckpt`. Training on a FASTA builds a memory-mapped `<fasta>.idiomstore/` sidecar
beside it on first run (git-ignored; delete to rebuild).

## `example_data/`

Demo-sized subsets (≤150 records) of curated IDR sets, with `_IDR_x-y` headers, so each drops
straight into `sae.encode`, `idiom_train`, and the enrichment pipeline. Not the full datasets used
in the paper.

```
protgps/    6 subcellular-condensate IDR sets (stress_granule, p-body, nuclear_speckle,
            nucleolus, chromosome, nuclear_pore_complex)
effector/   activation (ad) and repression (rd) domain IDRs
disprot/    DisProt proteins with annotated IDR spans, held out of pretraining -- these
            carry real flanks, so they are the reference set for prompted generation
```

`protgps/` and `effector/` records are fully disordered (the whole record is the span). `effector/`
headers carry extra free-text fields after the accession, which IDiom ignores.

**Provenance.** `protgps/` — [ProtGPS](https://github.com/pgmikhael/protgps) (Kilgore et al.).
`effector/` — DelRosso et al., *Nature* 2023. `disprot/` — [DisProt](https://disprot.org/), CC BY
4.0. Redistributed for demonstration only; cite the original works and check their licenses for any
other use.
