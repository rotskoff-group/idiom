# Examples

Two kinds, split by what they are for. **Notebooks** in [`notebooks/`](notebooks/) are for looking
at things — generating, reading and steering SAE features, finding which features matter for your
sequences, and seeing what a reward term does. **Training scripts** in [`slurm/`](slurm/) are for
running things — pretraining, SFT, SAE training, and RL — each a single script that spells out
every config value.

Small input sets live in [`example_data/`](example_data/), so the analysis examples run with no
arguments. Run everything from the repo root.

## Notebooks

```bash
uv sync --group notebooks
uv run jupyter lab examples/notebooks
```

| Notebook | Covers | Needs |
|----------|--------|-------|
| [`generate_and_embed.ipynb`](notebooks/generate_and_embed.ipynb) | unprompted (de novo) and prompted (context-conditioned) generation; residual-stream embeddings, pooled and per-residue | GPU, weights |
| [`sae_features.ipynb`](notebooks/sae_features.ipynb) | which SAE features fire on a sequence, and **causal steering** along one of them | GPU, weights |
| [`feature_enrichment.ipynb`](notebooks/feature_enrichment.ipynb) | which features are enriched in your own set → signature + **volcano** plot + **sequence logos** of what they detect | GPU, weights |
| [`rewards.ipynb`](notebooks/rewards.ipynb) | rewards and shaping: what a reward term is made of, and how to write your own | nothing |

They read in that order, and each ends by pointing at the next. `device="auto"` in every parameter
cell falls back to CPU, and `rewards.ipynb` needs neither a GPU nor weights, so it runs anywhere.

## The RL-SAE pipeline, end to end

`feature_enrichment.ipynb → slurm/grpo.bash` is the RL-SAE story on your own sequences: the notebook
finds the SAE features enriched in a set (against the held-out validation background, downloaded
from the Hub), shows the residue grammar those features encode, and writes a signature; the training
script then post-trains a model to reproduce that feature code. Point `IDIOM_SAEREWARD_FEATURES` at your
signature and switch on the RL-SAE term:

```bash
IDIOM_SAEREWARD_FEATURES=enr/signature.json IDIOM_SAEREWARD_CASE=top30 \
  idiom_grpo init_from=jxliu2/idiom-300M \
    reward.terms.2.enabled=true reward.terms.2.reward=sae_only_<name>
```

`sft.bash` is the supervised counterpart — specialize a model on the set directly — and
`grpo_external.bash` swaps in a reward model that runs in its own environment (sparrow) instead of
the SAE.

Every training script warm-starts from anything `model/io.load_model` accepts: a HF repo id, a local
released directory, or a Lightning `.ckpt`. Training on a FASTA auto-builds a memory-mapped
`<fasta>.idiomstore/` sidecar next to it on first run (git-ignored; delete it to rebuild).

## Training scripts

Each spells out **every config value as a Hydra override** (so the run is reproducible from the
script alone), self-logs its own contents into the job log, pins `out_dir` / `hydra.run.dir`, and
auto-resumes from a rolling `last.ckpt`.

| Script | Job | GPUs |
|--------|-----|------|
| `pretrain.bash` | pretrain 24L from scratch (`idiom_train`) | 8 |
| `sft.bash` | fine-tune a released model (`idiom_train --config-name sft`) | 1 |
| `grpo.bash` | RL post-training toward an SAE signature (`idiom_grpo`) | 1 |
| `grpo_external.bash` | RL toward an external reward model in its own env (sparrow) | 1 |
| `sae.bash` | train a top-k SAE (`idiom_sae`) | 1 |

They are `.bash` because they run **either way** — submit them to Slurm, or run them directly on a
machine you already have:

```bash
mkdir -p slurm_out                 # #SBATCH --output writes here; Slurm won't create it
sbatch examples/slurm/pretrain.bash
squeue --me
tail -f slurm_out/slurm-*.out
```

```bash
bash examples/slurm/sft.bash       # same script, no scheduler; the #SBATCH lines are just comments
```

Under `sbatch` the repo is `$SLURM_SUBMIT_DIR`, so submit from the repo root; under `bash` it is
resolved from the script's own location, so you can run it from anywhere. Either way the script
activates `.venv`, which `uv sync` created.

**Before you submit, edit:**

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

**Multi-GPU: no `srun`.** These follow the convention of **not** wrapping the command in `srun`: one
Slurm task owns the node, and Lightning launches one DDP process per GPU itself from
`trainer.devices`. So on one node:

```
--gpus-per-node  ==  trainer.devices           # 8 for pretrain, 1 for the rest
--cpus-per-task   covers all DataLoader workers  # e.g. 32 = 4 workers/GPU x 8 GPUs
```

`data.batch_size` is **per GPU**; global batch = `batch_size × devices × accumulate_grad_batches`.
For **multi-node** DDP, Lightning's in-process launcher is single-node — launch with `srun` and
`--ntasks-per-node = gpus-per-node` instead, and add `+trainer.num_nodes=$SLURM_NNODES`.

**Auto-resume.** Each training script checks `$OUT/checkpoints/last.ckpt` and, if present, adds
`resume_from=…` to continue (optimizer, global step, LR schedule, and RNG are all restored). A fresh
run starts from scratch; after a Slurm timeout, just re-submit the same script and it picks up where
it left off. (The GRPO scripts set `trainer.checkpoint_every=500` so a `last.ckpt` exists to resume
from; `sae.bash` has no fixed `last.ckpt` — pass `resume_from=<ckpt>` by hand.)

## `example_data/`

Small IDR sets used by the enrichment notebook and the SFT and RL scripts. Each FASTA is a **subset** (≤150
records) of a curated positive set, with IDiom `_IDR_x-y` headers, so it drops straight into
`sae.encode`, `build_feature_dataset`, `idiom_train`, and the enrichment pipeline. These are
demo-sized excerpts, not the full datasets used in the paper.

```
example_data/
  protgps/     6 subcellular-condensate IDR sets, from ProtGPS
    stress_granule.fasta  p-body.fasta  nuclear_speckle.fasta
    nucleolus.fasta       chromosome.fasta  nuclear_pore_complex.fasta
  effector/    2 transcriptional-effector IDR sets, from DelRosso et al. 2023
    ad.fasta   activation domains
    rd.fasta   repression domains
```

Every sequence is a fully disordered IDR (the whole record is the span, no flanks). The `effector/`
headers also carry the source gene and measured strength as free-text fields after the accession;
IDiom reads only the leading `{accession}_IDR_{x}-{y}` token and ignores the rest.

**Provenance and licensing.** `protgps/` — condensate IDR sets from **ProtGPS** (Kilgore et al.;
<https://github.com/pgmikhael/protgps>). `effector/` — activation/repression domain IDRs from
**DelRosso et al., *Nature* 2023**, "Large-scale mapping and mutagenesis of human transcriptional
effector domains." These excerpts are redistributed for demonstration only; cite the original works
if you use them, and consult their licenses for any other use.
