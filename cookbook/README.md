# Cookbook

Runnable material, indexed by what you want to do. [`notebooks/`](notebooks/) holds the walkthroughs,
[`scripts/`](scripts/) the training scripts, [`rewards/`](rewards/) what GRPO optimizes.

The notebooks install IDiom themselves and pull inputs from
[`jxliu2/idiom-data`](https://huggingface.co/datasets/jxliu2/idiom-data), so **nothing there needs a
clone** — click a badge and run. The scripts use your installed IDiom package and a clone for
scorers, custom rewards, and example data.

| I want to... | run | needs |
|---|---|---|
| **generate IDRs** de novo or between flanks, and pull out embeddings | [`generate_and_embed.ipynb`](notebooks/generate_and_embed.ipynb) | GPU |
| **see what an SAE feature is**, and steer generation along it | [`sae_features.ipynb`](notebooks/sae_features.ipynb) | GPU |
| **find what my sequences share** — enriched features and their residue grammar | [`feature_enrichment.ipynb`](notebooks/feature_enrichment.ipynb) | GPU |
| **design toward an SAE feature code** (the RL-SAE result) | [`scripts/grpo/sae_features.bash`](scripts/grpo/sae_features.bash) | 1 GPU, hours |
| **design toward my own reward** (in this interpreter) | [`scripts/grpo/custom_reward.bash`](scripts/grpo/custom_reward.bash) | 1 GPU, hours |
| **design toward a published reward model** (its own environment) | [`scripts/grpo/`](scripts/grpo/) — one script each | 1 GPU, hours |
| **specialize a model on my own set** | [`scripts/sft.bash`](scripts/sft.bash) | 1 GPU |
| **train an SAE on another layer** | [`scripts/train_sae.bash`](scripts/train_sae.bash) | 1 GPU |
| **pretrain from scratch** | [`scripts/pretrain.bash`](scripts/pretrain.bash) | 8 GPUs, days |

## Notebooks

Three walkthroughs, read in this order, each ending by pointing at the next. Colab links are in
[`notebooks/README.md`](notebooks/README.md).

1. **`generate_and_embed`** — unprompted and prompted generation, redesigning a real protein's IDR,
   and residual-stream embeddings.
2. **`sae_features`** — which SAE features fire on a sequence, and steering generation along one.
3. **`feature_enrichment`** — which features are enriched in *your* set, their residue grammar, and
   the signature that turns them into an RL target.

Each opens with a setup cell that `pip install`s IDiom if missing and defines `example_data()`, which
pulls a demo FASTA from the Hub — so they run identically in Colab, local Jupyter, or on a cluster.
Edit the parameter cell to point one at your own model, SAE, or sequences.

`feature_enrichment.ipynb → scripts/grpo/sae_features.bash` is the RL-SAE pipeline on your own
sequences: the notebook writes a signature of the features enriched in a set, the training script
post-trains a model to reproduce that feature code.

```bash
# edit REPO, OUT, FEATURES, and SIGNATURE in the script first
bash cookbook/scripts/grpo/sae_features.bash
```

The script refuses to start if the signature is not there, so the two stay in step.

## Scripts

Plain bash, no scheduler. Each spells out every config value as a Hydra override. First activate
your chosen Python environment, install IDiom, and clone the repository to access the cookbook:

```bash
python -m pip install git+https://github.com/rotskoff-group/idiom.git
git clone https://github.com/rotskoff-group/idiom.git
cd idiom
```

Use the same tag or commit for the installation and clone to keep the examples matched to the
installed API. External-scorer examples also need `uv` (`python -m pip install uv`); their
`uv run --script` commands manage each scorer's separate dependencies. No `uv sync` is required.

Edit the path placeholders and run settings before launching:

```bash
bash cookbook/scripts/sft.bash
```

Set `REPO` to the absolute path of your repository checkout and `OUT` to the desired run output
directory in each script. The scripts `cd` to `REPO` and use `python` and the `idiom_*` commands
from your active environment; they do not activate a checkout's `.venv/`. For SAE
feature GRPO, also set `FEATURES` to the signature JSON written by the notebook. Pretraining and
SAE training need corpus paths (`TRAIN_FASTA` / `VAL_FASTA` or `FASTA`). Other `# EDIT` markers
identify run choices such as target, width, weight, property, and compartment.

W&B is offline by default; `wandb login` and set `WANDB_MODE=online` for live logging.

**To submit to a scheduler**, wrap rather than edit — the scripts take no arguments and read no
scheduler variables. Make sure the job uses the environment where you installed IDiom:

```bash
sbatch --gpus-per-node=1 --cpus-per-task=8 --time=12:00:00 \
    --wrap "bash $PWD/cookbook/scripts/grpo/sparrow.bash"
```

**Multi-GPU.** Lightning launches one process per GPU from `trainer.devices`, so do not put a
launcher in front of these. `data.batch_size` is per GPU; global batch is
`batch_size × devices × accumulate_grad_batches`. For multi-node, launch with `srun`, one task per
GPU, and `+trainer.num_nodes=<N>`.

**Resuming.** `pretrain.bash` and `sft.bash` pick up `$OUT/checkpoints/last.ckpt` automatically. GRPO
and SAE keep only a final checkpoint — pass `resume_from=<ckpt>`, or set `trainer.checkpoint_every=<N>`.

Every script warm-starts from anything `model/io.load_model` accepts: a HF repo id, a released
directory, or a `.ckpt`. Training on a FASTA builds a memory-mapped `<fasta>.idiomstore/` sidecar
beside it on first run (git-ignored; delete to rebuild).

## Rewards

[`rewards/`](rewards/) holds what GRPO optimizes — rewards and shaping that run in this process
([`custom_rewards.py`](rewards/custom_rewards.py)) and reward models that run in their own
environment ([`scorers/`](rewards/scorers/)). Nothing there ships in the wheel; these are files you
read, copy, and edit, named by a run on the command line.

The shipped objective is **empty** — `reward.terms: []`, no defaults and no presets — so a run names
every term it optimizes. Each script in [`scripts/grpo/`](scripts/grpo/) writes one whole objective
out and passes it to `reward.terms`.

**See [`rewards/README.md`](rewards/README.md)** for the scorers, the reward and shaping forms,
picking targets and weights, and writing your own.

## Example data

The notebooks' inputs live on the Hub under `example_data/` in
[`jxliu2/idiom-data`](https://huggingface.co/datasets/jxliu2/idiom-data) — demo-sized subsets (≤150
records) of curated IDR sets, with `_IDR_x-y` headers, so each drops straight into `sae.encode`,
`idiom_train_autoreg`, and the enrichment pipeline. Not the full datasets used in the paper.

```
protgps/       6 subcellular-condensate IDR sets (stress_granule, p-body, nuclear_speckle,
               nucleolus, chromosome, nuclear_pore_complex)
effector/      activation (ad) and repression (rd) domain IDRs
disprot/       DisProt proteins with annotated IDR spans, held out of pretraining — these
               carry real flanks, so they are the reference set for prompted generation
sae_features/  sae_signatures.json, the released SAE's signatures, as a format reference
```

```bash
hf download jxliu2/idiom-data --repo-type dataset --include "example_data/*"
```

`protgps/` and `effector/` records are fully disordered (the whole record is the span). `effector/`
headers carry extra free-text fields after the accession, which IDiom ignores. A copy is mirrored in
[`example_data/`](example_data/) for the bash scripts, which name their inputs by path.

**Provenance.** `protgps/` — [ProtGPS](https://github.com/pgmikhael/protgps) (Kilgore et al.).
`effector/` — DelRosso et al., *Nature* 2023. `disprot/` — [DisProt](https://disprot.org/), CC BY
4.0. Redistributed for demonstration only; cite the original works and check their licenses for any
other use.
