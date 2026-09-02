# Examples

Short, runnable scripts covering everything IDiom does — generation, embeddings, SAE
interpretability and steering, feature enrichment and logos, and RL/SFT post-training. Each loads a
model by HF repo id or local path, so they work against released weights or your own checkpoints.
Small input sets live in `example_data/` (see below), so the analysis and training examples run with
no arguments.

```bash
# generation, embeddings, interpretability
uv run python examples/01_generate.py        --model jxliu2/idiom-300M
uv run python examples/02_embeddings.py      --model jxliu2/idiom-300M
uv run python examples/03_sae_features.py    --sae   jxliu2/idiomsae-300M-L18-k32   # + steering

# custom rewards, feature enrichment + logos
uv run python examples/04_custom_reward.py                                          # no GPU/weights
uv run python examples/05_feature_enrichment.py --positive example_data/protgps/nucleolus.fasta
uv run python examples/06_feature_logos.py      --positive example_data/protgps/nucleolus.fasta

# post-training (run a real, tiny loop end to end)
uv run python examples/07_sft.py            --init-from jxliu2/idiom-20M --steps 30
uv run python examples/08_grpo.py           --init-from jxliu2/idiom-20M --steps 5
uv run python examples/09_sparrow_reward.py --init-from jxliu2/idiom-20M --steps 5   # external reward
```

A GPU is recommended for anything that runs the model (1–3, 5–9); pass `--device cpu` to force CPU.
`04` needs neither GPU nor weights.

## What each shows

| Script | Covers |
|--------|--------|
| `01_generate.py` | unprompted (de novo) and prompted (context-conditioned) IDR generation |
| `02_embeddings.py` | residual-stream embeddings, pooled and per-residue |
| `03_sae_features.py` | SAE feature activations **and causal steering** along a feature |
| `04_custom_reward.py` | defining a custom `f(idr)->float` reward and the commands that use it |
| `05_feature_enrichment.py` | which SAE features are enriched in a set → signature + **volcano** plot |
| `06_feature_logos.py` | **information-content logos** of what those enriched features detect |
| `07_sft.py` | **SFT**: warm-start a released model and fine-tune it on one set (completion-only) |
| `08_grpo.py` | **RL (GRPO)**: optimize a released model toward a weighted-sum reward (in-process) |
| `09_sparrow_reward.py` | **RL with an external reward model**: a sparrow biophysics scorer in its own env |

## The RL-SAE pipeline, end to end

`05 → 06 → 08` is the RL-SAE story on your own sequences: **05** finds the SAE features enriched in a
set (against the held-out validation background, downloaded from the Hub) and writes a signature;
**06** shows the residue grammar those features encode; and **08** (or `idiom_grpo
reward.rl_sae.enabled=true reward.rl_sae.signature=<name>`) post-trains a model to reproduce that
feature code. `07` (SFT) is the supervised counterpart — specialize a model on the set directly, and
`09` swaps in an external reward model (sparrow) instead of the SAE.

`07`–`09` warm-start from any base `model/io.load_model` accepts: a HF repo id (as above), a local
released directory, or a Lightning `.ckpt` (e.g. the checkpoint `07 --out` saves). They run a handful
of steps on a tiny model so they finish quickly; scale up with `--steps`, a larger `--init-from`, and
a GPU, or use the `idiom_train` / `idiom_grpo` CLIs for full runs. Training on a FASTA auto-builds a
memory-mapped `<fasta>.idiomstore/` sidecar next to it on first run (git-ignored; delete it to
rebuild).

## Running on a Slurm cluster

For real training runs, [`slurm/`](slurm/) has `sbatch` templates for each entrypoint — multi-GPU
DDP pretraining plus single-GPU SFT, GRPO, and SAE training — with the Lightning + Slurm launch
pattern set up correctly. Submit from the repo root, e.g. `sbatch examples/slurm/pretrain.sbatch`;
see [`slurm/README.md`](slurm/README.md) for the `#SBATCH` fields to edit and the multi-node/DDP rule.

## `example_data/`

Small IDR sets used by the enrichment, logo, SFT, and RL examples. Each FASTA is a **subset** (≤150
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
