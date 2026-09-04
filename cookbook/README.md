# Cookbook

Runnable material, indexed by what you want to do. [`scripts/python/`](scripts/python/) holds the
walkthroughs — top-to-bottom Python scripts you can run right now; [`scripts/bash/`](scripts/bash/)
holds the training scripts — GRPO, SFT, SAE, and pretraining, as job scripts;
[`rewards/`](rewards/) holds what GRPO optimizes. Inputs live in
[`example_data/`](example_data/), so nothing needs arguments to start.

This directory is **not** shipped in the wheel — clone the repository to get it. It runs against
either install: the clone's own `uv sync` venv, or an environment you ran
`pip install git+https://github.com/rotskoff-group/idiom.git` into.

| I want to... | run | needs |
|---|---|---|
| **generate IDRs** de novo or between flanks, and pull out embeddings | [`python/generate_and_embed.py`](scripts/python/generate_and_embed.py) | GPU |
| **see what an SAE feature is**, and steer generation along it | [`python/sae_features.py`](scripts/python/sae_features.py) | GPU |
| **find what my sequences share** — enriched features and their residue grammar | [`python/feature_enrichment.py`](scripts/python/feature_enrichment.py) | GPU |
| **design toward an SAE feature code** (the RL-SAE result) | [`bash/grpo/sae_features.bash`](scripts/bash/grpo/sae_features.bash) | 1 GPU, hours |
| **design toward my own reward** (in this interpreter) | [`bash/grpo/my_reward.bash`](scripts/bash/grpo/my_reward.bash) | 1 GPU, hours |
| **design toward a published reward model** (its own environment) | [`bash/grpo/`](scripts/bash/grpo/) — one script each | 1 GPU, hours |
| **specialize a model on my own set** | [`bash/sft.bash`](scripts/bash/sft.bash) | 1 GPU |
| **train an SAE on another layer** | [`bash/train_sae.bash`](scripts/bash/train_sae.bash) | 1 GPU |
| **pretrain from scratch** | [`bash/pretrain.bash`](scripts/bash/pretrain.bash) | 8 GPUs, days |

## `scripts/python/` — walkthroughs

Plain top-to-bottom scripts — no arguments, no `main()`. Edit the block of constants at the top to
point one at your own model, SAE, or sequences:

```bash
uv run cookbook/scripts/python/generate_and_embed.py
```

They read in that order, each ending by pointing at the next. `DEVICE = "auto"` falls back to CPU,
and figures are written as PNGs rather than shown, so they behave the same over SSH.

`feature_enrichment.py → scripts/bash/grpo/sae_features.bash` is the RL-SAE pipeline on your own sequences:
the walkthrough writes a signature of the features enriched in a set, and the training script
post-trains a model to reproduce that feature code.

```bash
uv run cookbook/scripts/python/feature_enrichment.py   # -> example_data/sae_features/signature.json
sbatch cookbook/scripts/bash/grpo/sae_features.bash    # set SIGNATURE=<name> at the top
```

The training script reads the signature the walkthrough wrote, and refuses to start if it is not
there, so the two stay in step without either one hard-coding a path you have to keep in sync.

## `scripts/bash/` — training scripts

Examples, not turnkey jobs: each spells out every config value as a Hydra override, so a run is
reproducible from the script alone, and every path in them is a placeholder. They work either way —
with a scheduler or without:

```bash
sbatch cookbook/scripts/bash/pretrain.bash   # #SBATCH header applies
bash cookbook/scripts/bash/sft.bash          # no scheduler; #SBATCH lines are comments
```

**Edit before running:** the `#SBATCH` header for your cluster; `REPO`, `OUT`, the venv passed to
`source`, and the data paths — all `/path/to/...` placeholders; `UV_CACHE_DIR` if you use an
external reward. W&B is offline by default — `wandb login` and set `WANDB_MODE=online` for live
logging.

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

## `rewards/` — what GRPO optimizes

**Nothing here ships in the wheel** — these are files you read, copy, and edit, named by a run on
the command line.

```
my_rewards.py     rewards that run in this process — copy this and edit it
scorers/          reward models that run in their OWN environment, one program each
```

The library defines exactly two rewards, `entropy` and `length`, because every run should carry
them. It takes no view on what you should design for, so the shipped config's objective is those two
and nothing else, and what a run optimizes is named at launch. Each script in
[`scripts/bash/grpo/`](scripts/bash/grpo/) spells out one whole objective as a `TERM` at the top, passed to
`reward.add`, so the run is reproducible from the script alone and the config is never edited:

| script | optimizes | reward |
|---|---|---|
| [`grpo/sae_features.bash`](scripts/bash/grpo/sae_features.bash) | a target's SAE feature code (the RL-SAE result) | library |
| [`grpo/my_reward.bash`](scripts/bash/grpo/my_reward.bash) | anything you can compute in-process | [`my_rewards.py`](rewards/my_rewards.py) |
| [`grpo/sparrow.bash`](scripts/bash/grpo/sparrow.bash) | radius of gyration, asphericity, kappa, FCR | [`scorers/sparrow.py`](rewards/scorers/sparrow.py) |
| [`grpo/finches.bash`](scripts/bash/grpo/finches.bash) | epsilon — interaction chemistry | [`scorers/finches.py`](rewards/scorers/finches.py) |
| [`grpo/protgps.bash`](scripts/bash/grpo/protgps.bash) | condensate compartment probability | [`scorers/protgps.py`](rewards/scorers/protgps.py) |
| [`grpo/paddle.bash`](scripts/bash/grpo/paddle.bash) | transcriptional activation strength | [`scorers/paddle.py`](rewards/scorers/paddle.py) |
| [`grpo/pspred.bash`](scripts/bash/grpo/pspred.bash) | phase-separation dG / c_sat | [`scorers/pspred.py`](rewards/scorers/pspred.py) |
| [`grpo/starling.bash`](scripts/bash/grpo/starling.bash) | ensemble Rg or end-to-end distance | [`scorers/starling.py`](rewards/scorers/starling.py) |
| [`grpo/my_scorer.bash`](scripts/bash/grpo/my_scorer.bash) | a scorer you wrote, in its own environment | [`scorers/my_scorer.py`](rewards/scorers/my_scorer.py) |

### How a reward is built

A run's objective is a list of **terms** in `src/idiom/configs/grpo.yaml`. Each pairs a **reward**
(one raw value in its own units) with the **shaping** that says what a good value is, and a
**weight** for how much it matters; `total = Σ weightᵢ · shapingᵢ(rewardᵢ)` drives the advantages.

```yaml
reward:
  terms:
    - {reward: entropy, weight: 1.0, shaping: {type: quadratic, target: 3.65, width: 0.2}}  # bits
    - {reward: length,  weight: 1.0, shaping: {type: quadratic, target: 100,  width: 1.0}}  # residues
```

| shaping | shape | use when |
|---|---|---|
| `quadratic` | 0 at the target, −1 one `width` out, unbounded below | one target, hit hard |
| `gaussian` | 1 at the target decaying to 0 — bounded | several targets must coexist |
| omitted | raw value passes through | already on a sensible scale, e.g. a probability |

`width` is a tolerance in the reward's units — a fraction of a nonzero `target` (`width: 0.2` on
`target: 25` is ±5 Å), absolute when the target is 0. `weight: 0` keeps a term running and logged
without optimizing it; `enabled: false` skips it entirely (nothing imported, no subprocess).

### Choosing the objective at launch

`grpo.yaml` carries the two guardrails and nothing else. What a run designs for is one `TERM` on the
command line, appended with `reward.add`, and several compose in the order you name them:

```bash
idiom_grpo init_from=jxliu2/idiom-300M \
  reward.add='[{reward: sae_only_nucleolus, module: idiom.train.grpo.reward.sae_feature, weight: 1.0}]'

idiom_grpo init_from=jxliu2/idiom-300M \
  reward.add='[{cmd: "uv run --script cookbook/rewards/scorers/sparrow.py --property radius_of_gyration",
                label: rg, weight: 0.5, shaping: {type: quadratic, target: 25, width: 0.2}}]'

idiom_grpo init_from=jxliu2/idiom-300M reward.add="[$SAE_TERM, $RG_TERM]"
```

An entry may also be a **name** from `reward.presets`, which is how you keep a tuned term without
retyping it: put a `presets` block in a config of your own, launch with `--config-dir . --config-name
my_grpo`, then `reward.add=[my_term]` and `reward.presets.my_term.shaping.target=30` to tune it. The
shipped `presets` is empty — there is no menu to pick from.

### Picking a target, a width, and a weight

The failure mode is quiet — the objective improves while the sequences stop being IDRs.

**Keep `entropy` and `length` on.** They cost nothing and contribute about **−1.8** to a starting
sequence. **Size every new term against that**: pick `width` so a typical base generation sits about
one width from the target, and `weight` so the starting contribution is order 1. A term worth −30 at
step 0 has made the guardrails invisible — `finches` aimed at −6.0 with `width: 1.0` started at
−31.5 and swamped everything; widened to match the spread of base generations it starts at −3.8 and
converges with the guardrails intact.

**Aim inside the natural range.** Base generations from `idiom-300M`:

| quantity | scorer | base (mean ± sd) | strong natural example |
|---|---|---|---|
| composition entropy | `entropy` | 3.66 bits | — |
| length | `length` | 88 (median 56) | — |
| radius of gyration | `sparrow` | 26.9 ± 15.7 Å | — |
| fraction charged | `fraction_charged` | 0.243 | — |
| epsilon (self-interaction) | `finches` | +3.6 ± 7.0 | FUS-LC ≈ −8.5 |
| transfer free energy | `pspred --target dG` | −0.12 ± 0.83 kT | LAF1 ≈ −6.1 |
| activation strength | `paddle` | +0.94 ± 1.68 Z | natural ADs ≈ 3–8 |
| predicted disorder | metapredict | 0.665 | — |

**Add a `weight: 0` term** for anything you'd notice losing, so drift shows up in the logs.

A reward whose optimum sits off the IDR distribution *will* be reached. Rewarding aromatic content
at weight 1.0 for 3000 steps hit its target exactly (0.1488 vs 0.15) while collapsing to hydrophobic
`LVIFA` segments — disorder fell from 0.665 to **0.091** — with entropy and length on target
throughout. Nothing in the reward said "still an IDR".

### In-process — a function in this interpreter

Use this whenever the reward can be imported into the environment IDiom is running in, which is most
of the time. A reward is one function, one raw value, no notion of "good".

**Built in.** Name it and it works — no `module`, nothing on disk: `entropy` and `length` (the
guardrails), `net_charge_fraction`, `fraction_charged`, `sumo_motif_count`, `ndsm_motif_count`, in
`idiom/train/grpo/reward/builtin.py`.

**Yours, registered by name.** Copy [`my_rewards.py`](rewards/my_rewards.py) into your project,
decorate with `@register_reward("fraction_aromatic")`, and point a term at the file (a `*.py` path or
a dotted module name):

```yaml
- {reward: fraction_aromatic, module: /path/to/my_rewards.py, weight: 1.0,
   shaping: {type: gaussian, target: 0.10, width: 0.5}}
```

**Yours, already written.** If IDiom is installed alongside code that can already score a sequence,
name the callable directly and change nothing about it — no decorator, no import of IDiom, no copy
into this repo:

```yaml
- {reward: "mypackage.scoring:score_idr", weight: 1.0}
- {reward: "mypackage.scoring:score_batch", batched: true, weight: 1.0}   # f(idrs, batch)
```

The term is logged under the function's own name unless you give it a `label`. A `module:function`
that cannot be imported fails while the config is parsed, not on the first training step.

### Out-of-process — a reward model in its own environment

**Only when the dependencies cannot coexist with IDiom's.** If the scorer imports in your
environment, use the in-process form above; a subprocess to call a function you could have imported
is pure overhead. But ProtGPS pins python 3.8, torch 2.0 and pytorch-lightning 1.6.4, none of which
can live beside IDiom (python >= 3.10, torch >= 2.4) — and that is what this is for.

Each shipped scorer is a standalone program speaking newline-delimited JSON on stdin/stdout, with
its environment declared in a [PEP 723](https://peps.python.org/pep-0723/) header — `uv` builds it
on first use and weights are fetched automatically, so there is no install step:

```yaml
- {cmd: "uv run --script ${idiom_scorer:sparrow.py} --property radius_of_gyration",
   label: rg, weight: 0.5, shaping: {type: quadratic, target: 25, width: 0.2}}
```

| scorer | rewards | environment and weights | per 32 seqs |
|---|---|---|---|
| `sparrow.py` | radius of gyration, asphericity, scaling exponent, FCR, kappa | [sparrow](https://github.com/idptools/sparrow); no weights | 2.8 s CPU |
| `finches.py` | epsilon, self or against a `--partner` | [finches](https://github.com/idptools/finches); parameters in-package | 0.1 s CPU |
| `pspred.py` | phase separation: dG in kT, or c_sat in mg/mL | [PSpred](https://github.com/KULL-Centre/_2024_buelow_PSpred), 3.7 MB | 4.7 s CPU |
| `protgps.py` | condensate compartment probability (ESM-2) | [ProtGPS](https://github.com/pgmikhael/protgps), py3.8/torch 2.0, 166 MB | 1.6 s CPU |
| `paddle.py` | transcriptional activation strength (max-Z) | [PADDLE](https://github.com/asanborn/PADDLE), TensorFlow, 36 MB | 3 s CPU |
| `starling.py` | ensemble Rg or end-to-end distance | [STARLING](https://github.com/idptools/starling), 1.5 GB | 9 s GPU |

Several combine in one run, each with its own environment. Check one before spending an allocation:

```bash
export UV_CACHE_DIR=/scratch/you/uv-cache   # on-demand envs are several GB

uv run python -m idiom.train.grpo.reward.external \
  --cmd "uv run --script cookbook/rewards/scorers/sparrow.py --property radius_of_gyration" \
  --shaping quadratic --target 25 --width 0.2
```

```
startup + 3 sequences in 0.5s

         raw    shaped  sequence
      4.5938    0.9952  MEEEKKKKSSSTTTDDDQQQQNNNN
      5.5250    0.5619  GSGSGSGSGSGSGSGSGSGSGSGSGSGSGS
      9.9880    0.0000  MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ
```

Every `grpo/*.bash` script runs exactly this as a pre-flight step, so a broken environment fails in
seconds instead of after the policy has warm-started. The scorers are ordinary files in the
repository — edit one in place, or copy it and point `cmd` at your copy.

**A GPU scorer shares your GPU.** `starling.py` runs on the same device as the policy, so it costs
both VRAM and time per step. Give the child its own device (`CUDA_VISIBLE_DEVICES=1 uv run
--script ...`) if you have one, or budget for the contention.

### `sae_only_<name>` — reproduce a target's SAE feature code

`sae_only_<signature>` scores an IDR by the fraction of a target's SAE feature signature firing in
it, read through a frozen IDiom + SAE lens — so a gain requires encoding the real code, not just
satisfying a classifier. Already a fraction in [0, 1], so no shaping, and no third-party dependency.
`sae_signatures.json` ships beside the reward and holds the signatures for the released SAE
(`IDIOM_SAEREWARD_CASE` selects `top30` or `private30`); a copy sits in
[`example_data/sae_features/`](example_data/sae_features/) as a reference for the format. The
workflow to follow is your own: [`feature_enrichment.py`](scripts/python/feature_enrichment.py)
writes a signature from your sequences, and `IDIOM_SAEREWARD_FEATURES` points the reward at it.

### Writing your own

Put it in **your** project, not here. Nothing needs registering ahead of time — a term names what to
import and it is loaded on demand:

```yaml
- {reward: my_reward, module: /path/to/my_rewards.py, weight: 1.0}   # registered by name
- {reward: "mypackage.scoring:score_idr", weight: 1.0}               # any importable callable
- {cmd: "uv run --script /path/to/my_scorer.py", label: mine, weight: 1.0}   # its own environment
```

Start from [`my_rewards.py`](rewards/my_rewards.py) for the first two and
[`my_scorer.py`](rewards/scorers/my_scorer.py) for the third — a working scorer in about 40 lines
that computes isoelectric point from Biopython, runs without a GPU, and carries the traps below as
comments. A scorer imports nothing from IDiom and speaks one exchange per GRPO step:

```
->  {"sequences": ["ACDEF...", "GHIKL..."]}
<-  {"scores": [24.8, 31.2]}          # or {"error": "..."}
```

One finite score per sequence, in order (a count mismatch is rejected); flush after each response;
load the model once at import.

**stdout is the protocol.** Libraries that print on import (ProtGPS, STARLING, TensorFlow) make the
run die at the handshake with `scorer wrote a non-JSON line`. Claim the real stdout:

```python
_PROTOCOL_STDOUT = sys.stdout
sys.stdout = sys.stderr              # library chatter goes to the log, not the protocol
print(json.dumps(response), file=_PROTOCOL_STDOUT, flush=True)
```

Two more traps: a scorer named after the package it wraps shadows it (drop the script's own
directory from `sys.path`), and its environment resolves its own torch, which can outrun your CUDA
driver (pin the build in the PEP 723 header). Since nothing is imported from IDiom, the same `cmd`
also covers a pre-built venv, a conda env, or `docker run -i`.

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
sae_features/  SAE feature signatures. sae_signatures.json is the released SAE's, copied from
            the package as a format reference; feature_enrichment.py writes signature.json here
```

`protgps/` and `effector/` records are fully disordered (the whole record is the span). `effector/`
headers carry extra free-text fields after the accession, which IDiom ignores.

**Provenance.** `protgps/` — [ProtGPS](https://github.com/pgmikhael/protgps) (Kilgore et al.).
`effector/` — DelRosso et al., *Nature* 2023. `disprot/` — [DisProt](https://disprot.org/), CC BY
4.0. Redistributed for demonstration only; cite the original works and check their licenses for any
other use.
