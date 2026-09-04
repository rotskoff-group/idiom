# Rewards

Everything GRPO can optimize, and how to point it at your own. **Nothing here ships in the wheel** —
these are files you read, copy, and edit, named by a run on the command line.

```
my_rewards.py     rewards that run in this process — copy this and edit it
scorers/          reward models that run in their OWN environment, one program each
```

The library defines exactly two rewards, `entropy` and `length`, because every run should carry
them. It takes no view on what you should design for, so the shipped config's objective is those two
and nothing else, and what a run optimizes is named at launch:

```bash
sbatch cookbook/slurm/grpo/sparrow.bash      # one ready-to-submit script per reward
```

| script | optimizes | reward |
|---|---|---|
| [`grpo/sae_features.bash`](../slurm/grpo/sae_features.bash) | a target's SAE feature code (the RL-SAE result) | library |
| [`grpo/my_reward.bash`](../slurm/grpo/my_reward.bash) | anything you can compute in-process | [`my_rewards.py`](my_rewards.py) |
| [`grpo/sparrow.bash`](../slurm/grpo/sparrow.bash) | radius of gyration, asphericity, kappa, FCR | [`scorers/sparrow.py`](scorers/sparrow.py) |
| [`grpo/finches.bash`](../slurm/grpo/finches.bash) | epsilon — interaction chemistry | [`scorers/finches.py`](scorers/finches.py) |
| [`grpo/protgps.bash`](../slurm/grpo/protgps.bash) | condensate compartment probability | [`scorers/protgps.py`](scorers/protgps.py) |
| [`grpo/paddle.bash`](../slurm/grpo/paddle.bash) | transcriptional activation strength | [`scorers/paddle.py`](scorers/paddle.py) |
| [`grpo/pspred.bash`](../slurm/grpo/pspred.bash) | phase-separation dG / c_sat | [`scorers/pspred.py`](scorers/pspred.py) |
| [`grpo/starling.bash`](../slurm/grpo/starling.bash) | ensemble Rg or end-to-end distance | [`scorers/starling.py`](scorers/starling.py) |
| [`grpo/my_scorer.bash`](../slurm/grpo/my_scorer.bash) | a scorer you wrote, in its own environment | [`scorers/my_scorer.py`](scorers/my_scorer.py) |

Each spells out the whole objective as one `TERM` at the top, passed to `reward.add`, so the run is
reproducible from the script alone and the config is never edited.

## How a reward is built

A run's objective is a list of **terms** in `src/idiom/configs/grpo.yaml`. Each pairs a **reward**
(one raw value in its own units) with the **shaping** that says what a good value is, and a
**weight** for how much it matters:

```yaml
reward:
  terms:
    - {reward: entropy, weight: 1.0, shaping: {type: quadratic, target: 3.65, width: 0.2}}  # bits
    - {reward: length,  weight: 1.0, shaping: {type: quadratic, target: 100,  width: 1.0}}  # residues
```

`total = Σ weightᵢ · shapingᵢ(rewardᵢ)` drives the GRPO advantages.

| shaping | shape | use when |
|---|---|---|
| `quadratic` | 0 at the target, −1 one `width` out, unbounded below | one target, hit hard |
| `gaussian` | 1 at the target decaying to 0 — bounded | several targets must coexist |
| omitted | raw value passes through | already on a sensible scale, e.g. a probability |

`width` is a tolerance in the reward's units — a fraction of a nonzero `target` (`width: 0.2` on
`target: 25` is ±5 Å), absolute when the target is 0.

`weight: 0` keeps a term running and logged without optimizing it; `enabled: false` skips it
entirely (nothing imported, no subprocess).

## Choosing the objective at launch

`grpo.yaml` carries the two guardrails and nothing else. What a run designs for is one `TERM` on the
command line, appended with `reward.add`:

```bash
idiom_grpo init_from=jxliu2/idiom-300M \
  reward.add='[{reward: sae_only_nucleolus, module: idiom.train.grpo.reward.sae_feature, weight: 1.0}]'

idiom_grpo init_from=jxliu2/idiom-300M \
  reward.add='[{cmd: "uv run --script cookbook/rewards/scorers/sparrow.py --property radius_of_gyration",
                label: rg, weight: 0.5, shaping: {type: quadratic, target: 25, width: 0.2}}]'
```

Several terms compose, in the order you name them:

```bash
idiom_grpo init_from=jxliu2/idiom-300M reward.add="[$SAE_TERM, $RG_TERM]"
```

The scripts in [`../slurm/grpo/`](../slurm/grpo/) are these commands with every knob spelled out —
copy the one closest to what you want.

## Picking a target, a width, and a weight

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

A reward whose optimum sits off the IDR distribution *will* be reached. Rewarding aromatic content at
weight 1.0 for 3000 steps hit its target exactly (0.1488 vs 0.15) while collapsing to hydrophobic
`LVIFA` segments — disorder fell from 0.665 to **0.091** — with entropy and length on target
throughout. Nothing in the reward said "still an IDR".

## The three kinds

### In-process — a function in this interpreter

Use this whenever the reward can be imported into the environment IDiom is running in, which is
most of the time. A reward is one function, one raw value, no notion of "good".

**Built in.** Name it and it works — no `module`, nothing on disk:

```yaml
- {reward: entropy, weight: 1.0, shaping: {type: quadratic, target: 3.65, width: 0.2}}
```

`entropy` and `length` (the guardrails), `net_charge_fraction`, `fraction_charged`,
`sumo_motif_count`, `ndsm_motif_count`. They live in `idiom/train/grpo/reward/builtin.py`.

**Yours, registered by name.** Copy [`my_rewards.py`](my_rewards.py) into your project, decorate,
and point a term at the file (a `*.py` path or a dotted module name):

```python
@register_reward("fraction_aromatic")
def fraction_aromatic(idr: str) -> float:
    return sum(idr.count(a) for a in "FWY") / len(idr) if idr else 0.0
```

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

Start from [`my_scorer.py`](scorers/my_scorer.py), which is a working scorer in about 40 lines of code and
carries the three traps below as comments. It computes isoelectric point from Biopython, and runs
without a GPU:

```bash
uv run python -m idiom.train.grpo.reward.external \
    --cmd "uv run --script cookbook/rewards/my_scorer.py --property isoelectric_point" \
    --shaping gaussian --target 4.5 --width 0.3
```

```
startup + 3 sequences in 0.5s

         raw    shaped  sequence
      4.5938    0.9952  MEEEKKKKSSSTTTDDDQQQQNNNN
      5.5250    0.5619  GSGSGSGSGSGSGSGSGSGSGSGSGSGSGS
      9.9880    0.0000  MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ
```

Each shipped scorer is the same standalone program
speaking newline-delimited JSON on stdin/stdout, with its environment declared in a
[PEP 723](https://peps.python.org/pep-0723/) header — `uv` builds it on first use and weights are
fetched automatically, so there is no install step:

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

```bash
idiom_grpo init_from=jxliu2/idiom-300M reward.add=[sae]
```

`idiomsae-300M-L18-k32.json` ships beside the reward and holds signatures for the released SAE
(`IDIOM_SAEREWARD_CASE` selects `top30` or `private30`). Build your own with
[`feature_enrichment.py`](../scripts/feature_enrichment.py) and point `IDIOM_SAEREWARD_FEATURES`
at it.

## Writing your own

Put it in **your** project, not here. Nothing needs registering ahead of time — a term names what
to import and it is loaded on demand:

```yaml
- {reward: my_reward, module: /path/to/my_rewards.py, weight: 1.0}   # registered by name
- {reward: "mypackage.scoring:score_idr", weight: 1.0}               # any importable callable
- {cmd: "uv run --script /path/to/my_scorer.py", label: mine, weight: 1.0}   # its own environment
```

Start from [`my_rewards.py`](my_rewards.py) for the first two and [`my_scorer.py`](scorers/my_scorer.py) for
the third. A scorer imports nothing from IDiom and speaks one exchange per GRPO step:

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
