# Rewards

**Nothing here ships in the wheel** — these are files you read, copy, and edit, named by a run on
the command line.

```
custom_rewards.py   rewards that run in this process — copy this and edit it
custom_shaping.py   what a good value is, when the three shipped rules don't fit
scorers/            reward models that run in their OWN environment, one program each
```

The library defines exactly two rewards, `entropy` and `length`, because every run should carry them.
It takes no view on what you should design for, so the shipped objective is those two and nothing
else, and what a run optimizes is named at launch. Each script in [`scripts/grpo/`](../scripts/grpo/)
spells out one whole objective as a `TERM` at the top, passed to `reward.add`:

| script | optimizes | reward |
|---|---|---|
| [`sae_features.bash`](../scripts/grpo/sae_features.bash) | a target's SAE feature code (the RL-SAE result) | library |
| [`custom_reward.bash`](../scripts/grpo/custom_reward.bash) | anything you can compute in-process | [`custom_rewards.py`](custom_rewards.py) |
| [`sparrow.bash`](../scripts/grpo/sparrow.bash) | radius of gyration, asphericity, kappa, FCR | [`scorers/sparrow.py`](scorers/sparrow.py) |
| [`finches.bash`](../scripts/grpo/finches.bash) | epsilon — interaction chemistry | [`scorers/finches.py`](scorers/finches.py) |
| [`protgps.bash`](../scripts/grpo/protgps.bash) | condensate compartment probability | [`scorers/protgps.py`](scorers/protgps.py) |
| [`paddle.bash`](../scripts/grpo/paddle.bash) | transcriptional activation strength | [`scorers/paddle.py`](scorers/paddle.py) |
| [`pspred.bash`](../scripts/grpo/pspred.bash) | phase-separation dG / c_sat | [`scorers/pspred.py`](scorers/pspred.py) |
| [`starling.bash`](../scripts/grpo/starling.bash) | ensemble Rg or end-to-end distance | [`scorers/starling.py`](scorers/starling.py) |
| [`custom_scorer.bash`](../scripts/grpo/custom_scorer.bash) | a scorer you wrote, in its own environment | [`scorers/custom_scorer.py`](scorers/custom_scorer.py) |

## How a reward is built

A run's objective is a list of **terms**. Each pairs a **reward** (one raw value in its own units)
with the **shaping** that says what a good value is, and a **weight** for how much it matters;
`total = Σ weightᵢ · shapingᵢ(rewardᵢ)` drives the advantages.

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

Those three cover a value you want *hit*. For anything else — a floor, a ceiling, a flat band —
register your own; see [writing your own shaping](#writing-your-own-shaping).

`width` is a tolerance in the reward's units — a fraction of a nonzero `target` (`width: 0.2` on
`target: 25` is ±5 Å), absolute when the target is 0. `weight: 0` keeps a term running and logged
without optimizing it; `enabled: false` skips it entirely.

## Choosing the objective at launch

`grpo.yaml` carries the two guardrails and nothing else. What a run designs for is one `TERM` on the
command line, appended with `reward.add`; several compose in the order you name them.

```bash
idiom_train_grpo init_from=jxliu2/idiom-300M \
  reward.add='[{reward: sae_only_nucleolus, module: idiom.train.grpo.reward.sae_feature, weight: 1.0}]'

idiom_train_grpo init_from=jxliu2/idiom-300M \
  reward.add='[{cmd: "uv run --script cookbook/rewards/scorers/sparrow.py --property radius_of_gyration",
                label: rg, weight: 0.5, shaping: {type: quadratic, target: 25, width: 0.2}}]'

idiom_train_grpo init_from=jxliu2/idiom-300M reward.add="[$SAE_TERM, $RG_TERM]"
```

An entry may also be a **name** from `reward.presets`, which is how you keep a tuned term without
retyping it: put a `presets` block in a config of your own, launch with `--config-dir . --config-name
my_grpo`, then `reward.add=[my_term]` and `reward.presets.my_term.shaping.target=30` to tune it. The
shipped `presets` is empty — there is no menu to pick from.

## In-process rewards

Use this whenever the reward can be imported into the environment IDiom runs in, which is most of the
time. A reward is one function, one raw value, no notion of "good".

**Built in.** Name it and it works — no `module`, nothing on disk: `entropy` and `length` (the
guardrails), `net_charge_fraction`, `fraction_charged`, `sumo_motif_count`, `ndsm_motif_count`, in
`idiom/train/grpo/reward/builtin.py`.

**Yours, registered by name.** Copy [`custom_rewards.py`](custom_rewards.py) into your project,
decorate with `@register_reward("fraction_aromatic")`, and point a term at the file (a `*.py` path or
a dotted module name):

```yaml
- {reward: fraction_aromatic, module: /path/to/custom_rewards.py, weight: 1.0,
   shaping: {type: gaussian, target: 0.10, width: 0.5}}
```

**Yours, already written.** If IDiom is installed alongside code that can already score a sequence,
name the callable directly — no decorator, no import of IDiom, no copy into this repo:

```yaml
- {reward: "mypackage.scoring:score_idr", weight: 1.0}
- {reward: "mypackage.scoring:score_batch", batched: true, weight: 1.0}   # f(idrs, batch)
```

The term is logged under the function's own name unless you give it a `label`. A `module:function`
that cannot be imported fails while the config is parsed, not on the first training step.

## Out-of-process scorers

**Only when the dependencies cannot coexist with IDiom's.** If the scorer imports in your
environment, use the in-process form; a subprocess to call a function you could have imported is pure
overhead. But ProtGPS pins python 3.8, torch 2.0 and pytorch-lightning 1.6.4, none of which can live
beside IDiom (python >= 3.10, torch >= 2.4) — and that is what this is for.

Each shipped scorer is a standalone program speaking newline-delimited JSON on stdin/stdout, with its
environment declared in a [PEP 723](https://peps.python.org/pep-0723/) header. `uv` builds it on
first use and weights are fetched automatically, so there is no install step:

```yaml
- {cmd: "uv run --script cookbook/rewards/scorers/sparrow.py --property radius_of_gyration",
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

The scorers are ordinary files in the repository — edit one in place, or copy it and point `cmd` at
your copy. Several combine in one run, each with its own environment. Check one before spending an
allocation:

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

Every `scripts/grpo/*.bash` runs exactly this as a pre-flight step, so a broken environment fails in
seconds instead of after the policy has warm-started.

**A GPU scorer shares your GPU.** `starling.py` runs on the same device as the policy, so it costs
both VRAM and time per step. Give the child its own device (`CUDA_VISIBLE_DEVICES=1 uv run --script
...`) if you have one, or budget for the contention.

## `sae_only_<name>`

`sae_only_<signature>` scores an IDR by the fraction of a target's SAE feature signature firing in
it, read through a frozen IDiom + SAE lens — so a gain requires encoding the real code, not just
satisfying a classifier. Already a fraction in [0, 1], so no shaping, and no third-party dependency.
`sae_signatures.json` ships beside the reward and holds the signatures for the released SAE
(`IDIOM_SAEREWARD_CASE` selects `top30` or `private30`); a copy sits in the Hub dataset under
`example_data/sae_features/` as a format reference.
[`feature_enrichment.ipynb`](../notebooks/feature_enrichment.ipynb) writes a signature from your own
sequences, and `IDIOM_SAEREWARD_FEATURES` points the reward at it.

## Picking a target, a width, and a weight

The failure mode is quiet — the objective improves while the sequences stop being IDRs.

**Keep `entropy` and `length` on.** They cost nothing and contribute about **−1.8** to a starting
sequence. **Size every new term against that**: pick `width` so a typical base generation sits about
one width from the target, and `weight` so the starting contribution is order 1. A term worth −30 at
step 0 has made the guardrails invisible — `finches` aimed at −6.0 with `width: 1.0` started at −31.5
and swamped everything; widened to match the spread of base generations it starts at −3.8 and
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

## Writing your own

Put it in **your** project, not here. Nothing needs registering ahead of time — a term names what to
import and it is loaded on demand:

```yaml
- {reward: my_reward, module: /path/to/custom_rewards.py, weight: 1.0}   # registered by name
- {reward: "mypackage.scoring:score_idr", weight: 1.0}               # any importable callable
- {cmd: "uv run --script /path/to/custom_scorer.py", label: mine, weight: 1.0}   # its own environment
```

Start from [`custom_rewards.py`](custom_rewards.py) for the first two and
[`custom_scorer.py`](scorers/custom_scorer.py) for the third — a working scorer in about 40 lines
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

Two more traps: a scorer named after the package it wraps shadows it (drop the script's own directory
from `sys.path`), and its environment resolves its own torch, which can outrun your CUDA driver (pin
the build in the PEP 723 header). Since nothing is imported from IDiom, the same `cmd` also covers a
pre-built venv, a conda env, or `docker run -i`.

## Writing your own shaping

Shaping extends the same way a reward does: a decorator, in a file you name at launch. The three
shipped rules all say *be here*, which is wrong whenever the objective is a threshold — an IDR that
must stay expanded wants Rg ≥ 30 Å, not Rg = 30 Å, and pinning it to the target spends optimization
pressure fighting improvements.

[`custom_shaping.py`](custom_shaping.py) is that rule, `one_sided`, and the file to copy:

```python
from idiom.train.grpo.reward import register_shaping, tolerance

@register_shaping("one_sided")
def one_sided(*, target: float, width: float = 1.0, direction: str = "above"):
    if direction not in ("above", "below"):
        raise ValueError(f"one_sided direction must be 'above' or 'below', got {direction!r}")
    scale = tolerance(target, width)
    sign = 1.0 if direction == "above" else -1.0
    def shaping(value: float) -> float:
        deficit = sign * (target - value)
        return -((deficit / scale) ** 2) if deficit > 0 else 0.0
    return shaping
```

A factory takes the spec's parameters and returns the rule itself, `f(raw) -> float`. Two things to
keep: use `tolerance` rather than dividing by `width` yourself, so `width` stays a fraction of a
nonzero target and absolute at 0; and validate in the factory, which runs once at config time, so a
bad `direction` fails in seconds instead of on the first step. A `TypeError` from a missing or
misspelled parameter is already reported as `bad parameters for shaping ...`.

Name it in a term's `shaping.type`, and bring the file in with the config-level `reward.module=`,
which is imported before every term:

```bash
idiom_train_grpo init_from=jxliu2/idiom-300M reward.module=cookbook/rewards/custom_shaping.py \
  reward.add='[{reward: fraction_charged, module: cookbook/rewards/custom_rewards.py, weight: 1.0,
                shaping: {type: one_sided, target: 0.30, width: 0.5}}]'
```

A term takes one `module`, which is why the shaping goes in `reward.module` here — the term's own is
already spent on the reward. If both live in one file of yours, that file in the term's `module` is
enough; every module a term names is imported before any shaping is built.

**Flat means no gradient.** A one-sided term stops steering once a completion clears the threshold,
so it cannot be the only thing driving a run — pair it with a term that has a preference, or the
policy settles just past the threshold and stays there.
