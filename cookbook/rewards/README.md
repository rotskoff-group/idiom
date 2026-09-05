# Rewards

**Nothing here ships in the wheel** — these are files you read, copy, and edit, named by a run on
the command line.

```
custom_rewards.py   reward and shaping factories that run in this process — copy and edit
scorers/            reward models that run in their OWN environment, one program each
```

The library takes no view on what you should design for, so **the shipped objective is empty**:
`reward.terms` is `[]` and there are no presets and no default terms. A run names every term it
optimizes, which makes the launch line the whole objective. Each script in
[`scripts/grpo/`](../scripts/grpo/) writes one out and passes it to `reward.terms`:

| script | objective | reward |
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
| [`combined.bash`](../scripts/grpo/combined.bash) | epsilon and a compartment at once, two scorers | [`scorers/finches.py`](scorers/finches.py), [`scorers/protgps.py`](scorers/protgps.py) |

## How a reward is built

A run's objective is a list of **terms**. Each pairs a **reward** (one raw value in its own units)
with the **shaping** that says what a good value is, and a **weight** for how much it matters;
`total = Σ weightᵢ · shapingᵢ(rewardᵢ)` drives the advantages.

```yaml
reward:
  terms:
    - {reward: entropy, shaping: {name: quadratic, target: 3.65, width: 0.2}, weight: 1.0}  # bits
    - {reward: length,  shaping: {name: quadratic, target: 100,  width: 1.0}, weight: 1.0}  # residues
```

**A term is four keys** — `reward`, `shaping`, `weight`, `label` — and there are no others. The
reward and the shaping are named the same way: a **name**, plus that thing's own arguments. A bare
string when it takes none, a mapping when it does:

```yaml
- {reward: entropy, weight: 1.0}                                    # no arguments
- {reward: {name: scorer, cmd: "uv run --script my_scorer.py", timeout: 600},
   label: mine, weight: 1.0}                                        # scorer's arguments
```

The name is either one the library ships or a `module:function` path to a factory of your own:

| | shipped names | your own |
|---|---|---|
| `reward` | `entropy`, `length`, `scorer`, `sae_signature` | `"mypkg.scoring:make_scorer"` |
| `shaping` | `quadratic`, `gaussian`, `identity` | `"mypkg.shaping:one_sided"` |

`label` defaults to the reward's name, and is what the term is logged under; give one explicitly
when two terms would otherwise collide (two `scorer` terms, say).

| shaping | shape | use when |
|---|---|---|
| `quadratic` | 0 at the target, −1 one `width` out, unbounded below | one target, hit hard |
| `gaussian` | 1 at the target decaying to 0 — bounded | several targets must coexist |
| `identity` (or omit `shaping`) | raw value passes through | already on a sensible scale, e.g. a probability |

Those three cover a value you want *hit*. For anything else — a floor, a ceiling, a flat band —
register your own; see [writing your own shaping](#writing-your-own-shaping).

`width` is a tolerance in the reward's units — a fraction of a nonzero `target` (`width: 0.2` on
`target: 25` is ±5 Å), absolute when the target is 0. `weight: 0` keeps a term running and logged
without optimizing it; to take a term out entirely, delete it.

## Naming the objective at launch

`grpo.yaml` carries no terms at all, so a run passes the list it wants to `reward.terms` and gets
exactly that. Terms compose in the order you name them.

```bash
idiom_train_grpo init_from=jxliu2/idiom-300M \
  reward.terms='[{reward: entropy, shaping: {name: quadratic, target: 3.65, width: 0.2}, weight: 1.0},
                 {reward: length,  shaping: {name: quadratic, target: 100,  width: 1.0}, weight: 1.0},
                 {reward: {name: scorer, cmd: "uv run --script cookbook/rewards/scorers/sparrow.py --property radius_of_gyration"},
                  shaping: {name: quadratic, target: 25, width: 0.2}, label: rg, weight: 0.5}]'
```

In a script that gets unreadable fast, so each one names its terms first and assembles them on one
line. Written in a fixed key order — `label`, `weight`, `reward`, `shaping` — they line up as a
table, and the objective is the one thing that differs between scripts:

```bash
ENTROPY='{label: entropy, weight: 1.0, reward: entropy, shaping: {name: quadratic, target: 3.65, width: 0.2}}'
LENGTH='{label: length,  weight: 1.0, reward: length,  shaping: {name: quadratic, target: 100,  width: 1.0}}'
RG="{label: rg, weight: 0.5, reward: {name: scorer, cmd: \"$SCORER\"}, shaping: {name: quadratic, target: 25, width: 0.2}}"

idiom_train_grpo init_from=jxliu2/idiom-300M reward.terms="[$ENTROPY, $LENGTH, $RG]"
```

Drop `$LENGTH` and there is no length term; keep only `$RG` and that is the whole objective. An empty
list is refused, since it would give every completion the same reward.

## In-process rewards

Use this whenever the reward can be imported into the environment IDiom runs in, which is most of the
time. A reward is one function, one raw value, no notion of "good".

**Shipped.** `entropy` and `length` (`idiom/train/grpo/reward/builtin.py`) need nothing on disk —
name one and it works. They are terms like any other; nothing puts them in an objective for you.

**Yours.** A reward is a **factory**: a function returning the thing that runs every step, which
maps the step's IDRs to one raw value each. `lift` covers the usual case, a function that scores one
IDR:

```python
from idiom.train.grpo.reward import lift

def fraction_aromatic():
    return lift(lambda idr: sum(idr.count(a) for a in "FWY") / len(idr) if idr else 0.0)
```

Name it by its path — a dotted module or a `*.py` file — and nothing needs registering:

```yaml
- {reward: "mypackage.scoring:fraction_aromatic", shaping: {name: gaussian, target: 0.10, width: 0.5},
   weight: 1.0}
- {reward: "/path/to/custom_rewards.py:fraction_aromatic", weight: 1.0}
```

**With settings.** The factory's parameters are the term's arguments, so a reward that needs
configuring takes it from the config and the run's saved `config.yaml` records what it was actually
optimizing:

```yaml
- {reward: {name: "mypackage.scoring:make_scorer", cutoff: 0.3}, label: mine, weight: 1.0}
```

Validate in the factory: it runs once at config time, so a bad setting fails in seconds rather than
on the first step. A path that cannot be imported fails there too.

**Skip `lift`** when scoring the whole step at once is cheaper — a GPU forward pass, a vectorized
model — and return a `list[str] -> list[float]` directly.

## Out-of-process scorers

**Only when the dependencies cannot coexist with IDiom's.** If the scorer imports in your
environment, use the in-process form; a subprocess to call a function you could have imported is pure
overhead. But ProtGPS pins python 3.8, torch 2.0 and pytorch-lightning 1.6.4, none of which can live
beside IDiom (python >= 3.10, torch >= 2.4) — and that is what this is for.

Each shipped scorer is a standalone program speaking newline-delimited JSON on stdin/stdout, with its
environment declared in a [PEP 723](https://peps.python.org/pep-0723/) header. `uv` builds it on
first use and weights are fetched automatically, so there is no install step:

`scorer` is the reward factory that runs one: give it the command, and optionally `timeout`,
`maxlen`, `cwd`, `env` and `label` (which prefixes the child's stderr).

```yaml
- {reward: {name: scorer,
            cmd: "uv run --script cookbook/rewards/scorers/sparrow.py --property radius_of_gyration"},
   shaping: {name: quadratic, target: 25, width: 0.2}, label: rg, weight: 0.5}
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
your copy. Several combine in one run, each with its own environment; see
[`combined.bash`](../scripts/grpo/combined.bash). A scorer's settings go in its **command line**, so
that the term — and therefore the run's saved config — records what was scored; `env: {KEY: value}`
on the term covers a scorer whose only knob is an environment variable. Check one before spending an
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

## The SAE feature reward

`sae_signature` scores an IDR by the fraction of a target's SAE feature signature firing in it, read
through a frozen IDiom + SAE lens — so a gain requires encoding the real code, not just satisfying a
classifier. Already a fraction in [0, 1], so no shaping, and no third-party dependency. Which
signature a run chases and where it is read from are the reward's arguments, saved with the run:

```yaml
- {reward: {name: sae_signature, signature: nucleolus, features: signature.json, case: top30},
   label: sae, weight: 1.0}
```

It takes `signature` (required), `features` (path to the signature JSON), `case` (`top30` or
`private30` in the shipped file), `sae` (the lens, a Hub repo id or directory) and `device`.
`sae_signatures.json` ships beside the reward and holds the signatures for the released SAE; a copy
sits in the Hub dataset under `example_data/sae_features/` as a format reference.
[`feature_enrichment.ipynb`](../notebooks/feature_enrichment.ipynb) writes a signature from your own
sequences — point `params.features` at it.

## Picking a target, a width, and a weight

The failure mode is quiet — the objective improves while the sequences stop being IDRs.

**Name `entropy` and `length` in most objectives.** Nothing adds them for you, and without them a
target is satisfiable by a low-complexity tract or a degenerate length. At the weights above they
cost nothing and contribute about **−1.8** to a starting sequence. **Size every new term against
that**: pick `width` so a typical base generation sits about one width from the target, and `weight`
so the starting contribution is order 1. A term worth −30 at step 0 has made them invisible —
`finches` aimed at −6.0 with `width: 1.0` started at −31.5 and swamped everything; widened to match
the spread of base generations it starts at −3.8 and converges with them intact.

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
`LVIFA` segments — disorder fell from 0.665 to **0.091** — with the entropy and length terms on
target throughout. Nothing in the reward said "still an IDR".

## Writing your own

Put it in **your** project, not here. Nothing needs registering ahead of time — a term names what to
import and it is loaded on demand:

```yaml
- {reward: "/path/to/custom_rewards.py:my_reward", weight: 1.0}          # a factory in a file
- {reward: {name: "mypackage.scoring:make_scorer", cutoff: 0.3}, label: mine, weight: 1.0}
- {reward: {name: scorer, cmd: "uv run --script /path/to/custom_scorer.py"}, label: mine, weight: 1.0}
```

Start from [`custom_scorer.py`](scorers/custom_scorer.py) — a working scorer in about 40 lines that
computes isoelectric point from Biopython and runs without a GPU. A scorer imports nothing from
IDiom and speaks one exchange per GRPO step:

```
->  {"sequences": ["ACDEF...", "GHIKL..."]}
<-  {"scores": [24.8, 31.2]}          # or {"error": "..."}
```

You write one function and paste one:

```python
def build():
    from mypredictor import predict            # heavy imports live here, not at module top
    return lambda seqs: [predict(s) for s in seqs]   # one finite score per sequence, in order

serve(build)                                   # pasted verbatim from any shipped scorer
```

`build()` returns `score_batch`, a function mapping a list of residue strings to one raw value each
(raise on a bad one — `serve` turns it into an `{"error": ...}` the run surfaces). Copy `serve`
unchanged: it drives the protocol and handles the two traps every scorer hits — **stdout is the
protocol**, so a library that prints on import (TensorFlow, ProtGPS, STARLING) would corrupt the
first response, and a scorer named after the package it wraps (`sparrow.py` importing `sparrow`)
shadows it. `serve` claims stdout and drops the script's own directory from `sys.path` before
calling `build()`, so neither can bite you. It cannot be a shared import — `uv run --script` runs
one isolated file — so it is pasted, identical, into every scorer.

One trap `serve` cannot take: a scorer's environment resolves its own torch, which can outrun your
CUDA driver — pin the build in the PEP 723 header. Since nothing is imported from IDiom, the same
`cmd` also covers a pre-built venv, a conda env, or `docker run -i` — even a program in another
language, as long as it speaks the JSON lines above.

## Writing your own shaping

Shaping is written exactly as a reward is — a factory, named by its path — and goes in the same
file. The three shipped rules all say *be here*, which is wrong whenever the objective
is a threshold: an IDR that must stay expanded wants Rg ≥ 30 Å, not Rg = 30 Å, and pinning it to the
target spends optimization pressure fighting improvements.

The Shaping section of [`custom_rewards.py`](custom_rewards.py) is that rule, `one_sided`:

```python
from idiom.train.grpo.reward import tolerance

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

A factory takes the rule's arguments and returns the rule itself, `f(raw) -> float`. Two things to
keep: use `tolerance` rather than dividing by `width` yourself, so `width` stays a fraction of a
nonzero target and absolute at 0; and validate in the factory, which runs once at config time, so a
bad `direction` fails in seconds instead of on the first step. A `TypeError` from a missing or
misspelled argument is already reported as `bad arguments for shaping ...`.

Name it in `shaping.name`, by the same kind of path a reward uses:

```bash
idiom_train_grpo init_from=jxliu2/idiom-300M \
  reward.terms='[{reward: "cookbook/rewards/custom_rewards.py:fraction_charged", weight: 1.0,
                  shaping: {name: "cookbook/rewards/custom_rewards.py:one_sided",
                            target: 0.30, width: 0.5}}]'
```

The reward and the shaping resolve independently, so they can live in different files or the same
one; neither needs the other to have been imported first.

**Flat means no gradient.** A one-sided term stops steering once a completion clears the threshold,
so it cannot be the only thing driving a run — pair it with a term that has a preference, or the
policy settles just past the threshold and stays there.
