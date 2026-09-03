# Rewards

Everything GRPO can optimize. This is **repository material, not library code**: `idiom` imports
none of it, and you are meant to read, copy, and edit it.

```
custom_rewards.py     in-process rewards: f(idr) -> float
external_rewards/     standalone scorer programs, each carrying its own environment
rl_sae_reward/        the SAE feature-code reward and its signature file
```

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
entirely (nothing imported, no subprocess). The shipped config uses the latter for a menu of terms
that cost nothing until switched on:

```bash
idiom_grpo init_from=jxliu2/idiom-300M reward.terms.2.enabled=true   # term 2 is the RL-SAE one
```

Each enabled term logs twice: `<label>` is its contribution to the objective, `<label>_raw` the raw
reward. Configs reach these files with `${idiom_rewards:...}`, which resolves wherever you launch
from because IDiom is installed from the clone.

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

### `custom_rewards.py` — in-process

One function, one raw value, no notion of "good":

```python
@register_reward("fraction_charged")
def fraction_charged(idr: str) -> float:
    return sum(idr.count(a) for a in "DEKR") / len(idr) if idr else 0.0
```

```yaml
- {reward: net_charge_fraction, module: "${idiom_rewards:custom_rewards.py}", weight: 1.0,
   shaping: {type: gaussian, target: 0.25, width: 0.5}}
```

Ships with `entropy` and `length` (the guardrails), plus `net_charge_fraction`, `fraction_charged`,
`sumo_motif_count`, and `ndsm_motif_count`. Anything computable from the residue string belongs here.

### `external_rewards/` — a reward model in its own environment

For a scorer whose dependencies can't coexist with IDiom's. Each file is a standalone program
speaking newline-delimited JSON on stdin/stdout, with its environment declared in a
[PEP 723](https://peps.python.org/pep-0723/) header — `uv` builds it on first use and weights are
fetched automatically, so there is no install step:

```yaml
- {cmd: "uv run --script ${idiom_rewards:external_rewards/sparrow.py} --property radius_of_gyration",
   label: rg, weight: 0.5, shaping: {type: quadratic, target: 25, width: 0.2}}
```

| scorer | rewards | environment and weights | per 32 seqs |
|---|---|---|---|
| [`sparrow.py`](external_rewards/sparrow.py) | radius of gyration, asphericity, scaling exponent, FCR, kappa | [sparrow](https://github.com/idptools/sparrow); no weights | 2.8 s CPU |
| [`finches.py`](external_rewards/finches.py) | epsilon, self or against a `--partner` | [finches](https://github.com/idptools/finches); parameters in-package | 0.1 s CPU |
| [`pspred.py`](external_rewards/pspred.py) | phase separation: dG in kT, or c_sat in mg/mL | [PSpred](https://github.com/KULL-Centre/_2024_buelow_PSpred), 3.7 MB | 4.7 s CPU |
| [`protgps.py`](external_rewards/protgps.py) | condensate compartment probability (ESM-2) | [ProtGPS](https://github.com/pgmikhael/protgps), py3.8/torch 2.0, 166 MB | 1.6 s CPU |
| [`paddle.py`](external_rewards/paddle.py) | transcriptional activation strength (max-Z) | [PADDLE](https://github.com/asanborn/PADDLE), TensorFlow, 36 MB | 3 s CPU |
| [`starling.py`](external_rewards/starling.py) | ensemble Rg or end-to-end distance | [STARLING](https://github.com/idptools/starling), 1.5 GB | 9 s GPU |

Several combine in one run, each with its own environment. Check one before spending an allocation:

```bash
export UV_CACHE_DIR=/scratch/you/uv-cache   # on-demand envs are several GB
uv run python -m idiom.train.grpo.reward.external \
  --cmd "uv run --script rewards/external_rewards/sparrow.py --property radius_of_gyration" \
  --shaping quadratic --target 25 --width 0.2
```

### `rl_sae_reward/` — reproduce a target's SAE feature code

`sae_only_<signature>` scores an IDR by the fraction of a target's SAE feature signature firing in
it, read through a frozen IDiom + SAE lens — so a gain requires encoding the real code, not just
satisfying a classifier. Already a fraction in [0, 1], so no shaping, and no third-party dependency.

```bash
idiom_grpo init_from=jxliu2/idiom-300M \
  reward.terms.2.enabled=true reward.terms.2.reward=sae_only_nucleolus
```

`idiomsae-300M-L18-k32.json` holds signatures for the released SAE (`IDIOM_SAEREWARD_CASE` selects
`top30` or `private30`). Build your own with
[`cookbook/scripts/feature_enrichment.py`](../cookbook/scripts/feature_enrichment.py) and point
`IDIOM_SAEREWARD_FEATURES` at it.

## Writing your own

Put the file in **your** project, not here. Nothing needs registering ahead of time — a term names
the file and it is imported on demand:

```yaml
- {reward: my_reward, module: /path/to/my_rewards.py, weight: 1.0}
- {cmd: "uv run --script /path/to/my_scorer.py", label: mine, weight: 1.0}
```

A scorer — copy [`external_rewards/sparrow.py`](external_rewards/sparrow.py) — imports nothing from
IDiom and speaks one exchange per GRPO step:

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
