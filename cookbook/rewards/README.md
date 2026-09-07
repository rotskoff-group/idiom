# Rewards

GRPO maximizes a weighted sum of shaped rewards:

`total = Σ weight × shaping(raw reward)`

Configure at least one term in `reward.terms`; IDiom adds no terms automatically.
For setup and launching scripts, see [running examples](../scripts/README.md#running-scripts).

# Examples

Clone the repository to access these scripts and their scorer files.

| Script | Objective |
|---|---|
| [sae_features.bash](../scripts/grpo/sae_features.bash) | Match an SAE feature signature |
| [custom_reward.bash](../scripts/grpo/custom_reward.bash) | Use a Python reward in IDiom's environment |
| [sparrow.bash](../scripts/grpo/sparrow.bash) | Target radius of gyration or another sequence property |
| [finches.bash](../scripts/grpo/finches.bash) | Target self- or partner-interaction epsilon |
| [protgps.bash](../scripts/grpo/protgps.bash) | Increase a compartment probability |
| [paddle.bash](../scripts/grpo/paddle.bash) | Increase predicted activation strength |
| [pspred.bash](../scripts/grpo/pspred.bash) | Target phase-separation free energy or saturation concentration |
| [starling.bash](../scripts/grpo/starling.bash) | Target ensemble radius of gyration or end-to-end distance |
| [custom_scorer.bash](../scripts/grpo/custom_scorer.bash) | Use a scorer in a separate environment |
| [combined.bash](../scripts/grpo/combined.bash) | Combine FINCHES and ProtGPS |

# Configuring terms

A reward measures a sequence property. Shaping converts that measurement into an objective, and
the weight scales its contribution. For example, this term favors sequences of 100 residues:

```yaml
reward:
  terms:
    - label: length
      reward: length
      shaping:
        name: quadratic
        target: 100
        width: 0.2
      weight: 1.0
```

| Field | Meaning | Default |
|---|---|---|
| `reward` | Reward factory name, with optional arguments | Required |
| `shaping` | Shaping factory name, with optional arguments | `identity` |
| `weight` | Multiplier applied after shaping | `1.0` |
| `label` | Name used for logging | Reward name |

Use a bare name when a factory takes no arguments (`reward: length`), or a mapping with `name`
and its arguments, as shown for shaping above. Custom factories are named by
`package.module:function` or `/path/to/file.py:function`; no registration is needed.
Reward arguments belong inside `reward`, and shaping arguments inside `shaping`.

Give terms distinct labels, especially when using multiple `scorer` rewards.
Logs include `train/reward_<label>_raw` for the measurement and `train/reward_<label>` for its
weighted contribution. A zero-weight term is still evaluated and logged; delete it to stop evaluating it.

The Bash examples define each term in a variable and pass the list through `reward.terms`.
See [combined.bash](../scripts/grpo/combined.bash) for a complete multi-term launch.

# Shaping

| Name | Behavior | Use |
|---|---|---|
| `quadratic` | 0 at the target, −1 one tolerance away, unbounded below | Penalize deviations from a target |
| `gaussian` | 1 at the target, approximately 0.37 one tolerance away, approaches 0 far away | Bound a target term's contribution |
| `identity` | Passes the raw value through | Maximize a measurement with positive weight, or minimize it with negative weight |

For quadratic and Gaussian shaping, tolerance is `abs(target) × width`.
When `target == 0`, tolerance is `width` in the reward's units. Width must be positive and defaults
to `1.0`. In the example above, `width: 0.2` gives a tolerance of 20 residues.

For a floor, ceiling, or acceptable band, use [custom shaping](#writing-your-own-shaping).

# Built-in rewards

| Name | Raw value |
|---|---|
| `entropy` | Shannon entropy of amino-acid composition, in bits |
| `length` | Number of residues |
| `sae_signature` | Fraction of signature features active in an IDR |
| `scorer` | Value returned by an external program |

## SAE signatures

The SAE reward uses a frozen IDiom model and SAE to measure which signature features activate
at any residue. Its score is in [0, 1], so identity shaping with positive weight rewards more matches.

```yaml
reward:
  name: sae_signature
  signature: nucleolus
  features: /path/to/signature.json
  case: top30
```

This is the `reward` field of a term. `signature` is required; `features` defaults to the bundled
signature file and `case` to `top30`. Optional `sae` and `device` select the SAE model and device.
The file format is `{case: {signature: [feature_ids]}}`.

To build a signature from your sequences, run [feature_enrichment.ipynb](../notebooks/feature_enrichment.ipynb)
and set the script's `FEATURES`, `SIGNATURE`, and `CASE` to match its output.

# External scorers

Use `scorer` when a reward needs a separate environment. Install `uv` with
`python -m pip install uv`; each script declares its dependencies and downloads required weights
on first use. The scorer process persists across batches.

```yaml
reward:
  name: scorer
  cmd: "uv run --script cookbook/rewards/scorers/sparrow.py --property radius_of_gyration"
  timeout: 600
```

This is the `reward` field of a term. Optional settings include `maxlen` (truncate inputs),
`cwd`, `env`, `cache_max`, and `label` (the scorer's stderr prefix).
The adapter deduplicates sequences, caches scores across steps, and assigns empty sequences raw zero.
Set `cache_max: 0` to disable caching across steps for stochastic scorers such as STARLING;
duplicates within a batch still share one score. The first actual batch validates the protocol,
and `timeout` covers writing the request and waiting for its response, including initial model loading.

| Scorer | Properties | Source |
|---|---|---|
| [sparrow.py](scorers/sparrow.py) | Radius of gyration, asphericity, scaling exponent, FCR, kappa | [sparrow](https://github.com/idptools/sparrow) |
| [finches.py](scorers/finches.py) | Self- or partner-interaction epsilon | [FINCHES](https://github.com/idptools/finches) |
| [pspred.py](scorers/pspred.py) | Free energy (kT) or saturation concentration | [PSpred](https://github.com/KULL-Centre/_2024_buelow_PSpred) |
| [protgps.py](scorers/protgps.py) | Compartment probabilities | [ProtGPS](https://github.com/pgmikhael/protgps) |
| [paddle.py](scorers/paddle.py) | Activation strength (max-Z) | [PADDLE](https://github.com/asanborn/PADDLE) |
| [starling.py](scorers/starling.py) | Ensemble radius of gyration or end-to-end distance | [STARLING](https://github.com/idptools/starling) |

From the clone, check a scorer before training:

```bash
python -m idiom.train.grpo.reward.external \
  --cmd "uv run --script cookbook/rewards/scorers/sparrow.py --property radius_of_gyration" \
  --shaping quadratic --target 25 --width 0.2
```

First use may take time to install dependencies and download models. Set `UV_CACHE_DIR` if you need
a different cache location. GPU scorers also need memory alongside the policy; the STARLING script
assigns a separate GPU by default.

# Choosing targets and weights

1. Score representative base-model generations to estimate each property's range.
2. Choose a target supported by your application and reference sequences.
3. Choose widths and weights so no term unintentionally dominates the initial objective.
4. Monitor raw measurements as well as the total reward. Use zero-weight terms for extra diagnostics.

Include entropy and length terms when you want to discourage low-complexity or extreme-length
solutions. They do not guarantee disorder: a sequence can match both targets while becoming hydrophobic.
Measure disorder separately when it matters to the design.

Quadratic penalties can dominate the sum far from their target. Gaussian scores remain bounded,
but can become effectively zero across a batch and provide little signal for distinguishing sequences.

<details>
<summary>Reference measurements from base idiom-300M generations</summary>

These measurements are starting points for calibration; check them on your own sample.

| Quantity | Base value |
|---|---|
| Composition entropy | 3.66 bits |
| Length | Mean 88 residues; median 56 |
| Radius of gyration (sparrow) | 26.9 ± 15.7 Å |
| Fraction charged | 0.243 |
| Self-interaction epsilon (FINCHES) | +3.6 ± 7.0 |
| Transfer free energy (PSpred) | −0.12 ± 0.83 kT |
| Activation strength (PADDLE) | +0.94 ± 1.68 Z |
| Predicted disorder (metapredict) | 0.665 |

± denotes standard deviation. In one aromatic-content optimization, the fraction reached 0.1488
against a target of 0.15 while predicted disorder fell from 0.665 to 0.091, despite entropy and
length remaining on target.

</details>

# Writing your own reward

A factory runs once during setup and returns a function mapping `list[str]` to `list[float]`.
Return one finite score per sequence, in input order. For a function that scores one sequence,
`batchify` supplies the batch wrapper:

```python
from idiom.train.grpo.reward import batchify

def fraction_aromatic():
    return batchify(lambda seq: sum(seq.count(a) for a in "FWY") / len(seq) if seq else 0.0)
```

Save it in your project and refer to it in a term:

```yaml
reward: /path/to/custom_rewards.py:fraction_aromatic
shaping: {name: gaussian, target: 0.10, width: 0.5}
weight: 1.0
```

Factories can take keyword arguments supplied alongside `name`. Validate those arguments during
setup. For batched model inference, return a batch-scoring function directly instead of using `batchify`.
See [custom_rewards.py](custom_rewards.py) for configurable examples.

## Writing an external scorer

Copy [custom_scorer.py](scorers/custom_scorer.py) together with
[_scorer_protocol.py](scorers/_scorer_protocol.py), and edit the dependency header and `build()` function.
Keep the helper adjacent to the scorer and retain the `serve(build)` call. Put heavy imports
inside `build()`; it returns a batch-scoring function.
Use dependency versions compatible with your hardware.

The program reads and writes one JSON object per line:

```text
→ {"sequences": ["ACDEF", "GHIKL"]}
← {"scores": [24.8, 31.2]}
```

Return an `{"error": "..."}` object on failure. Stdout is reserved for responses; use stderr for logs.
The shared `serve()` validates requests and output counts, rejects non-finite scores, redirects
Python library output and prevents the script filename from shadowing the package it imports. Scorers do not need to import IDiom.

# Writing your own shaping

A shaping factory returns `float -> float`. Supply its path in `shaping.name`, with arguments
beside the name. The [one_sided example](custom_rewards.py) penalizes values on the wrong side of a
threshold and assigns zero penalty on the acceptable side:

```yaml
reward: cookbook/rewards/custom_rewards.py:fraction_charged
shaping:
  name: cookbook/rewards/custom_rewards.py:one_sided
  target: 0.30
  width: 0.5
  direction: above
weight: 1.0
```

For your own rules, validate parameters in the factory. Use `tolerance(target, width)` if you want
the same scale convention as the built-in rules. Once all completions satisfy a threshold, that
term gives them equal scores; add another preference if you want further optimization.
