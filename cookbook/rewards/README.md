# Rewards

The IDiom reinforcement learning implementation uses GRPO to maximize a weighted sum of shaped rewards: `R = Σ weight × shaping(raw reward)`.

<!-- Define at least one term in `reward.terms` -->
<!-- none are added automatically. See [running scripts](../scripts/README.md#running-scripts) for setup. -->


| File | Provides |
|---|---|
| [custom_rewards.py](custom_rewards.py) | Custom reward and reward shaping |
| [custom_scorer.py](scorers/custom_scorer.py) | Template for a scorer in a separate environment |
| [sparrow.py](scorers/sparrow.py) | [SPARROW](https://github.com/idptools/sparrow): sequence properties, including radius of gyration |
| [finches.py](scorers/finches.py) | [FINCHES](https://github.com/idptools/finches): self- or partner-interaction epsilon |
| [protgps.py](scorers/protgps.py) | [ProtGPS](https://github.com/pgmikhael/protgps): subcellular compartment localization probabilities |
| [paddle.py](scorers/paddle.py) | [PADDLE](https://github.com/asanborn/PADDLE): predicted transcriptional activation strength |
| [starling.py](scorers/starling.py) | [STARLING](https://github.com/idptools/starling/): ensemble radius of gyration or end-to-end distance |


<!-- ## Examples

| Script | Reward |
|---|---|
| [sae_features.bash](../scripts/grpo/sae_features.bash) | Reinforcement learning with sparse autoencoder features |
| [custom_reward.bash](../scripts/grpo/custom_reward.bash) | Use a custom Python reward |
| [custom_scorer.bash](../scripts/grpo/custom_scorer.bash) | Run a custom scorer as a subprocess in a separate environment |
| [sparrow.bash](../scripts/grpo/sparrow.bash) | Target an IDR sequence property such as radius of gyration using [SPARROW](https://github.com/idptools/sparrow) |
| [finches.bash](../scripts/grpo/finches.bash) | Match ProTalpha interaction strength with H1.0 CTD using [FINCHES](https://github.com/idptools/finches) |
| [protgps.bash](../scripts/grpo/protgps.bash) | Increase compartment localization probability using [ProtGPS](https://github.com/pgmikhael/protgps) |
| [paddle.bash](../scripts/grpo/paddle.bash) | Increase predicted transcriptional activation strength using [PADDLE](https://github.com/asanborn/PADDLE) |
| [starling.bash](../scripts/grpo/starling.bash) | Target predicted ensemble dimensions using [STARLING](https://github.com/idptools/starling/) |
| [prompted_linker_rg.bash](../scripts/grpo/prompted_linker_rg.bash) | Target linker dimensions within fixed flanks |
| [combined.bash](../scripts/grpo/combined.bash) | Combine FINCHES and ProtGPS | -->

## Configuring terms

This term favors sequences of 100 residues:

```yaml
reward:
  terms:
    - label: length
      reward: length
      shaping: {name: quadratic, target: 100, width: 0.2}
      weight: 1.0
```

Built-in rewards are `length`, `entropy` (composition entropy in bits), `sae_signature`
(fraction of signature features active), and `scorer` (an external program's score).
Factories accept a bare name or a mapping with `name` and arguments.
Shaping defaults to `identity` and weight to `1.0`.

| Shaping | Behavior |
|---|---|
| `identity` | Maximize the raw value with positive weight; minimize with negative weight |
| `quadratic` | 0 at the target, −1 one tolerance away; unbounded below |
| `gaussian` | 1 at the target, approaching 0 far away |

Tolerance is `abs(target) × width`, or `width` when the target is zero. Width must be positive.
Give terms distinct labels. Logs report raw measurements and weighted contributions;
zero-weight terms still run as diagnostics. Calibrate targets and weights on representative
sequences. Length and entropy objectives do not guarantee disorder.

## External scorers

Install `uv` with `python -m pip install uv`. The [scorer adapters](scorers/) install their
own dependencies and download weights on first use. To use one, put this in a term's `reward` field:

```yaml
reward:
  name: scorer
  cmd: "uv run --script cookbook/rewards/scorers/sparrow.py --property radius_of_gyration"
  timeout: 600
```

Scorers persist across batches and cache scores by sequence. Use `cache_max: 0` for stochastic
scorers such as STARLING. GPU scorers need memory alongside the policy; STARLING uses a
separate GPU by default. Keep ProtGPS defaults (`IDIOM_PROTGPS_DEVICE=cpu`, `PROTGPS_BATCH=1`)
for device compatibility and consistent per-sequence scores.

## Custom rewards and shaping

See [custom_rewards.py](custom_rewards.py) for examples. A reward factory returns a function
mapping `list[str]` to one finite score per sequence, in order, including empty sequences.
A shaping factory returns a `float -> float` function. Reference either factory as
`package.module:function` or `/path/to/file.py:function`, with arguments alongside `name`.

For a separate environment, copy [custom_scorer.py](scorers/custom_scorer.py) and
[_scorer_protocol.py](scorers/_scorer_protocol.py) together, then edit the dependency header
and `build()`. Keep `serve(build)`; use stderr for logs. The protocol is one JSON object per line:

```text
→ {"sequences": ["ACDEF", "GHIKL"]}
← {"scores": [24.8, 31.2]}
```

## Design examples

The FINCHES example targets native-like interaction epsilon using the
[reference constructs](../example_data/README.md). Recalculate targets when changing force
field or salt; matching predicted epsilon does not establish equal binding affinity.

The [prompted linker example](../scripts/README.md#prompted-linker-redesign) targets Rg 25 Å,
length 45, and entropy 3.65 bits. Rg describes the isolated linker, not attached domains or
full-protein dimensions. Targets are editable demonstration settings. GRPO logs metapredict
disorder separately from the reward; set `grpo.track_disorder=false` to disable it.
