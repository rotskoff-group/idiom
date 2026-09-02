# rewards/ — GRPO reward definitions

The GRPO reward is a **weighted sum of terms** (`configs/grpo.yaml`, the `reward` block):

```yaml
reward:
  entropy: {enabled: true,  weight: 1.0, target_entropy: 3.68, width: 0.2}   # naturalness guardrail
  length:  {enabled: true,  weight: 1.0, target_length: 100, width: 1.0}
  rl_sae:  {enabled: false, weight: 1.0, signature: nucleolus}               # reproduce an SAE feature code
  external: []                                                               # bring your own (below)
```

`total = Σ weightᵢ · termᵢ`. Toggle a term with `enabled`, scale it with `weight`. That total is
what GRPO turns into advantages. `entropy` and `length` are built in; the two interesting slots are
`rl_sae` and `external`.

## The files

| File | What it is |
|------|-----------|
| `rl_sae.py` | The **RL-SAE reward**. Registers `sae_only_<signature>`; enabled by the `rl_sae` block. |
| `signatures/*.json` | The target SAE feature sets `rl_sae.py` reads. Build your own with `examples/05_feature_enrichment.py`. |
| `builtin_rewards.py` | **Copy this** to write a simple in-process reward (`f(idr) -> float`). |
| `scorers/` | **External reward models** that run in their own environment. See [`scorers/README.md`](scorers/README.md). |

That is the whole taxonomy: a reward is either **built in** (`entropy`, `length`, `rl_sae`) or
**yours** — and yours is either a **Python function** (in-process) or a **command** (its own
environment). Nothing here is old or superseded.

## rl_sae — the flagship

```bash
idiom_grpo init_from=/path/base.ckpt reward.rl_sae.enabled=true reward.rl_sae.signature=nucleolus
```

`rl_sae.py` is imported automatically. The `signature` names one of the sets in
`signatures/idiomsae-300M-L18-k32.json`; point `IDIOM_SAEREWARD_FEATURES` at your own file (same
shape) to use a signature you built. No third-party dependency — the reward model is IDiom itself.

## external — bring your own reward model

Each `external` entry is one of two things.

### In-process — a Python function

Your reward installs alongside IDiom. Copy `builtin_rewards.py`, register a function, and name it:

```yaml
reward.external:
  - {enabled: true, weight: 1.0, name: aromatic_fraction, module: rewards/builtin_rewards.py}
```

### A command — its own environment

Your reward model conflicts with IDiom (a different python, torch, or CUDA). Run it as a subprocess
in its own environment — with no install step, let uv build and cache the environment on demand:

```yaml
reward.external:
  - {enabled: true, weight: 0.5, target: 25, width: 3,
     cmd: "uv run --isolated --no-project --with 'sparrow @ git+https://github.com/idptools/sparrow.git' python rewards/scorers/sparrow.py --property radius_of_gyration"}
```

The scorer returns a raw value (a radius of gyration in angstroms); `target`/`width` band it to
`(0, 1]` so it composes with the other terms. Because the command lives in the config, several
external rewards — each its own environment and target — combine in one run. Point uv's cache at
scratch (`export UV_CACHE_DIR=/scratch/you/uv-cache`) and verify a command before launching:

```bash
python -m idiom.train.grpo.external \
  --cmd "uv run --isolated --no-project --with 'sparrow @ git+https://github.com/idptools/sparrow.git' python rewards/scorers/sparrow.py --property radius_of_gyration" \
  --target 25 --width 3
```

Any entry may set `monitor: true` to be logged without being added to the total. The worked scorer is
**sparrow** (biophysics — Rg, asphericity, charge patterning; runs from a clean checkout); it and the
protocol for writing your own are in [`scorers/README.md`](scorers/README.md).
