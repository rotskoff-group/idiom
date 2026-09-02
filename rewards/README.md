# rewards/ — GRPO reward content

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

This directory holds the **user-editable** reward content. The reward engine itself is library code
(`idiom.train.grpo.reward`).

| File | What it is |
|------|-----------|
| `rl_sae_targets/*.json` | The target SAE feature sets the `rl_sae` reward reads. Build your own with `examples/05_feature_enrichment.py`. |
| `example_rewards.py` | **Copy this** to write a simple in-process reward (`f(idr) -> float`). |
| `scorers/` | **External reward models** that run in their own environment (see below). |

A reward is either **built in** (`entropy`, `length`, `rl_sae`) or **yours** — and yours is either a
**Python function** (in-process) or a **command** (a scorer in its own environment).

## rl_sae — the flagship

```bash
idiom_grpo init_from=/path/base.ckpt reward.rl_sae.enabled=true reward.rl_sae.signature=nucleolus
```

The `rl_sae` reward is built in (`idiom.train.grpo.reward.rl_sae_reward`, imported automatically when
enabled). The `signature` names one of the sets in `rl_sae_targets/idiomsae-300M-L18-k32.json`; point
`IDIOM_SAEREWARD_FEATURES` at your own file (same shape) to use a signature you built. No third-party
dependency — the reward model is IDiom itself.

## external — bring your own reward model

Each `external` entry is one of two things.

### In-process — a Python function

Your reward installs alongside IDiom. Copy `example_rewards.py`, register a function, and name it:

```yaml
reward.external:
  - {enabled: true, weight: 1.0, name: aromatic_fraction, module: rewards/example_rewards.py}
```

### A command — a scorer in its own environment

Your reward model conflicts with IDiom (a different Python, torch, or CUDA). Run it as a subprocess
scorer — the `scorers/` programs below. It returns a raw value; `target`/`width` band it to `(0, 1]`
so it composes with the other terms:

```yaml
reward.external:
  - {enabled: true, weight: 0.5, target: 25, width: 3,
     cmd: "uv run --isolated --no-project --with 'sparrow @ git+https://github.com/idptools/sparrow.git' python rewards/scorers/sparrow.py --property radius_of_gyration"}
```

Because the command lives in the config, several external rewards — each its own environment and
target — combine in one run. Any entry may set `monitor: true` to be logged without being added to
the total.

## scorers/ — the scorer protocol

A **scorer** is a standalone program that scores sequences for a `cmd` external term. It runs in its
own environment, so a reward model whose dependencies conflict with IDiom's works without compromise.

Newline-delimited JSON on stdin/stdout. One request line per GRPO step, one response line back:

```
->  {"sequences": ["ACDEF...", "GHIKL..."]}
<-  {"scores": [24.8, 31.2]}
<-  {"error": "CUDA out of memory"}
```

Rules:

1. **One score per sequence, in the order received.** A mismatch is rejected rather than trusted —
   misaligned rewards would corrupt training silently.
2. **Flush after every response.** Otherwise the parent blocks on a buffered line.
3. **stdout is the protocol.** Logging, warnings and tracebacks go to stderr, which IDiom forwards
   to the job log prefixed with the term name.
4. **Scores are finite numbers.** NaN and infinity are rejected.
5. **Load the model once**, at import, outside the loop. The process is reused for the whole run.
6. **Import nothing from IDiom.** Stdlib plus your own package is the entire contract.

Return the **raw value** your model computes (a radius of gyration, a probability), not a reward.
Keeping `target`/`width` on the IDiom side lets you retune the objective without touching this
environment, and the raw value is what gets logged. Empty and duplicate sequences are filtered out
before the request is sent, so a scorer never needs to handle them for efficiency.

## Setting up the scorer's environment

The environment is per-user and per-machine — never committed. The term's `cmd` names it. Three
idioms, easiest first:

**On demand with uv (no install step) — the default.** uv builds the environment from a package spec
on first use and caches it, so the config is the only thing you write:

```yaml
cmd: "uv run --isolated --no-project --with 'sparrow @ git+https://github.com/idptools/sparrow.git' python rewards/scorers/sparrow.py --property radius_of_gyration"
```

Point uv's cache at scratch, not your home directory (it is several GB). uv reads these from the job
environment — the subprocess inherits them; do not put `VAR=...` inside the `cmd`, which is run
without a shell:

```bash
export UV_CACHE_DIR=/scratch/you/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch/you/uv-python   # only if uv must fetch a Python
```

The environment builds once at the startup handshake (not per step — the scorer process is
persistent), so the first training step waits for the build (~30s for sparrow, which needs a C
compiler on the node) and the rest are cache hits. For a reproducible run, pin to a commit — a bare
git URL re-resolves the branch each fresh build:
`--with 'sparrow @ git+https://github.com/idptools/sparrow.git@03aa232'`.

**A venv you install once (referenced by path).** Preferable when you want a fixed, pre-built
environment; put it on scratch, the `cmd` carries the full interpreter path:

```bash
uv venv /scratch/you/idiom-envs/sparrow --python 3.11
uv pip install --python /scratch/you/idiom-envs/sparrow/bin/python "sparrow @ git+https://github.com/idptools/sparrow.git"
# cmd: "/scratch/you/idiom-envs/sparrow/bin/python rewards/scorers/sparrow.py --property radius_of_gyration"
```

The same `cmd` form also covers a conda env (`conda run -n <env> python ...`), `docker run -i`, and
`apptainer exec --nv`.

## Writing your own scorer

Copy `scorers/example.py` — the whole loop is fifteen lines:

```python
import json, sys
model = load_my_model()                        # once, at import

for line in sys.stdin:
    if not line.strip():
        continue
    try:
        seqs = json.loads(line)["sequences"]
        out = {"scores": [float(x) for x in model.score(seqs)]}
    except Exception as e:
        out = {"error": f"{type(e).__name__}: {e}"}
    print(json.dumps(out), flush=True)
```

Scorer-specific settings ride on the command line (`--property`, `--compartment`), so one config key
is all IDiom needs. Verify before spending a GPU allocation:

```bash
python -m idiom.train.grpo.reward.external_reward --cmd "<your command>" --target 25 --width 3
```

## The scorers here

| Scorer | Environment | Runs from a clean checkout |
|--------|-------------|----------------------------|
| `scorers/example.py` | none (stdlib only) | yes |
| `scorers/sparrow.py` | sparrow from git, on demand via uv | yes |

`sparrow.py` wraps a real external codebase ([idptools/sparrow](https://github.com/idptools/sparrow))
predicting real biophysics (radius of gyration, asphericity, scaling exponent, charge patterning),
and works end to end from a clean checkout — read it first, then `example.py` for the bare template.
