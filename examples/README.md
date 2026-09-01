# Examples

Short, runnable scripts covering the three things IDiom does. Each takes a `--model` (an HF repo id
or a local directory), so they work against released weights or your own checkpoints.

```bash
uv run python examples/01_generate.py       --model jxliu2/idiom-300M
uv run python examples/02_embeddings.py     --model jxliu2/idiom-300M
uv run python examples/03_sae_features.py   --sae   jxliu2/idiomsae-300M-L18-k32
uv run python examples/04_custom_reward.py                # no GPU / no weights needed
```

A GPU is recommended for 1–3 (they run on CPU, just slowly). Pass `--device cpu` to force CPU.
