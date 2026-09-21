# Rewards

See the [cookbook guide](../README.md#reward-definition) for reward configuration and examples.

## Custom reward and shaping

[custom_rewards_shapings.py](custom_rewards_shapings.py) provides custom reward and reward shaping examples that run in the training process.

## External scorers

The adapters in `cookbook/rewards/scorers/` run scorers as subprocesses.
The [GRPO scripts](../scripts/README.md#grpo-examples) launch them in separate
environments with `uv run`; install `uv` with `python -m pip install uv`.

| File | Provides |
|---|---|
| [custom_scorer.py](scorers/custom_scorer.py) | Template for a custom external scorer run as a subprocess in a separate environment, with setup, scoring, communication, and testing instructions |
| [sparrow.py](scorers/sparrow.py) | [SPARROW](https://github.com/idptools/sparrow): IDR sequence properties such as radius of gyration |
| [finches.py](scorers/finches.py) | [FINCHES](https://github.com/idptools/finches): self- or partner-interaction epsilon |
| [protgps.py](scorers/protgps.py) | [ProtGPS](https://github.com/pgmikhael/protgps): compartment localization probabilities |
| [paddle.py](scorers/paddle.py) | [PADDLE](https://github.com/asanborn/PADDLE): predicted transcriptional activation strength |
