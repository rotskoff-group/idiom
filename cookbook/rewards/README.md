# Rewards

See the [cookbook](../README.md#reward-definition) for details.

## In-process rewards

[custom_rewards_shapings.py](custom_rewards_shapings.py) provides custom reward and reward shaping examples that run in the training process.

## Scorer-based rewards

The adapters in `cookbook/rewards/scorers/` run scorers as subprocesses in separate environments.

| File | Provides |
|---|---|
| [custom_scorer.py](scorers/custom_scorer.py) | Template for a scorer in a separate environment |
| [sparrow.py](scorers/sparrow.py) | [SPARROW](https://github.com/idptools/sparrow): sequence properties, including radius of gyration |
| [finches.py](scorers/finches.py) | [FINCHES](https://github.com/idptools/finches): self- or partner-interaction epsilon |
| [protgps.py](scorers/protgps.py) | [ProtGPS](https://github.com/pgmikhael/protgps): subcellular compartment localization probabilities |
| [paddle.py](scorers/paddle.py) | [PADDLE](https://github.com/asanborn/PADDLE): predicted transcriptional activation strength |
| [starling.py](scorers/starling.py) | [STARLING](https://github.com/idptools/starling/): ensemble radius of gyration or end-to-end distance |
