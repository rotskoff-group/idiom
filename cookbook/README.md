# IDiom Cookbook

Here we provide examples of how to use IDiom!

## Table of Contents

- [Getting started](#getting-started)
- [Notebooks](#notebooks)
- [Bash scripts](#bash-scripts)
- [Reinforcement learning](#reinforcement-learning)
- [Reward definition](#reward-definition)
- [Example data](#example-data)


## Getting started

For local use, first install IDiom and clone the repository:

```bash
pip install "idiom @ git+https://github.com/rotskoff-group/idiom.git@v1"
git clone --branch v1 https://github.com/rotskoff-group/idiom.git
cd idiom
```

Demo inputs are included in `cookbook/example_data`; see [Example data](#example-data) for details.

### Notebooks

Run the [notebooks](#notebooks) interactively in Colab or locally to explore IDR generation and prediction, embeddings, SAE analysis, and post-training.

### Bash scripts and reinforcement learning examples

The [Bash scripts](#bash-scripts) cover training, inference, and SAE workflows, including [reinforcement learning examples](#grpo-examples) in `cookbook/scripts/training/grpo/`. To run a script, activate your IDiom-installed environment, edit the `REPO` and `OUT` directories, then run a script from the repository root:

```bash
bash cookbook/scripts/generation/generate_unprompted.bash
```

Adjust the default parameters for your use. Most training and inference examples use one GPU.

### Custom RL rewards

Use the examples in [`cookbook/rewards/`](rewards/README.md) to define custom reinforcement learning rewards or connect external scorers. Configure reward terms in the matching GRPO YAML file. See [Reward definition](#reward-definition) for built-in rewards, custom functions, and scorer setup.

<br>

## Notebooks

Run any notebook independently in Colab or locally, using example data or your own FASTA.

| Notebook | Colab |
|---|---|
| [Generate IDRs](notebooks/generate_idrs.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/generate_idrs.ipynb) |
| [Predict IDRs and generate replacements](notebooks/predict_idrs.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/predict_idrs.ipynb) |
| [Extract embeddings](notebooks/extract_embeddings.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/extract_embeddings.ipynb) |
| [Interpret SAE features](notebooks/interpret_sae_features.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/interpret_sae_features.ipynb) |
| [Enriched feature signature](notebooks/enriched_feature_signature.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/enriched_feature_signature.ipynb) |
| [SFT and generate](notebooks/sft_and_generate.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/sft_and_generate.ipynb) |
| [RL with custom rewards](notebooks/rl_with_custom_rewards.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/rl_with_custom_rewards.ipynb) |
| [RL with SAE rewards](notebooks/rl_with_sae_rewards.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/rl_with_sae_rewards.ipynb) |

Run cells in order. Colab notebooks will install dependencies and download example data.
For local use, install `idiom`. Input and parameter details are in each notebook.
A GPU is recommended for SAE analysis, fine-tuning, and RL.
Download results from Colab before the runtime ends.

<br>

## Bash scripts

In `cookbook/scripts/`, we provide the following Bash scripts. Details are provided within the scripts themselves.

### IDR generation

| Script | Task |
|---|---|
| [generate_unprompted.bash](scripts/generation/generate_unprompted.bash) | Generate unprompted IDRs |
| [generate_prompted.bash](scripts/generation/generate_prompted.bash) | Generate IDRs prompted with provided flanking contexts |

### SAE analysis

| Script | Task |
|---|---|
| [build_feature_dataset.bash](scripts/sae/build_feature_dataset.bash) | Extract per-residue SAE features from sequences |
| [feature_enrichment.bash](scripts/sae/feature_enrichment.bash) | Find enriched features and export top features |
| [train_sae.bash](scripts/sae/train_sae.bash) | Train and export an SAE |

### Training

| Script | Task |
|---|---|
| [sft.bash](scripts/training/sft.bash) | Supervised fine-tuning on a set of sequences |
| [pretrain.bash](scripts/training/pretrain.bash) | Pretrain IDiom from scratch |
| [cookbook/scripts/training/grpo/](scripts/training/grpo/) | Reinforcement learning with custom rewards, SAE features, or external scorers |

`pretrain.bash`, `sft.bash`, `train_sae.bash`, and all scripts in `cookbook/scripts/training/grpo/` load default YAML configs from `src/idiom/configs/` and apply Hydra command-line overrides.

`generate_unprompted.bash`, `generate_prompted.bash`, `build_feature_dataset.bash`, and `feature_enrichment.bash` use command-line arguments without YAML configs.


<br>

## Reinforcement learning

IDiom uses GRPO for reinforcement learning post-training. RL aims to maximize a weighted sum of shaped rewards:

$$
R(x) = \sum_i \text{weight}_i \times \text{shaping}_i\!\left(\text{raw reward}_i(x)\right)
$$

where $x$ is the generated IDR sequence and $i$ indexes the reward terms. Reward functions receive **only the generated IDR sequence.**

Rewards can be calculated in three ways:

- **Built-in rewards:** use the reward functions provided by IDiom.
- **Custom Python rewards:** run your own functions in the training process when their dependencies are compatible with the training environment.
- **External scorers:** run scoring code in a separate Python subprocess, with its own environment.

At least one reward must be enabled for training, and multiple rewards can be combined. All rewards can be numerically shaped before they are added to the overall reward. 

In this cookbook, we provide several packaged examples for running GRPO training in `cookbook/scripts/training/grpo/`.

### GRPO examples

| Script | Reward |
|---|---|
| [sae_features.bash](scripts/training/grpo/sae_features.bash) | Reinforcement learning with sparse autoencoder features (RL-SAE) |
| [custom_reward.bash](scripts/training/grpo/custom_reward.bash) | Use a custom Python reward |
| [custom_scorer.bash](scripts/training/grpo/custom_scorer.bash) | Run a custom external scorer as a subprocess in a separate environment |
| [sparrow.bash](scripts/training/grpo/sparrow.bash) | Target an IDR sequence property such as radius of gyration using [SPARROW](https://github.com/idptools/sparrow) |
| [finches.bash](scripts/training/grpo/finches.bash) | Match ProTalpha interaction strength with H1.0 CTD using [FINCHES](https://github.com/idptools/finches) |
| [protgps.bash](scripts/training/grpo/protgps.bash) | Increase compartment localization probability using [ProtGPS](https://github.com/pgmikhael/protgps) |
| [paddle.bash](scripts/training/grpo/paddle.bash) | Increase predicted transcriptional activation strength using [PADDLE](https://github.com/asanborn/PADDLE) |
| [prompted_linker_rg.bash](scripts/training/grpo/prompted_linker_rg.bash) | Target linker dimensions within fixed flanks |
| [combined.bash](scripts/training/grpo/combined.bash) | Combine FINCHES and ProtGPS |

<br>

## Reward definition

GRPO maximizes the sum of weighted, shaped reward terms for each sequence:

```text
total reward = sum(weight × shaping(raw reward))
```

Each script in `cookbook/scripts/training/grpo/` loads `reward.terms` from its matching
YAML file (for example, `custom_reward.bash` loads `custom_reward.yaml`). 


<!-- Edit rewards, -->
<!-- scorer commands, shaping, and weights in YAML. Edit training settings and runtime paths -->
<!-- in Bash. No Hydra `defaults` section is needed in the reward YAML. -->

<br>

### Configuring reward terms

Reward terms have four fields: 

| Field | Choices |
|---|---|
| `label` | Any unique name, used in logs. |
| `weight` | Any finite number, defaults to `1.0`. Zero logs the term without optimizing it, and negative values reverse its contribution. |
| `reward` | A mapping with `name` and any arguments: a built-in (`length`, `entropy`, or `sae_signature`), a custom Python factory (returns a function), or an external program via `external_scorer`. |
| `shaping` | A mapping with `name` and any arguments: a built-in (`identity`, `quadratic`, or `gaussian`) or a custom Python factory, defaults to `identity`. |

Example terms are provided below. 

<br>

### Write and run your own reward

Use an in-process custom reward when its dependencies fit your training environment, and use an external scorer when it needs a separate environment.

#### In-process reward

1. Define your reward using [custom_rewards_shapings.py](rewards/custom_rewards_shapings.py) as a template. The reward function takes a list of IDRs and returns one finite score per IDR, in the same order.
2. In [custom_reward.yaml](scripts/training/grpo/custom_reward.yaml), set `reward.name` to `/path/to/file.py:factory` or `package.module:factory`. The factory creates your reward function, and you must add its arguments under `reward`, alongside `name`. Set the shaping function and weight.
3. Test your reward on a few IDRs.
4. Set `REPO`, `OUT`, and training parameters in [custom_reward.bash](scripts/training/grpo/custom_reward.bash), then run from the repository root:

```bash
bash cookbook/scripts/training/grpo/custom_reward.bash
```

#### External scorer

1. Copy [custom_scorer.py](rewards/scorers/custom_scorer.py) and [_scorer_protocol.py](rewards/scorers/_scorer_protocol.py) into the same directory. Update the dependency header and scoring function.
2. Edit [custom_scorer.yaml](scripts/training/grpo/custom_scorer.yaml). Point `reward.cmd` to your script and choose shaping and weight. Review the included entropy and length terms too, since they also contribute to training.
3. Test the scorer before training, replacing the path below with your script's location.

```bash
python -m idiom.train.grpo.reward.external \
  --cmd "uv run --script /path/to/my_scorer.py"
```

4. Set `REPO`, `OUT`, and training settings in [custom_scorer.bash](scripts/training/grpo/custom_scorer.bash), then run it from the repository root.

```bash
bash cookbook/scripts/training/grpo/custom_scorer.bash
```



<br>

## Example reward terms 

The following examples show reward terms you can use or adapt.

### Built-in reward and shaping

Built-in rewards are `length` (residue count), `entropy` (composition entropy in bits),
and `sae_signature` (fraction of signature features active).

Built-in shaping functions are `identity` (use the raw score), `quadratic` (0 at the target,
−1 one tolerance-width away), and `gaussian` (1 at the target, approaching 0 away from it).
For `quadratic` and `gaussian`, tolerance is `abs(target) × width`, or `width` when the
target is zero, and width must be positive.

See [builtin.py](../src/idiom/train/grpo/reward/builtin.py) for length and entropy,
[sae_feature.py](../src/idiom/train/grpo/reward/sae_feature.py) for SAE signatures, and
[shaping.py](../src/idiom/train/grpo/reward/shaping.py) for built-in shaping functions.

For example, this term targets lengths of 100 residues, scoring 0 at 100 and −1 at 80 or 120:

```yaml
reward:
  terms:
    - label: length
      reward:
        name: length
      shaping:
        name: quadratic
        target: 100
        width: 0.2
      weight: 1.0
```

<br>

### Custom reward with built-in shaping

The `fraction_charged` reward factory (returns a function) in [custom_rewards_shapings.py](rewards/custom_rewards_shapings.py) measures
the fraction of D/E/K/R residues. Gaussian shaping targets a charged fraction of 0.30:

```yaml
reward:
  terms:
    - label: charged_fraction
      reward:
        name: cookbook/rewards/custom_rewards_shapings.py:fraction_charged
      shaping:
        name: gaussian
        target: 0.3
        width: 0.5
      weight: 1.0
```

To add your own reward, define a factory in your Python file and reference it as
`/path/to/file.py:factory` or `package.module:factory`. It must return a function that
produces one finite score per input sequence, in order, including empty sequences.

More information is provided in [custom_rewards_shapings.py](rewards/custom_rewards_shapings.py).

<br>

### Custom reward and shaping

This reward uses an example custom reward shaping, `absolute_error` from [custom_rewards_shapings.py](rewards/custom_rewards_shapings.py).
It returns `−abs(value − target)`, penalizing distance from the target linearly. For a
target of 0.30, the score is 0 at 0.30 and −0.15 at either 0.15 or 0.45:

```yaml
reward:
  terms:
    - label: charged_fraction
      reward:
        name: cookbook/rewards/custom_rewards_shapings.py:fraction_charged
      shaping:
        name: cookbook/rewards/custom_rewards_shapings.py:absolute_error
        target: 0.30
      weight: 1.0
```

Information on how to add your own custom reward shaping is also provided in [custom_rewards_shapings.py](rewards/custom_rewards_shapings.py).

<br>

### External scorer with built-in shaping

`reward.name: external_scorer` tells IDiom to run the command-line program specified by
`cmd`, send it generated sequences, and read back scored values. The provided [finches.py](rewards/scorers/finches.py)
scorer calculates interaction epsilon. This example uses homotypic (self-interaction)
scoring with Mpipi and a quadratic penalty toward epsilon −6:

```yaml
reward:
  terms:
    - label: epsilon
      reward:
        name: external_scorer
        cmd: "uv run --script cookbook/rewards/scorers/finches.py --mode homotypic --forcefield mpipi"
        timeout: 300
      shaping:
        name: quadratic
        target: -6.0
        width: 1.0
      weight: 1.0
```

Other provided scorers include SPARROW, ProtGPS, and PADDLE in
`cookbook/rewards/scorers/`. To add your own scorer, adapt [custom_scorer.py](rewards/scorers/custom_scorer.py), which documents setup, scoring, communication, and testing. See [custom_scorer.yaml](scripts/training/grpo/custom_scorer.yaml) for reward configuration and [custom_scorer.bash](scripts/training/grpo/custom_scorer.bash) for running training.

<br>

### Combining multiple reward terms

Add multiple entries to `reward.terms` to combine objectives. This example targets composition
entropy of 3.65 bits, length of 100 residues, and SPARROW-predicted radius of gyration of 25 Å:

```yaml
reward:
  terms:
    - label: entropy
      reward:
        name: entropy
      shaping:
        name: quadratic
        target: 3.65
        width: 0.2
      weight: 1.0
    - label: length
      reward:
        name: length
      shaping:
        name: quadratic
        target: 100
        width: 1.0
      weight: 1.0
    - label: radius_of_gyration
      reward:
        name: external_scorer
        cmd: "uv run --script cookbook/rewards/scorers/sparrow.py --property radius_of_gyration"
        timeout: 300
      shaping:
        name: quadratic
        target: 25
        width: 0.2
      weight: 1.0
```

<br>

## Example data

Demo inputs are included in `cookbook/example_data`.

| Directory | Contents |
|---|---|
| `cookbook/example_data/protgps/` | IDRs associated with six subcellular compartments |
| `cookbook/example_data/effector/` | Experimentally measured activation and repression domain IDRs |
| `cookbook/example_data/disprot/` | Held-out proteins with annotated IDRs and flanking context |
| [`cookbook/example_data/sae_features/`](example_data/sae_features/) | Enriched top-30 and original private-30 SAE targets |
| `cookbook/example_data/prompted_grpo/` | HP1α (P45973), IDR residues 79–123 |
| `cookbook/example_data/finches/` | ProTalpha and H1.0 C-terminal reference constructs |

FASTA headers end with `_IDR_x-y`, with 1-based, inclusive coordinates, to mark an IDR span.
ProtGPS and effector sequences are isolated IDRs with no flanking context. DisProt sequences include flanks and may repeat accessions with different spans. Prompted IDR generation uses the flanks of a full protein around the marked IDR span as the prompt, and the HP1α sequence is used as an example of this.

See the main README's [sequence conventions](../README.md#sequence-conventions) for more information.

[`sae_signatures.json`](example_data/sae_features/sae_signatures.json) contains two sets of SAE feature targets for reinforcement learning with sparse autoencoder features (RL-SAE):

- `top30`: enriched features for each compartment
- `private30`: top private features for each compartment

Both sets cover the main biomolecular compartments discussed in the paper. In [`sae_features.yaml`](scripts/training/grpo/sae_features.yaml), choose a compartment with `signature` (e.g. `nucleolus`) and a feature set with `case` (`top30` or `private30`).
