# IDR-PLM

![IDR-PLM](assets/github_fig.png)

## Table of contents
- <u>[Environment setup](#environment-setup)</u>
- <u>[Model checkpoints and data](#model-checkpoints-and-data)</u>
  - <u>[Model architecture](#model-architecture)</u>
- <u>[Workflows](#workflows)</u>
  - <u>[1. Generating unprompted IDPs](#1-generating-unprompted-idps)</u>
  - <u>[2. Generating context-conditioned IDRs](#2-generating-context-conditioned-idrs)</u>
  - <u>[3. GRPO post-training for IDPs with a custom reward](#3-grpo-post-training-for-idps-with-a-custom-reward)</u>
  - <u>[4. GRPO post-training for context-conditioned IDRs with a custom reward](#4-grpo-post-training-for-context-conditioned-idrs-with-a-custom-reward)</u>
  - <u>[5. GRPO post-training for IDPs with the ProtGPS reward](#5-grpo-post-training-for-idps-with-the-protgps-reward)</u>
  - <u>[6. GRPO post-training for context-conditioned IDRs with the ProtGPS reward](#6-grpo-post-training-for-context-conditioned-idrs-with-the-protgps-reward)</u>
  - <u>[7. Generating sequences after post-training](#7-generating-sequences-after-post-training)</u>
  - <u>[8. Pre-training from scratch](#8-pre-training-from-scratch)</u>

IDR-PLM is an autoregressive transformer trained on 37M intrinsically disordered regions from the AlphaFold Database. The model can generate intrinsically disordered proteins (IDPs) as well as intrinsically disordered regions (IDRs) conditioned on their flanking context. The model can be post-trained with reinforcement learning to optimize for user-defined rewards. Paper can be found at: bioarxiv link 


## Environment setup

First, clone the repository to a directory with at least 30 GB of free space (the space is necessary for model checkpoints). Then, `uv sync` the repository and install it in editable mode.


```bash
git clone https://github.com/rotskoff-group/idr-plm.git
cd idr-plm
uv sync
uv pip install -e .
```

Requires Python ≥ 3.10 and PyTorch 2.4.0.

---

## Model checkpoints and data

Next, download the pre- and post-trained `IDR-PLM` model checkpoints from the HuggingFace repository and move them into the `idr-plm/` project root directory. 

**Model checkpoints**: https://huggingface.co/jxliu2/idr-plm

You can do so with the following commands. First, make an hf account. From the root of the cloned idr-plm directory, do: 

```bash
uv tool install hf # Install the hf cli tool 
hf auth login # Enter your credentials 

# Download models (26 GB)
hf download jxliu2/idr-plm --local-dir ./models
```



Additional datasets which not necessary for running the code repository can be found in the following HuggingFace repository. This includes the 37M IDRs used to pre-train `IDR-PLM` as well as the generated sequences which we analyze in the paper. (However, this data is necessary if you would like to replicate pre-training.)

**Datasets**: https://huggingface.co/datasets/jxliu2/idr-plm-dataset 

```bash 
# Optionally download the IDR data (174 GB): 
hf download jxliu2/idr-plm-dataset --repo-type=dataset --local-dir ./datasets
```


After this, the project structure should be:

```
idr-plm/
|
├── src/                       # Main Python package
│   └── idr_plm/
│       ├── nn/                # Model architecture
│       ├── scripts/           # CLI entry points and Hydra configs
│       └── utils/             # Shared utilities
|
├── entrypoints/               # Scripts for training and inference
│   ├── infer/                 # Inference scripts and output
│   ├── precompute/            # Data preprocessing scripts
│   └── train/                 # Pre- and post-training scripts
|
├── rewards/                   # Reward functions and models
│   ├── custom_rewards/        # User-defined reward functions
│   └── protgps/               # ProtGPS localization reward model
|
├── models/                    # Model checkpoints
|
└── datasets/                  # Datasets (optional)
```


---

### Model architecture

The core model is `GeometricMolTransformer`, a 12-layer causal transformer with:

| Hyperparameter | Value |
|---|---|
| Model dimension | 896 |
| Transformer layers | 12 |
| Attention heads | 14 |
| FFN type | SwiGLU (2.667× expansion) |
| Vocabulary size | 20 amino acids, 3 location tokens, 4 special tokens |
| Positional encoding | Rotary (RoPE) |
| Attention mask | Causal |

---

## Workflows

Each workflow script can be run with either `bash` (directly) or `sbatch` (via SLURM). When using `sbatch`, first create the output directory with `mkdir -p ./slurm_out`. All scripts automatically detect the execution context and set the repository root accordingly.

### 1. Generating unprompted IDPs

```bash
bash entrypoints/infer/scripts/generate_idps.bash
# or:
sbatch entrypoints/infer/scripts/generate_idps.bash
```

Builds IDP prompts (no flanking context) and runs autoregressive inference. Edit `--num_duplicates` to control how many sequences are generated (default 1000) and `CKPT_PATH` to use a different checkpoint.

**Output** (in `entrypoints/infer/output/idps/`):
- `tst_autoregressive.pkl` — raw generated token sequences
- `generated_idrs.fasta` — IDR regions only (same thing for IDPs)
- `generated_full.fasta` — full sequences with IDR coordinates in the header
- `inference_config.yaml` — inference config 

---

### 2. Generating context-conditioned IDRs

`entrypoints/infer/scripts/generate_idrs.bash` generates IDRs conditioned on flanking protein sequences (fill-in-the-middle / FIM mode).

**Input format.** Provide a FASTA file at `entrypoints/infer/scripts/example_sequences.fasta` where each entry has the header `>ACCESSION_IDR_START-END` and the sequence is the full-length protein. The prefix and suffix flanking the IDR region are extracted automatically. For example:

```
>P06748_IDR_119-242
MEDSMDMDMSPLRPQNYLFGCELKADKDYHFKVDN...
>P09651_IDR_186-372
MSKSESPKEPEQLRKLFIGGLSFETTDESL...
```

```bash
bash entrypoints/infer/scripts/generate_idrs.bash
# or:
sbatch entrypoints/infer/scripts/generate_idrs.bash
```

Builds FIM prompts from the FASTA and runs autoregressive inference, generating `--num_duplicates` sequences per protein in the FASTA (default 1000 each). Edit `--num_duplicates` to control this and `CKPT_PATH` to use a different checkpoint.

**Output** (in `entrypoints/infer/output/idrs/`):
- `tst_autoregressive.pkl` — raw generated token sequences
- `generated_idrs.fasta` — IDR regions only (extracted from between sentinel tokens)
- `generated_full.fasta` — full sequences with IDR coordinates in the header
- `inference_config.yaml` — inference config

---


### 3. GRPO post-training for IDPs with a custom reward

```bash
bash entrypoints/train/post-train/train_rl_idp_custom.bash
```

Creates an IDP RL dataset and runs GRPO training with a user-defined reward. Edit `reward_function_name` in the script to point to your function (default: `compute_fraction_proline`, a worked example in `rewards/custom_rewards/custom_rewards.py`) and set `REWARD_TARGET_VALUE` to an appropriate target. Your reward function must accept `(tokens, token_info, device)` and return a scalar `torch.Tensor` on `device`.

> **Out-of-memory errors during training.** If you encounter GPU OOM errors, reduce `BATCH_SIZE` and increase `ACCUMULATE_GRAD_BATCHES` by the same factor to keep the effective batch size constant (e.g. `BATCH_SIZE=2, ACCUMULATE_GRAD_BATCHES=4` → `BATCH_SIZE=1, ACCUMULATE_GRAD_BATCHES=8`). This applies to all post-training workflows. 


---

### 4. GRPO post-training for context-conditioned IDRs with a custom reward

**Input format.** Provide a FASTA file at `entrypoints/train/post-train/rl_sequence.fasta` using the same `>ACCESSION_IDR_START-END` header format as the inference workflows.

```bash
bash entrypoints/train/post-train/train_rl_idr_custom.bash
```

Creates an IDR RL dataset from the FASTA and runs GRPO training with a user-defined reward. Identical to workflow 3 except prompts include flanking sequence context. Edit `reward_function_name` and `REWARD_TARGET_VALUE` to match your function.

---

### 5. GRPO post-training for IDPs with the ProtGPS reward

```bash
bash entrypoints/train/post-train/train_rl_idp_protgps.bash
# or:
sbatch entrypoints/train/post-train/train_rl_idp_protgps.bash
```

Creates an IDP RL dataset and runs GRPO training using the ProtGPS subcellular localization classifier as the reward. Edit `COMPARTMENT` to select the target compartment (default: `stress_granule`) and adjust reward, length, and entropy parameters as needed. Available compartments: `nuclear_speckle`, `p-body`, `pml-body`, `post_synaptic_density`, `stress_granule`, `chromosome`, `nucleolus`, `nuclear_pore_complex`, `cajal_body`, `rna_granule`, `cell_junction`, `transcriptional`. Checkpoints are saved every 500 steps.

---

### 6. GRPO post-training for context-conditioned IDRs with the ProtGPS reward

**Input format.** Provide a FASTA file at `entrypoints/train/post-train/rl_sequence.fasta` using the same `>ACCESSION_IDR_START-END` header format as the inference workflows.

```bash
bash entrypoints/train/post-train/train_rl_idr_protgps.bash
# or:
sbatch entrypoints/train/post-train/train_rl_idr_protgps.bash
```

Creates an IDR RL dataset from the FASTA and runs GRPO training with the ProtGPS reward. Identical to workflow 5 except prompts include flanking sequence context. Default compartment is `nucleolus`.

---

### 7. Generating sequences after post-training

```bash
bash entrypoints/infer/scripts/generate_idps.bash  # or generate_idrs.bash
```

Identical to workflows 1 and 2. Set `CKPT_PATH` in the script to your post-trained checkpoint before running.

---

### 8. Pre-training from scratch

```bash
bash entrypoints/precompute/split_parts.bash        # split raw HDF5 into parts
sbatch entrypoints/precompute/precompute.bash       # tokenize each part (SLURM array)
sbatch entrypoints/train/pre-train/pretrain.bash    # train for 250k steps on 8 GPUs
```

Download `AFDB_IDR_90_FIM_512.h5` from HuggingFace first. The split script divides it into ~500 parts for parallel precomputation; each resulting shard contains tokenized inputs and targets. Pre-training runs for 250,000 steps with a linear warmup cosine annealing schedule; the best validation checkpoint is saved as `best_model_step_<N>.ckpt`.

<!-- ---

## Hydra parameters: inference (`transformer_infer`)

All inference scripts invoke `transformer_infer` with Hydra overrides. Parameters fall into four groups.

### Model parameters (`model.*`)

| Parameter | Value | Description |
|---|---|---|
| `model.model` | `GeometricMolTransformer` | Model class |
| `model.model_args.d_model` | `896` | Hidden dimension |
| `model.model_args.unified_transformer_args.n_layers` | `12` | Number of transformer layers |
| `model.model_args.unified_transformer_args.mha_args.num_heads` | `14` | Attention heads |
| `model.model_args.unified_transformer_args.mha_args.mask_mode` | `causal` | Attention mask type |
| `model.model_args.unified_transformer_args.mha_layer_indices` | `[0,...,11]` | Which layers use MHA |

These must match the checkpoint being loaded and should not be changed for the provided pretrained models.

### Inference parameters (`inference.*`)

| Parameter | Default | Description |
|---|---|---|
| `inference.checkpoint_path` | (required) | Path to the `.ckpt` file to load |
| `inference.savedir` | (required) | Directory where output files are written |
| `inference.inference_mode` | `autoregressive` | Generation mode |
| `inference.batch_size` | `100` | Sequences generated per batch |
| `inference.use_multi_gpu` | `True` | Distribute inference across available GPUs |
| `inference.dataset_filename` | (required) | Path to reference HDF5 shard (used for tokenizer metadata) |
| `inference.sampler_args.method` | `full` | Sampling method (`full` = unrestricted autoregressive) |
| `inference.sampler_args.sample_val` | `1` | Number of samples per prompt position |
| `inference.sampler_args.temperature` | `1.0` | Sampling temperature (higher = more diverse) |

### Prompt parameters (`inference.addn_args.*`)

| Parameter | Default | Description |
|---|---|---|
| `inference.addn_args.use_input_residues` | `True` | Use a pre-built prompt array instead of sampling from the shard |
| `inference.addn_args.residues_path` | (required) | Path to the `.pkl` prompt array file produced by `make_infer_prompt` |

The prompt file is produced by `make_infer_prompt` in either `idp` mode (sentinel-only prompts) or `idr` mode (FIM prompts with flanking sequences from a FASTA). The `--num_duplicates` argument controls how many copies of each prompt are included, and thus the total number of sequences generated.

### Output files

The following files are written to `inference.savedir` after a completed run:

| File | Contents |
|---|---|
| `tst_autoregressive.pkl` | Raw token sequences as a list of integer arrays |
| `generated_idrs.fasta` | IDR regions only, extracted from between sentinel tokens |
| `generated_full.fasta` | Full sequences with IDR start/end coordinates in the FASTA header |

---

## Hydra parameters: GRPO post-training (`transformer_train` with `training_mode=grpo`)

All RL scripts invoke `transformer_train` with `training.training_mode=grpo`. Parameters fall into five groups.

### Data parameters (`data.*`)

| Parameter | Value | Description |
|---|---|---|
| `data.dataset` | `TransformerOnlineDataset` | Dataset class for GRPO (generates completions online) |
| `data.collate_fn` | `transformer_online_collate_fn` | Collate function for online generation |
| `data.dataset_filename` | (required) | Path to the HDF5 RL dataset created by `make_rl_dataset` |
| `data.dloader_args.batch_size` | `4` (IDP) / `2` (IDR) | Prompts per gradient step (before accumulation) |
| `data.data_in_memory` | `False` | Load dataset into RAM (False = stream from disk) |

The RL dataset is created by `make_rl_dataset`. For IDP mode (`make_rl_dataset idp`), no FASTA is needed. For IDR mode (`make_rl_dataset idr`), provide `--fasta` pointing to a file with `>ACCESSION_IDR_START-END` headers.

### Model parameters (`model.*`)

Same as inference. The loaded checkpoint is specified via `model.load_model` (not `inference.checkpoint_path`):

| Parameter | Value | Description |
|---|---|---|
| `model.load_model` | (required) | Path to starting checkpoint (typically the pretrained base model) |

### Trainer parameters (`training.trainer_args.*`)

| Parameter | Default | Description |
|---|---|---|
| `training.trainer_args.max_steps` | `1500` | Total training steps |
| `training.trainer_args.devices` | `1` | Number of GPUs |
| `training.trainer_args.accumulate_grad_batches` | `2` (IDP) / `4` (IDR) | Gradient accumulation |
| `training.trainer_args.strategy` | `ddp_find_unused_parameters_true` | DDP strategy |
| `training.trainer_args.limit_val_batches` | `0.0` | Disable validation during RL |
| `training.trainer_args.gradient_clip_val` | `null` | Gradient clipping (disabled) |
| `training.trainer_args.log_every_n_steps` | `1` | Logging frequency |

Effective batch size = `batch_size × accumulate_grad_batches × group_size`. For IDP scripts this is `4 × 2 × 8 = 64` completions per update; for IDR scripts `2 × 4 × 8 = 64`.

### Optimizer and scheduler parameters (`training.lightning_model_args.*`)

| Parameter | Default | Description |
|---|---|---|
| `training.lightning_model_args.optimizer_args.lr` | `5e-6` | AdamW learning rate |
| `training.lightning_model_args.lr_scheduler` | `null` | No LR schedule during RL |
| `training.lightning_model_args.every_epoch_checkpoint_args.every_n_train_steps` | `500` | Save checkpoint every N steps |

### GRPO algorithm parameters (`training.lightning_model_args.*`)

| Parameter | Default | Description |
|---|---|---|
| `training.lightning_model_args.group_size` | `8` | Completions generated per prompt |
| `training.lightning_model_args.epsilon_clip` | `0.2` | PPO-style clipping threshold |
| `training.lightning_model_args.beta_kl` | `0.02` | KL divergence penalty weight against reference model |
| `training.lightning_model_args.mu_grpo` | `1` | GRPO loss scale |
| `training.lightning_model_args.normalize_advantage` | `True` | Normalize advantages within each group |
| `training.lightning_model_args.sampler_args.method` | `full` | Autoregressive sampling method |
| `training.lightning_model_args.sampler_args.temperature` | `1` | Sampling temperature during rollout |
| `training.lightning_model_args.sampler_args.token_limit` | `1000` | Maximum tokens generated per completion |

### Reward parameters (`training.lightning_model_args.*`)

The total reward combines up to three components: a task reward, a length reward, and an entropy reward.

**Task reward:**

| Parameter | Default | Description |
|---|---|---|
| `training.lightning_model_args.reward_function_name` | (required) | Name of the reward function (e.g. `compute_protgps_score`, `compute_fraction_proline`) |
| `training.lightning_model_args.use_reward_shaping` | `True` | Apply quadratic shaping toward `reward_target_value` |
| `training.lightning_model_args.reward_target_value` | `0.9` (ProtGPS) / `0.2` (custom) | Target value for reward shaping |
| `training.lightning_model_args.reward_scale` | `1` | Scalar multiplier on the task reward |

**ProtGPS-specific parameters** (only used when `reward_function_name=compute_protgps_score`):

| Parameter | Default | Description |
|---|---|---|
| `training.lightning_model_args.protgps_target_compartment` | (required) | Target compartment name (e.g. `stress_granule`) |
| `training.lightning_model_args.protgps_aggregation` | (same as compartment) | Aggregation key for ProtGPS scores |
| `training.lightning_model_args.protgps_parent_dir` | `models/protgps` | Directory containing ProtGPS model weights |

**Custom reward:** When `reward_function_name` is set to a custom function, `rewards/custom_rewards/` must be on `PYTHONPATH` so that `custom_rewards.py` is importable. The function must accept `(tokens, token_info, device)` and return a scalar `torch.Tensor`.

**Length reward:**

| Parameter | Default | Description |
|---|---|---|
| `training.lightning_model_args.use_target_length` | `True` | Enable length reward |
| `training.lightning_model_args.target_length` | `100` | Target IDR length in residues |
| `training.lightning_model_args.length_reward_weight` | `1.0` | Weight for length reward |
| `training.lightning_model_args.length_reward_width` | `1` | Gaussian width (standard deviation in residues) |

**Entropy reward:**

| Parameter | Default | Description |
|---|---|---|
| `training.lightning_model_args.use_target_entropy` | `True` | Enable entropy reward |
| `training.lightning_model_args.target_entropy` | `2.7` | Target Shannon entropy of sequence composition (nats) |
| `training.lightning_model_args.entropy_reward_weight` | `1.0` | Weight for entropy reward |
| `training.lightning_model_args.entropy_reward_width` | `0.2` | Gaussian width (standard deviation in nats) |

The length and entropy rewards are Gaussian functions of the deviation from their respective targets: sequences that match `target_length` and `target_entropy` exactly receive maximum reward (1.0), and reward decays with distance controlled by the `*_width` parameters. -->
