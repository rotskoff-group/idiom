# IDR-PLM
Official repository for IDR-PLM: a protein language model for generating intrinsically disordered regions (IDRs) and intrinsically disordered proteins (IDPs), with support for reward-guided post-training via Group Relative Policy Optimization (GRPO).

## Overview

IDR-PLM is an autoregressive transformer trained on intrinsically disordered regions from the AlphaFold Database (AFDB). It uses a Fill-In-Middle (FIM) objective, which allows the model to generate an IDR conditioned on its flanking structured regions. After pre-training, the model can be post-trained with reinforcement learning (GRPO) to optimize any user-defined reward—for example, predicted subcellular localization via ProtGPS.


## Environment setup

First, clone and sync the repository: 

```bash
git clone https://github.com/rotskoff-group/idr-plm.git
cd idr-plm
uv sync
uv pip install -e .
```

Requires Python ≥ 3.10 and PyTorch 2.4.0.

## Model checkpoints and data 

Next, download the pre- and post-trained `IDR-PLM` model checkpoints, as described in the paper, from the HuggingFace repository and move it into the `idr-plm/` project root directory. 

**Model checkpoints**: https://huggingface.co/jxliu2/idr-plm

The project structure should be:


```
models/
├── data/
│   ├── shard/
│   │   └── 0001_file.h5          # tokenized sequence shard
│   ├── prompts/
│   │   ├── idp_prompt_*_array.pkl
│   │   ├── idp_prompt_*_metadata.pkl
│   │   ├── p06748_prompt_*_array.pkl
│   │   └── p06748_prompt_*_metadata.pkl
│   └── rl_datasets/
│       └── idp_dataset/
│           └── idp_prompt_1e3x_grpo_dataset.h5
├── idr-plm/
│   └── base/
│       └── version_2/
│           └── checkpoints/
│               └── best_model_step_243022.ckpt
└── protgps/                      # ProtGPS reward model weights
```

Additional datasets which not necessary for running the code repository can be found in the following HuggingFace repository. This includes the 37M IDRs used to pre-train `IDR-PLM` as well as the generated sequences which we analyze in the paper. (This data is necessary if you would like to replicate the pre-training run.)

**Datasets**: https://huggingface.co/datasets/jxliu2/idr-plm-dataset 



### Model architecture

The core model is `GeometricMolTransformer`, a 12-layer causal transformer with:

| Hyperparameter | Value |
|---|---|
| Model dimension (`d_model`) | 896 |
| Transformer layers | 12 |
| Attention heads | 14 |
| FFN type | SwiGLU (2.667× expansion) |
| Vocabulary size | 24 amino acids + special tokens |
| Positional encoding | Rotary (RoPE) |
| Attention mask | Causal |

### Tokenization and fill-in-the-middle format

| Token | Index | Role |
|---|---|---|
| PAD | 23 | Padding |
| START | 24 | Beginning of sequence |
| STOP | 25 | End of sequence |
| MASK | 26 | Masking |
| Sentinel `1` | — | FIM prefix (N-terminal structured region) |
| Sentinel `2` | — | FIM middle (IDR region to generate) |
| Sentinel `3` | — | FIM suffix (C-terminal structured region) |

FIM prompt format: `<START> 1 <prefix> 3 <suffix> 2 <middle tokens...> <STOP>`

During inference, the model receives `<START> 1 <prefix> 3 <suffix> 2` and generates the middle (IDR) autoregressively.

---

## CLI commands

After installation, five CLI commands are available:

| Command | Purpose |
|---|---|
| `transformer_precompute` | Precompute and tokenize training shards from raw sequences |
| `transformer_train` | Train or post-train the model |
| `transformer_infer` | Generate sequences with a trained model |
| `make_infer_prompt` | Build inference prompt files |
| `make_rl_dataset` | Build a GRPO training dataset from a FASTA file |

All commands use [Hydra](https://hydra.cc/) for configuration. Default configs live in `src/idr_plm/scripts/cfgs/`.

---

## Workflows

### 1. Generating unprompted IDPs

To generate IDPs (full disordered sequences) without any flanking context, first create an IDP prompt file (or use the one already in `models/data/prompts/`):

```bash
# Step 1: Make IDP prompts (repeated sentinel pattern, no flanking residues)
bash entrypoints/infer/scripts/make_idp_prompts.bash
# equivalent to:
make_infer_prompt \
    --shard   models/data/shard/0001_file.h5 \
    --out_dir models/data/prompts \
    idp \
    --num_duplicates 10000
```

```bash
# Step 2: Run inference
bash entrypoints/infer/scripts/infer_idp.bash
# equivalent to:
transformer_infer \
    "model=transformer" \
    "model.model_args.d_model=896" \
    "model.model_args.unified_transformer_args.n_layers=12" \
    "inference.checkpoint_path=models/idr-plm/base/version_2/checkpoints/best_model_step_243022.ckpt" \
    "inference.dataset_filename=models/data/shard/0001_file.h5" \
    "inference.savedir=entrypoints/infer/output" \
    "inference.batch_size=100" \
    "inference.use_multi_gpu=True" \
    "inference.sampler_args.method=full" \
    "inference.sampler_args.temperature=1.0" \
    "++inference.addn_args.use_input_smiles=True" \
    "++inference.addn_args.smiles_path=models/data/prompts/idp_prompt_1e5x_array.pkl"
```

**Output** (in `inference.savedir`):
- `tst_autoregressive.pkl` — raw generated token sequences
- `generated_idrs.fasta` — IDR regions only (extracted from between sentinel `2` tokens)
- `generated_full.fasta` — full sequences with IDR coordinates in the header

---

### 2. Generating prompted IDRs (context-conditioned)

To generate IDRs conditioned on flanking protein sequences (FIM mode), provide a FASTA file where each entry has the header format `>ACCESSION_IDR_START-END` and the sequence is the full-length protein. The prefix and suffix flanking the IDR region are automatically extracted.

```bash
# Step 1: Make specific (FIM) prompts from FASTA
bash entrypoints/infer/scripts/make_specific_prompts.bash
# equivalent to:
make_infer_prompt \
    --shard   models/data/shard/0001_file.h5 \
    --out_dir models/data/prompts \
    specific \
    --fasta   entrypoints/infer/scripts/example_sequences.fasta \
    --num_duplicates 10000
```

An example FASTA is provided at `entrypoints/infer/scripts/example_sequences.fasta`:

```
>P06748_IDR_119-242
MEDSMDMDMSPLRPQNYLFGCELKADKDYHFKVDN...
>P09651_IDR_186-372
MSKSESPKEPEQLRKLFIGGLSFETTDESL...
```

```bash
# Step 2: Run inference (same as above, point smiles_path to new prompt file)
transformer_infer \
    ... \
    "++inference.addn_args.smiles_path=models/data/prompts/<your_prompt>_array.pkl"
```

---

### 3. Post-training with the ProtGPS reward

Post-training uses GRPO to optimize ProtGPS-predicted subcellular localization scores. Twelve compartment-specific models are provided under `models/idr-plm/post_trained/protgps_reward/`.

Available compartments: `nuclear_speckle`, `p-body`, `pml-body`, `post_synaptic_density`, `stress_granule`, `chromosome`, `nucleolus`, `nuclear_pore_complex`, `cajal_body`, `rna_granule`, `cell_junction`, `transcriptional`.

**For generating IDPs** (no flanking context, uses `idp_prompt_1e3x_grpo_dataset.h5`):

```bash
bash entrypoints/train/post-train/train_rl_protgps.bash
```

Key parameters (edit at the top of the script):

| Parameter | Default | Description |
|---|---|---|
| `LR` | `5e-6` | Learning rate |
| `BETA_KL` | `2e-2` | KL divergence penalty weight |
| `REWARD_TARGET_VALUE` | `0.9` | Reward shaping target (ProtGPS score) |
| `TARGET_LENGTH` | `100` | Target IDR length (residues) |
| `LENGTH_REWARD_WEIGHT` | `1.0` | Weight for length reward |
| `TARGET_ENTROPY` | `2.7` | Target Shannon entropy of composition |
| `ENTROPY_REWARD_WEIGHT` | `1.0` | Weight for entropy reward |
| `GROUP_SIZE` | `8` | Number of completions per prompt (GRPO) |
| `COMPARTMENT` | `stress_granule` | ProtGPS localization target |

**For generating context-conditioned IDRs** (specific protein of interest):

```bash
# First, create an RL dataset from your protein FASTA:
bash entrypoints/train/post-train/make_rl_dataset.bash
# equivalent to:
make_rl_dataset \
    --fasta entrypoints/train/post-train/rl_sequence.fasta \
    --shard models/data/shard/0001_file.h5 \
    --out_dir models/data/rl_datasets/specific_dataset

# Then run training pointing DATASET_FILENAME to the new dataset.
```

The `rl_sequence.fasta` must use the same `>ACCESSION_IDR_START-END` header format as above.

---

### 4. Post-training with a custom reward function

You can post-train with any reward function by editing `rewards/custom_rewards/custom_rewards.py`. Your function must:
- Be named starting with `compute_`
- Accept `(tokens, token_info, device)` as arguments
- Return a scalar `torch.Tensor` on `device`

A worked example (`compute_fraction_proline`) is provided in that file.

```bash
bash entrypoints/train/post-train/train_rl_custom.bash
```

The script adds `rewards/custom_rewards/` to `PYTHONPATH` and sets `reward_function_name=compute_fraction_proline` (edit to match your function name).

---

### 5. Generating sequences after post-training

Inference after post-training is identical to the base model inference (workflows 1 and 2 above), except you point `inference.checkpoint_path` to your post-trained checkpoint:

```bash
transformer_infer \
    ... \
    "inference.checkpoint_path=models/idr-plm/post_trained/protgps_reward/<version>/checkpoints/<step>.ckpt" \
    ...
```

---

### 6. Pre-training from scratch

#### Step 1: Prepare the pre-training data

Download `AFDB_IDR_90_FIM_512.h5` from HuggingFace. Split it into parts for parallel precomputation:

```bash
bash entrypoints/precompute/split_parts.bash
# This splits the main HDF5 into ~500 parts, one per SLURM array job
```

#### Step 2: Precompute tokenized shards

Run `transformer_precompute` on each part (parallelized as a SLURM array):

```bash
sbatch entrypoints/precompute/precompute.bash
# equivalent to (per part):
transformer_precompute "precompute=smiles" \
    "precompute.smiles_file=part_${N}_idrs.h5" \
    "precompute.target_file=part_${N}_targs.h5" \
    "precompute.output_file=precompute_shards/${N}_file.h5" \
    "precompute.input_generator_addn_args.apply_start=true" \
    "precompute.input_generator_addn_args.apply_stop=false" \
    "precompute.target_generator_addn_args.apply_start=false" \
    "precompute.target_generator_addn_args.apply_stop=true" \
    "precompute.tokenizer=CharTokenizer"
```

Each output shard contains `res_tokens`, `targets`, `structural_tokens`, `sequence_id` (attention mask), and tokenizer metadata.

#### Step 3: Pre-train the model

```bash
sbatch entrypoints/train/pre-train/pretrain.bash
# equivalent to:
transformer_train \
    "data=transformer" \
    "data.dataset=TransformerShardedAutoregDataset" \
    "data.dataset_filename=<SHARDS_DIR>" \
    "data.dloader_args.batch_size=128" \
    "model=transformer" \
    "model.model_args.d_model=896" \
    "model.model_args.unified_transformer_args.n_layers=12" \
    "training=transformer" \
    "training.training_mode=autoregressive" \
    "training.trainer_args.devices=8" \
    "training.trainer_args.max_steps=250000" \
    "training.lightning_model_args.optimizer_args.lr=4.0e-4" \
    "training.lightning_model_args.lr_scheduler=LinearWarmupCosineAnnealingLR" \
    "++training.lightning_model_args.lr_scheduler_args.warmup_epochs=3000" \
    "++training.lightning_model_args.lr_scheduler_args.max_epochs=250000" \
    "++training.lightning_model_args.lr_scheduler_args.eta_min=4.0e-5"
```

Training runs for 250,000 steps on 8 GPUs with mixed precision (fp16). Checkpoints are saved every 1,000 steps (`restart_checkpoint.ckpt`) and the best validation-loss model is saved as `best_model_step_<N>.ckpt`.

---

<!-- ## Repository structure

```
idr-plm/
├── src/idr_plm/                         # Main Python package
│   ├── nn/
│   │   ├── layers/                      # Transformer building blocks
│   │   │   ├── blocks.py                #   UnifiedTransformerBlock (SwiGLU / GELU FFN)
│   │   │   ├── transformer_stack.py     #   TransformerStack (stacked blocks + LayerNorm)
│   │   │   ├── mha.py                   #   Multi-head attention
│   │   │   ├── rotary.py                #   Rotary positional embeddings (RoPE)
│   │   │   └── regression_head.py       #   Output projection head
│   │   └── transformer/
│   │       ├── nn.py                    #   GeometricMolTransformer (main model)
│   │       ├── module.py                #   PyTorch Lightning module (autoregressive + GRPO)
│   │       ├── dataset.py               #   Dataset classes (sharded / online)
│   │       ├── losses/
│   │       │   ├── autoreg_loss.py      #   Cross-entropy for pre-training
│   │       │   └── grpo_loss.py         #   GRPO loss (rewards, KL, PPO clip)
│   │       ├── scores.py                #   IDR region extraction and scoring
│   │       └── utils/
│   │           ├── tokenizer.py         #   CharTokenizer
│   │           ├── sampling.py          #   Autoregressive sampling functions
│   │           └── misc.py              #   Policy log-probability utilities
│   ├── scripts/
│   │   ├── cfgs/                        # Hydra config files
│   │   │   ├── model/transformer.yaml
│   │   │   ├── training/transformer.yaml
│   │   │   ├── data/transformer.yaml
│   │   │   └── inference/transformer.yaml
│   │   ├── data/
│   │   │   ├── precompute.py            # transformer_precompute
│   │   │   ├── make_infer_prompt.py     # make_infer_prompt
│   │   │   └── make_rl_dataset.py       # make_rl_dataset
│   │   └── transformer/
│   │       ├── train.py                 # transformer_train
│   │       └── inference.py             # transformer_infer
│   └── utils/
│       ├── sampler.py                   # TokenSampler (top-k / top-p / full)
│       ├── data_utils.py                # HDF5 loading utilities
│       └── misc.py                      # Sequence utilities, IDR extraction
├── entrypoints/
│   ├── precompute/                      # Data preparation scripts
│   ├── train/
│   │   ├── pre-train/pretrain.bash      # Pre-training launcher
│   │   └── post-train/
│   │       ├── train_rl_protgps.bash    # ProtGPS GRPO post-training
│   │       ├── train_rl_custom.bash     # Custom reward GRPO post-training
│   │       ├── make_rl_dataset.bash     # Build GRPO dataset from FASTA
│   │       └── rl_sequence.fasta        # Example input FASTA
│   └── infer/scripts/
│       ├── infer_idp.bash               # Unprompted IDP generation
│       ├── infer_p06748.bash            # Prompted IDR generation (example)
│       ├── make_idp_prompts.bash        # Build IDP prompt file
│       ├── make_specific_prompts.bash   # Build FIM prompt file from FASTA
│       └── example_sequences.fasta     # Example input FASTA for prompts
├── models/                              # Checkpoints and data (populated from HuggingFace)
│   ├── data/                            # Shards, prompts, RL datasets
│   ├── idr-plm/base/                    # Pre-trained base model
│   ├── idr-plm/post_trained/            # Post-trained models (one per compartment)
│   └── protgps/                         # ProtGPS reward model weights
├── rewards/
│   ├── protgps/                         # ProtGPS model code (ESM2-based classifier)
│   └── custom_rewards/
│       └── custom_rewards.py            # User-defined reward functions
└── pyproject.toml
```

--- -->

## GRPO post-training details

Post-training uses GRPO (Group Relative Policy Optimization), an RL algorithm analogous to PPO but designed for language model fine-tuning. For each prompt, the model generates `GROUP_SIZE=8` completions, computes rewards, normalizes advantages within the group, and updates the policy with a PPO-style clipped objective plus a KL penalty against the frozen reference (pre-trained) model.

The total reward at each step combines up to three components:

1. **Task reward** (`reward_function_name`): ProtGPS localization score or custom function. Optional quadratic shaping toward `reward_target_value=0.9`.
2. **Length reward**: Gaussian penalty for deviation from `target_length` (width controlled by `length_reward_width`).
3. **Entropy reward**: Gaussian penalty for deviation from `target_entropy` (Shannon entropy of sequence composition; target ≈ 2.7 nats).

**GRPO hyperparameters:**

| Parameter | Default | Description |
|---|---|---|
| `group_size` | 8 | Completions per prompt |
| `epsilon_clip` | 0.2 | PPO clipping threshold |
| `beta_kl` | 0.02 | KL divergence penalty |
| `mu_grpo` | 1 | GRPO loss scale |
| `normalize_advantage` | True | Group-wise advantage normalization |
| `max_steps` | 1500 | Total training steps |
| `lr` | 5e-6 | AdamW learning rate |
| `batch_size` | 4 | Prompts per step (×`group_size` completions) |
| `accumulate_grad_batches` | 2 | Gradient accumulation |

---
<!-- 
## ProtGPS reward model

ProtGPS is an ESM2-based classifier that predicts IDR localization to 12 subcellular compartments. It is stored under `rewards/protgps/` and its weights under `models/protgps/`. Post-trained checkpoints are available for all 12 compartments:

| Compartment | Version |
|---|---|
| `nuclear_speckle` | version_59380_0 |
| `p-body` | version_59380_1 |
| `pml-body` | version_59380_2 |
| `post_synaptic_density` | version_59380_3 |
| `stress_granule` | version_59380_4 |
| `chromosome` | version_59380_5 |
| `nucleolus` | version_59392_0 |
| `nuclear_pore_complex` | version_59392_1 |
| `cajal_body` | version_59392_2 |
| `rna_granule` | version_59392_3 |
| `cell_junction` | version_59392_4 |
| `transcriptional` | version_59392_5 | -->
