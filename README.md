# IDR-PLM

IDR-PLM is an autoregressive transformer trained on 37M intrinsically disordered regions from the AlphaFold Database. The model can generate intrinsically disordered proteins (IDPs) as well as intrinsically disordered regions (IDRs) conditioned on their flanking context. The model can be post-trained with reinforcement learning to optimize for custom reward functions. The associated paper can be found at: bioarxiv link

<p align="center">
  <img src="assets/github_fig.png" alt="IDR-PLM" width="1100px" align="middle"/>
</p>

# Table of Contents
- [IDR-PLM](#idr-plm)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
  - [Model checkpoints and data](#model-checkpoints-and-data)
  - [Model architecture](#model-architecture)
- [Generating sequences](#generating-sequences)
  - [Generating unprompted IDPs](#generating-unprompted-idps)
  - [Generating context-conditioned IDRs](#generating-context-conditioned-idrs)
- [Post-training](#post-training)
  - [Custom reward](#custom-reward)
    - [IDPs](#idps)
    - [IDRs](#idrs)
  - [ProtGPS reward](#protgps-reward)
    - [IDPs](#idps-1)
    - [IDRs](#idrs-1)
  - [Generating sequences after post-training](#generating-sequences-after-post-training)
- [Pre-training](#pre-training)
- [Citation](#citation)

# Installation

## Environment setup
First, install the [uv](https://docs.astral.sh/uv/) package manager if not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Next, clone the IDR-PLM repository into a directory with at least 30 GB of free space (the space is necessary for model checkpoints) and install dependencies:

```bash
git clone https://github.com/rotskoff-group/idr-plm.git
cd idr-plm
uv sync
uv pip install -e .
```



## Model checkpoints and data

Next, download the `IDR-PLM` model checkpoints from the HuggingFace repository into the project root directory.

**Model checkpoints**: https://huggingface.co/jxliu2/idr-plm

You can do so with the following commands. You must first create a HuggingFace account. Then, from the root of the cloned `IDR-PLM` directory, do: 

```bash
uv tool install hf # Install the hf cli tool
hf auth login # Enter your credentials when prompted

# Download models (26 GB)
# Execute from IDR-PLM root directory: 
hf download jxliu2/idr-plm --local-dir ./models
```

Additional datasets which are not necessary for running this code repository can be found in the following HuggingFace repository. 

**Datasets**: https://huggingface.co/datasets/jxliu2/idr-plm-dataset

This includes the 37M IDRs used to pre-train `IDR-PLM` as well as the generated sequences which we analyze in our paper. (This data is necessary if you would like to replicate pre-training.) 

```bash
# Optionally download the IDR data (174 GB):
# Execute from IDR-PLM root directory: 
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
│       └── utils/             # Utilities
|
├── entrypoints/               # Scripts for training and inference
│   ├── infer/                 # Inference scripts and output
│   ├── precompute/            # Data preprocessing scripts
│   └── train/                 # Pre- and post-training scripts
|
├── rewards/                   # Reward functions and models
│   ├── custom_rewards/        # Custom reward functions
│   └── protgps/               # ProtGPS localization reward model
|
├── models/                    # Model checkpoints
|
├── assets/                    # Images
|
└── datasets/                  # Datasets (optional)
```


## Model architecture

The core model is `GeometricMolTransformer`, a 12-layer causal transformer with 122M trainable parameters:

| Hyperparameter | Value |
|---|---|
| Model dimension | 896 |
| Transformer layers | 12 |
| Attention heads | 14 |
| FFN type | SwiGLU (2.667× expansion) |
| Vocabulary size | 20 amino acids, 3 location tokens, 4 special tokens |
| Positional encoding | Rotary (RoPE) |
| Attention mask | Causal |



# Generating sequences

After installation, each script can be run directly using `bash` or via SLURM using `sbatch`. All scripts automatically detect the execution context and set the repository root accordingly.


## Generating unprompted IDPs

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



## Generating context-conditioned IDRs

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

# Post-training

> **Out-of-memory errors during training.** If you encounter GPU OOM errors, reduce `BATCH_SIZE` and increase `ACCUMULATE_GRAD_BATCHES` by the same factor to keep the effective batch size constant (e.g. `BATCH_SIZE=2, ACCUMULATE_GRAD_BATCHES=4` → `BATCH_SIZE=1, ACCUMULATE_GRAD_BATCHES=8`). This applies to all post-training workflows.

## Custom reward

### IDPs

```bash
bash entrypoints/train/post-train/train_rl_idp_custom.bash
```

Creates an IDP RL dataset and runs GRPO training with a user-defined reward. Edit `reward_function_name` in the script to point to your function (default: `compute_fraction_proline`, a worked example in `rewards/custom_rewards/custom_rewards.py`) and set `REWARD_TARGET_VALUE` to an appropriate target. Your reward function must accept `(tokens, token_info, device)` and return a scalar `torch.Tensor` on `device`.

### IDRs

**Input format.** Provide a FASTA file at `entrypoints/train/post-train/rl_sequence.fasta` using the same `>ACCESSION_IDR_START-END` header format as the inference workflows.

```bash
bash entrypoints/train/post-train/train_rl_idr_custom.bash
```

Creates an IDR RL dataset from the FASTA and runs GRPO training with a user-defined reward. Identical to IDPs above except prompts include flanking sequence context. Edit `reward_function_name` and `REWARD_TARGET_VALUE` to match your function.

## ProtGPS reward

### IDPs

```bash
bash entrypoints/train/post-train/train_rl_idp_protgps.bash
# or:
sbatch entrypoints/train/post-train/train_rl_idp_protgps.bash
```

Creates an IDP RL dataset and runs GRPO training using the ProtGPS subcellular localization classifier as the reward. Edit `COMPARTMENT` to select the target compartment (default: `stress_granule`) and adjust reward, length, and entropy parameters as needed. Available compartments: `nuclear_speckle`, `p-body`, `pml-body`, `post_synaptic_density`, `stress_granule`, `chromosome`, `nucleolus`, `nuclear_pore_complex`, `cajal_body`, `rna_granule`, `cell_junction`, `transcriptional`. Checkpoints are saved every 500 steps.

### IDRs

**Input format.** Provide a FASTA file at `entrypoints/train/post-train/rl_sequence.fasta` using the same `>ACCESSION_IDR_START-END` header format as the inference workflows.

```bash
bash entrypoints/train/post-train/train_rl_idr_protgps.bash
# or:
sbatch entrypoints/train/post-train/train_rl_idr_protgps.bash
```

Creates an IDR RL dataset from the FASTA and runs GRPO training with the ProtGPS reward. Identical to IDPs above except prompts include flanking sequence context. Default compartment is `nucleolus`.

## Generating sequences after post-training

```bash
bash entrypoints/infer/scripts/generate_idps.bash  # or generate_idrs.bash
```

Set `CKPT_PATH` in the script to your post-trained checkpoint before running. Otherwise identical to the generation workflows above.



# Pre-training

```bash
bash entrypoints/precompute/split_parts.bash        # split raw HDF5 into parts
sbatch entrypoints/precompute/precompute.bash       # tokenize each part (SLURM array)
sbatch entrypoints/train/pre-train/pretrain.bash    # train for 250k steps on 8 GPUs
```

Download `AFDB_IDR_90_FIM_512.h5` from HuggingFace first. The split script divides it into ~500 parts for parallel precomputation; each resulting shard contains tokenized inputs and targets. Pre-training runs for 250,000 steps with a linear warmup cosine annealing schedule; the best validation checkpoint is saved as `best_model_step_<N>.ckpt`.

# Citation

```bibtex
@article{liu2025idrplm,
  author = {},
  title = {},
  journal = {bioRxiv},
  year = {2025},
  doi = {},
  URL = {},
}
```
