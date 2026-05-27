# IDiom

IDiom is a 122M parameter autoregressive transformer trained on 37M intrinsically disordered regions from the AlphaFold Database. The model can generate intrinsically disordered proteins (IDPs) as well as intrinsically disordered regions (IDRs) conditioned on their flanking context. The model can also be post-trained with reinforcement learning to optimize for custom reward functions. Hidden state activations can also be extracted from the transformer residual streams for downstream tasks. The associated preprint is: [Generative design of intrinsically disordered protein regions with IDiom](https://doi.org/10.64898/2026.04.10.717777)

<p align="center">
  <img src="assets/github_fig.png" alt="IDiom" width="900px" align="middle"/>
</p>

# Table of Contents
- [IDiom](#IDiom)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
  - [Environment setup](#environment-setup)
  - [Model checkpoints and data](#model-checkpoints-and-data)
- [Generating sequences](#generating-sequences)
  - [Generating intrinsically disordered proteins](#generating-intrinsically-disordered-proteins)
  - [Generating intrinsically disordered regions](#generating-intrinsically-disordered-regions)
  - [Generating sequences of a specific length](#generating-sequences-of-a-specific-length)
- [Post-training](#post-training)
  - [Custom reward functions](#custom-reward-functions)
    - [Optimizing IDP generation](#optimizing-idp-generation)
    - [Optimizing prompted IDR generation](#optimizing-prompted-idr-generation)
  - [ProtGPS reward](#protgps-reward)
  - [Tracking training progress using Tensorboard](#tracking-training-progress-using-tensorboard)
  - [Generating sequences after post-training](#generating-sequences-after-post-training)
- [Extracting activations](#extracting-activations)
- [Pre-training](#pre-training)
- [Citation](#citation)

# Installation

## Environment setup
First, install the [uv](https://docs.astral.sh/uv/) package manager if not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Next, clone the IDiom repository into a directory with at least 30 GB of free space and install the dependencies:

```bash
git clone https://github.com/rotskoff-group/idiom.git
cd idiom
uv sync
uv pip install -e .
```



## Model checkpoints and data

Next, download the `IDiom` model checkpoints from the HuggingFace repository into the project root directory.

**Model checkpoints**: https://huggingface.co/jxliu2/idiom

You can do so with the following commands. From the root of the cloned `IDiom` directory, do: 

```bash
# Download models (26 GB)
# Execute from IDiom root directory:
hf download jxliu2/idiom --local-dir ./models
```

Additional datasets which are NOT necessary for running this code repository can be found in the following HuggingFace repository: https://huggingface.co/datasets/jxliu2/idiom-datasets

This includes the 37M IDRs used to pre-train `IDiom` as well as the generated sequences which we analyze in our paper. To download this OPTIONAL data, use the following command to download the entire dataset or manually download specific files of interest from the HuggingFace URL. 

```bash
# OPTIONALLY download the IDR data:
# Execute from IDiom root directory (186 GB): 
hf download jxliu2/idiom-datasets --repo-type=dataset --local-dir ./datasets

# If you only want the FASTA files containing the curated IDRs (12 GB and 3 GB), run: 
hf download jxliu2/idiom-datasets \
  idr_datasets/training_sequences/AFDB_IDR_90_FIM_512_full.fasta \
  --repo-type=dataset \
  --local-dir ./datasets

hf download jxliu2/idiom-datasets \
  idr_datasets/training_sequences/AFDB_IDR_90_FIM_512_idrs.fasta \
  --repo-type=dataset \
  --local-dir ./datasets
```

After this, the project structure should be:

```
idiom/
|
├── src/                       # Main Python package
│   └── idiom/
│       ├── nn/                # Model architecture
│       ├── scripts/           # CLI entry points and Hydra configs
│       └── utils/             # Utilities
|
├── entrypoints/               # Scripts for training and inference
│   ├── extract_activations/   # Scripts for extracting residual stream activations
│   ├── generate/              # Scripts for generating sequences 
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

Now, the example bash scripts described below can be run directly using `bash` or via SLURM using `sbatch`. 



# Generating sequences

IDiom allows for the generation of unprompted intrinsically disordered proteins (IDPs) or intrinsically disordered regions (IDRs) prompted by their surrounding flanking context within a protein. We have tested inference on NVIDIA GeForce RTX 4080 GPUs with 16 GB VRAM. Sequences should be post-processed after generation to filter for sequence metrics of interest. 



## Generating intrinsically disordered proteins

To generate unprompted IDPs, execute the `generate_idps.bash` script. 

```bash
cd entrypoints/generate/scripts
bash generate_idps.bash # or: sbatch generate_idps.bash
```

This script uses the pre-trained base model described in the paper to generate unprompted IDPs. You can specify the number of IDPs to generate by modifying the `NUM_DUPLICATES` variable (default 1000) near the top of `generate_idps.bash`. 

Generated sequences are output as FASTA files in the `entrypoints/generate/output/idps` directory. The following files will be created: 

- `tst_autoregressive.pkl` — Raw generated token sequences
- `generated_idrs.fasta` — FASTA file containing the generated disordered sequences 
- `generated_full.fasta` — Same as above, with indices of the disordered region in each sequence header header
- `inference_config.yaml` — Inference configuration file 



## Generating intrinsically disordered regions

To generate IDRs conditioned on their surrounding context, you must provide a FASTA file containing the full-length protein(s) you would like to generate IDRs within. An example file is provided at: `entrypoints/generate/scripts/example_sequences.fasta`. 

This FASTA contains two full-length protein sequences. Each sequence entry MUST have a header which ends with the string "_IDR_x-y" where x and y indicate the start and end indices (1-indexed) of the wild type IDR. For example, in the provided FASTA, the wild type IDR of the first sequence begins at 119 and ends at 242. 

```
>P06748_IDR_119-242
MEDSMDMDMSPLRPQNYLFGCELKADKDYHFKVDN...

>P09651_IDR_186-372
MSKSESPKEPEQLRKLFIGGLSFETTDESL...
```

The code automatically extracts the N-terminal prefix and C-terminal suffix to the indicated IDR and uses those as the conditioning for generation. 

To generate IDRs, execute the example bash script: 

```bash
cd entrypoints/generate/scripts
bash generate_idrs.bash # or: sbatch generate_idrs.bash
```

This script also uses the pre-trained base model described in the paper. You can specify the number of IDRs to generate by modifying the `NUM_DUPLICATES` variable (default 1000) near the top of `generate_idrs.bash`. The script will generate `NUM_DUPLICATES` IDRs for each sequence provided in the FASTA file. 

Generated sequences are output as FASTA files in the `entrypoints/generate/output/idrs` directory. The following files will be created: 

- `tst_autoregressive.pkl` — Raw generated token sequences
- `generated_idrs.fasta` — Contains the generated IDR sequences
- `generated_full.fasta` — Contains the full length sequences with indices of the generated disordered region in each sequence's header
- `inference_config.yaml` — Inference configuration file 




## Generating sequences of a specific length

IDiom also supports length-controlled generation, where generated sequences are filtered to only keep those whose disordered region falls within a target length window. Generation repeats automatically until the requested number of valid-length sequences has been generated.

To generate IDPs of a specific length, execute the `generate_idps_length.bash` script:

```bash
cd entrypoints/generate/scripts
bash generate_idps_length.bash # or: sbatch generate_idps_length.bash
```

To generate IDRs of a specific length, execute the `generate_idrs_length.bash` script:

```bash
cd entrypoints/generate/scripts
bash generate_idrs_length.bash # or: sbatch generate_idrs_length.bash
```

In either script, set the following variables near the top of the file before running:

- `SEQ_LENGTH` — target disordered region length in residues (default: 100)
- `SEQ_LENGTH_RANGE` — allowed deviation from the target; sequences with IDR length within `SEQ_LENGTH +/- SEQ_LENGTH_RANGE` are kept (default: 5)
- `NUM_DUPLICATES` — number of valid-length sequences to generate (default: 1000)

Output files are written to `entrypoints/generate/output/idps_length` or `entrypoints/generate/output/idrs_length` respectively, with the same file structure as the standard generation scripts.

<!-- These parameters can also be passed directly as Hydra overrides to `transformer_infer` without using the bash scripts:

```bash
transformer_infer \
    ... \
    "++inference.addn_args.seq_length=50" \
    "++inference.addn_args.seq_length_range=10"
``` -->



# Post-training

Here we describe the post-training workflows that can be done with IDiom. Post-training can be done with any custom reward function, and post-training can be used to optimize the generation of either IDPs or IDRs. We have tested post-training on NVIDIA GeForce RTX 4080 GPUs with 16 GB VRAM. 

**Out-of-memory errors during training.** If you encounter GPU OOM errors during post-training, in the training submission scripts, reduce the `BATCH_SIZE` hyperparameter and increase `ACCUMULATE_GRAD_BATCHES` by the same factor to keep the effective batch size constant. This applies to all post-training workflows.

## Custom reward functions

You can define your own custom reward function in `rewards/custom_rewards/custom_rewards.py`. An example function is given: `compute_fraction_proline()`. 

This example reward function extracts the disordered region from the generated sequence and calculates the fraction of proline residues in the IDR as the reward. Reward values should be in the range 0 to 1. 

### Optimizing IDP generation 

To run post-training with this example reward function on generated unprompted IDPs, execute this script: 

```bash
bash entrypoints/train/post-train/train_rl_idp_custom.bash # or sbatch 
```

When you define your own custom reward function in `custom_rewards.py`, the function must begin with "compute_". Then, you should modify the configuration parameter `reward_function_name` in the bash script to be your function's name. 

### Optimizing prompted IDR generation 

To optimize the generation of IDRs prompted with flanking context, you must provide a FASTA file containing a single protein sequence. Again, this sequence's header MUST have a header which ends with the string "_IDR_x-y" where x and y indicate the start and end indices (1-indexed) of the wild type IDR. 

An example sequence is provided at `entrypoints/train/post-train/rl_sequence.fasta`. To run training, execute the bash script: 

```bash
bash entrypoints/train/post-train/train_rl_idr_custom.bash # or sbatch 
```

This will use the flanking context of the IDR in the FASTA file as the prompt in generating IDRs for RL optimization. 


## ProtGPS reward

As examples, we also provide training scripts to replicate our training runs with the ProtGPS localization score as the reward. 

The script used to optimize unprompted IDPs is: 

```bash
bash entrypoints/train/post-train/train_rl_idp_protgps.bash # or sbatch 
```

And a script for optimizing prompted IDRs is: 

```bash
bash entrypoints/train/post-train/train_rl_idr_protgps.bash # or sbatch 
```

## Tracking training progress using Tensorboard 

To track progress on post-training runs, first activate the virtual environment. From the repo root:

```bash
source .venv/bin/activate 
```

Then, use Tensorboard by first navigating to the directory containing `lightning_logs` and run: 

```bash
tensorboard --logdir . 
```

## Generating sequences after post-training

To generate sequences from a post-trained model checkpoint, set the `CKPT_PATH` in `generate_idps.bash` or `generate_idrs.bash` to be the post-trained checkpoint (.ckpt) located in the lightning_log. Then run the generation script as above: 

```bash
bash entrypoints/generate/scripts/generate_idps.bash  # or generate_idrs.bash
```


# Extracting activations

Residual stream activations after each transformer block can also be extracted from IDiom for downstream analysis. The `extract_activations.bash` script runs the pre-trained base model (or any post-trained checkpoint) over a set of sequences and saves the per-layer activations to an HDF5 file. Note that only residue positions are saved. Both control tokens (`BOS`, `EOS`, `PAD`, `MASK`) and the FIM markers `1`, `3`, `2` are filtered out during extraction. Each saved row is labeled with a `fim_segment` value (`1` = prefix, `3` = suffix, `2` = IDR) so the segment that a residue came from can be identified. Activation extraction can be done using the following command: 

```bash
cd entrypoints/extract_activations/scripts
bash extract_activations.bash # or: sbatch extract_activations.bash
```

As an example, the script extracts activations from the last transformer block for the two example proteins in `entrypoints/generate/scripts/example_sequences.fasta`. To run on your own sequences, set `DATA_PATH` near the top of the script to: 

- A FASTA file whose sequence headers end with `_IDR_x-y` or 
- A raw sequences `.h5` file containing a `residues` field, where the sequences in the `residues` field are already transformed into a fill-in-the-middle format with the `1`, `2`, and `3` characters present. The `residues` dataset must use an h5py utf-8 string dtype `bytes`.  

The following options can be adjusted near the top of the script:

- `++extract.layers` — 0-indexed transformer blocks to extract activations from (e.g. `[11]` for the last block only, or `[0,1,2,3,4,5,6,7,8,9,10,11]` for all blocks)
- `++extract.save_dtype` — `float16` (default) or `float32`
- `++extract.max_sequences` — cap the maximum number of sequences processed (`null` = all)
- `++extract.use_multi_gpu` — set to `true` and increase `--gpus` to parallelize activation extraction 

Activations are written to `entrypoints/extract_activations/output/activations.h5` with the following structure:

- `activations/layer_<i>/data` — activation matrix for block `i`, shape `[num_tokens, d_model]` (one row per kept residue; control tokens `BOS/EOS/PAD/MASK` and FIM markers `1/2/3` are filtered out)
- `activations/layer_<i>/seq_idx` — index of the sequence each row belongs to (global, across all GPUs)
- `activations/layer_<i>/pos_idx` — token position within the original tokenized sequence (control tokens and FIM markers included in the count)
- `activations/layer_<i>/local_pos_idx` — rank of the row among kept residues in its sequence (i.e. index into the matching `aligned_strings` entry)
- `activations/layer_<i>/fim_segment` — FIM segment label for each row: `1` = prefix, `3` = suffix, `2` = IDR
- `sequences/tokens` — kept token IDs for each sequence (variable-length, control/FIM tokens removed)
- `sequences/strings` — raw FIM-formatted residue string for each sequence (includes `1/2/3` markers)
- `sequences/aligned_strings` — residue string with one character per kept activation row, aligned 1-to-1 with `data` rows of the same `seq_idx`
- `metadata/alphabet`, `metadata/layers` — the token alphabet and the list of extracted layers
- `extract_config.yaml` — the extraction configuration, written alongside the output file
- To obtain the IDR-only activations for a sequence, mask with `fim_segment == 2`.


# Pre-training

To replicate the model pre-training, you must first download the appropriate datasets from HuggingFace. 

**Datasets**: https://huggingface.co/datasets/jxliu2/idiom-datasets

From the repo root directory, execute: 

```bash
# Download the IDR data (186 GB):
# Execute from IDiom root directory: 
hf download jxliu2/idiom-datasets --repo-type=dataset --local-dir ./datasets
```

Then, execute the precompute to prepare the training sequences for model training. Note that at least 1 TB of space is required for the precompute. 

```bash
sbatch combined_precompute.bash 
```

Next, execute the training script: 

```bash
sbatch pretrain.bash 
```

# Contributing

We welcome all contributions to this open-source project! Please feel free to fork the repository, raise issues, contribute reward functions, and initiate pull requests. Do not hesitate to contact the authors if you have questions, ideas, or comments. Thank you!

# Citation

If you find this work useful, please cite: 

```bibtex
@article{liu2026idiom,
  author = {Liu, Jason and Ibarraran, Sebastian and Hu, Frank and Park, Abigail and Dunn, Alexander and Rotskoff, Grant},
  title = {Generative design of intrinsically disordered protein regions with {IDiom}},
  journal = {bioRxiv},
  year = {2026},
  doi = {10.64898/2026.04.10.717777},
  URL = {https://doi.org/10.64898/2026.04.10.717777},
}
```
