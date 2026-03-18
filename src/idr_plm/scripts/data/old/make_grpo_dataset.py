#!/usr/bin/env python3
# source /home/groups/ardunn/jxliu2/MoLE/.venv/bin/activate # Sherlock
# source /home/jxliu2/MoLE/.venv/bin/activate # H100 cluster
"""
Make GRPO dataset HDF5. Create tokens and masks datasets (prompts) and copy alphabet and input_metadata from a shard file.
"""

# %%
import os
import pickle
from pathlib import Path
import numpy as np
import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]

# prompts_pkl = '/home/scratch/group_scratch/idr_plm/2025-11-12_protgps_RL/dataset/idp_prompt_prompt_1e3x_array.pkl'
prompts_pkl = str(SCRIPT_DIR / "idp_prompt_1e3x_array.pkl")

# Shard h5 containing alphabet and input_metadata
# shard = '/home/scratch/group_scratch/idr_plm/102425OnlineERADebugging/dataset/0001_file.h5'
shard = str(REPO_ROOT / "data" / "shard" / "0001_file.h5")

# Output path for the GRPO dataset
output_h5 = str(SCRIPT_DIR / "idp_prompt_1e3x_grpo_dataset.h5")

# Load prompts
with open(prompts_pkl, "rb") as f:
    # These prompts are already tokenized
    prompts = pickle.load(f)

# Open shard and read metadata
src = h5py.File(shard, "r")
alphabet_bytes = src["alphabet"][:]
alphabet = [x.decode("utf-8") for x in alphabet_bytes]
pad_token = src["input_metadata"]["ctrl_tokens"]["TOK_PAD"][()]

# Pad sequences to uniform length
maxlen = max(len(arr) for arr in prompts)
tokens = np.stack(
    [np.pad(arr, (0, maxlen - len(arr)), constant_values=pad_token) for arr in prompts],
    axis=0,
)

# Create masks
masks = (tokens != pad_token).astype(np.int32)
# masks = np.array([np.ones_like(prompt, dtype=np.int32) for prompt in prompts], dtype=object)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_h5), exist_ok=True)

# Write datasets and copy metadata
with h5py.File(output_h5, "w") as dst:
    # Create fixed-length datasets for tokens and masks
    dst.create_dataset("tokens", data=tokens, dtype=np.int32)
    dst.create_dataset("masks", data=masks, dtype=np.int32)
    src.copy("alphabet", dst)
    src.copy("input_metadata", dst)

src.close()
print(f"Wrote GRPO dataset to {output_h5}")
