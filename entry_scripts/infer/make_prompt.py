"""
Make prompts from specific proteins 
"""

#%%
import numpy as np
import h5py
import os
import pickle
from idr_plm.nn.transformer.tokenizer import CharTokenizer

#%% Obtain alphabet 
data_dir = "/data2/scratch/group_scratch/idr_plm/2026-01-23_infer/prompts"

# shard = '/oak/stanford/groups/rotskoff/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_50/AFDB_IDR_50_FIM_512/AFDB_IDR_50_FIM_512_parts/precompute_shards/0001_file.h5'
shard = '/data2/scratch/group_scratch/idr_plm/rsync_4080/AFDB_IDR_90_FIM_512_splits_parts/precompute_shards/0001_file.h5'
precomputed_shard = h5py.File(shard, 'r')
alphabet = precomputed_shard['alphabet'][:]
alphabet = [x.decode('utf-8') for x in alphabet]
# idx = alphabet.index('[*]')
idx = alphabet.index('2')
print(idx)

#%% Define parameters

acc = "IDP_prompt"
# RNG_SEED = 42

#%% Create protein-based prompts with 1000 duplicates

# Set parameters for protein prompts
num_duplicates = 10000
header = f"{acc.lower()}_prompt_1e4x"
array_filename = f"{header}_array.pkl"
metadata_filename = f"{header}_metadata.pkl"

# Create 50% '132' and 50% '312' prompts
# num_each = num_duplicates // 2
# prompts = ['132'] * num_each + ['312'] * num_each
prompts = ['132'] * num_duplicates

# Shuffle to randomize order
# random.seed(RNG_SEED)
# random.shuffle(prompts)

# Create corresponding metadata for each prompt (simplified for IDP prompts)
metadata_list = []
for prompt in prompts:
    # Create metadata tuple with None for unused fields
    metadata = (acc, 0, None, None, None, None, prompt, prompt)
    metadata_list.append(metadata)

# print(f"Created {len(prompts)} total prompts (50% '132', 50% '312')")
# print(f"Count of '132': {prompts.count('132')}, Count of '312': {prompts.count('312')}")

#%% Tokenize protein prompts

# tokenizer = BasicSmilesTokenizer()
tokenizer = CharTokenizer() # For sequences 

# Tokenize all prompts and convert to numpy arrays
prompt_arrays = []
for p in prompts:
    tokens = tokenizer.tokenize(p)
    indices = [alphabet.index(x) for x in tokens]  # Token indices
    prompt_arrays.append(np.array(indices, dtype=np.int32))

prompt_array = prompt_arrays 

#%% Save protein prompts 

# Create directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Save prompt_array separately
with open(os.path.join(data_dir, array_filename), 'wb') as f:
    pickle.dump(prompt_array, f)

# Save other variables together
with open(os.path.join(data_dir, metadata_filename), 'wb') as f:
    data_to_dump = {
        'prompts': prompts,
        'metadata_list': metadata_list,
    }
    pickle.dump(data_to_dump, f)

print(f"Saved protein prompts to {data_dir}")
print(f"Array file: {array_filename}")
print(f"Metadata file: {metadata_filename}")

#%% View protein prompts

# Load the saved protein data
print("Loading saved protein data...")

# Load prompt_array
with open(os.path.join(data_dir, array_filename), 'rb') as f:
    loaded_prompt_array = pickle.load(f)

# Load metadata
with open(os.path.join(data_dir, metadata_filename), 'rb') as f:
    loaded_data = pickle.load(f)

# Extract variables from loaded data
loaded_prompts = loaded_data['prompts']
loaded_metadata_list = loaded_data['metadata_list']

print(f"\nLoaded protein data summary:")
print(f"- Total prompts: {len(loaded_prompts)}")
print(f"- Total metadata entries: {len(loaded_metadata_list)}")
print(f"- Prompt array shape: {len(loaded_prompt_array)}")

print(f"\nFirst 5 prompts and metadata:")
for i in range(min(5, len(loaded_prompts))):
    metadata = loaded_metadata_list[i]
    acc, region_idx, start, end, idr_seq, full_seq, fim_seq, fim_prompt = metadata
    print(f"{i+1}. Protein ID: {acc}, Region: {region_idx}, IDR position: {start}-{end}")
    print(f"   Prompt: {loaded_prompts[i]}")
    print(f"   IDR sequence: {idr_seq}")
    print()

print(f"\nFirst 3 tokenized prompts:")
for i in range(min(3, len(loaded_prompt_array))):
    print(f"Prompt {i+1}: {loaded_prompts[i]}")
    print(f"Tokenized: {loaded_prompt_array[i]}")
    print()

# Show alphabet information
print(f"\nAlphabet information:")
print(f"Alphabet size: {len(alphabet)}")
print(f"First 20 alphabet chars: {alphabet[:20]}")
print(f"Sentinel positions - '1': {alphabet.index('1')}, '2': {alphabet.index('2')}, '3': {alphabet.index('3')}")
print(f"Sample amino acids - A: {alphabet.index('A')}, G: {alphabet.index('G')}, P: {alphabet.index('P')}")

# Verify all prompts are identical
print(f"\nVerifying all prompts are identical:")
unique_prompts_count = len(set(loaded_prompts))
print(f"Number of unique prompts: {unique_prompts_count}")
if unique_prompts_count == 1:
    print("✓ All prompts are identical as expected")
else:
    print("⚠ Warning: Found multiple unique prompts")

