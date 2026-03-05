#!/home/groups/ardunn/jxliu2/MoLE/.venv/bin/python
# source /home/groups/ardunn/jxliu2/MoLE/.venv/bin/activate # Sherlock

"""
Make prompts from specific proteins
"""

# %%
import numpy as np
import h5py
import os
import pickle
from idr_plm_figures.utils.tokenizer import CharTokenizer
import random


# Skip DisProt loading - using specific protein sequence instead


# %% Obtain alphabet

shard = "/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_90/AFDB_IDR_90_FIM_512/AFDB_IDR_90_FIM_512_parts/precompute_shards/0001_file.h5"
precomputed_shard = h5py.File(shard, "r")
alphabet = precomputed_shard["alphabet"][:]
alphabet = [x.decode("utf-8") for x in alphabet]
# idx = alphabet.index('[*]')
idx = alphabet.index("2")
print(idx)


# %% Define specific prompt protein sequence

# %% Define protein sequence and IDR region hnRNPA1 (MACAQUE!)
# # https://www.uniprot.org/uniprotkb/Q28521/entry#sequences
# # https://alphafold.ebi.ac.uk/entry/Q28521
# acc = "Q28521" # This is the macaque protein not human oops I'm a macaque
# og_sequence = "MSKSESPKEPEQLRKLFIGGLSFETTDESLRSHFEQWGTLTDCVVMRDPNTKRSRGFGFVTYATVEKVDAAMNARPHKVDGRVVEPKRAVSREDSQRPGAHLTVKKIFVGGIKEDTEEHHLRDYFEQYGKIEVIEIMTDRGSGKKRGFAFVTFDDHNSVDKIVIQKYHTVNGHNCEVRKALSKQEMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSLGPYGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF"
# idr_start = 186 # 186 matches the A1-LCD of 'valence and patterning'
# # IDR: MASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSLGPYGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF
# idr_end = 320 # This is should be the last AA

# %% Define protein sequence and IDR region hnRNPA1 (human)
# # https://www.uniprot.org/uniprotkb/P09651/entry#sequences
# # https://alphafold.ebi.ac.uk/entry/P09651
# acc = "P09651" # human
# og_sequence = "MSKSESPKEPEQLRKLFIGGLSFETTDESLRSHFEQWGTLTDCVVMRDPNTKRSRGFGFVTYATVEEVDAAMNARPHKVDGRVVEPKRAVSREDSQRPGAHLTVKKIFVGGIKEDTEEHHLRDYFEQYGKIEVIEIMTDRGSGKKRGFAFVTFDDHDSVDKIVIQKYHTVNGHNCEVRKALSKQEMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGGYGGGGPGYSGGSRGYGSGGQGYGNQGSGYGGSGSYDSYNNGGGGGFGGGSGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSGPYGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF"
# idr_start = 186 # 186 matches the A1-LCD of 'valence and patterning'
# # IDR: MASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGGYGGGGPGYSGGSRGYGSGGQGYGNQGSGYGGSGSYDSYNNGGGGGFGGGSGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSGPYGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF
# idr_end = 372 # This is should be the last AA

# %% Define protein sequence and IDR region hnRNPA1 (human)
# # https://www.uniprot.org/uniprotkb/P09651/entry#sequences
# # https://alphafold.ebi.ac.uk/entry/P09651
# acc = "P09651-2" # human, most common isoform by presence (not canonical isoform)
# # This is the one whose IDR is ~135 long
# og_sequence = "MSKSESPKEPEQLRKLFIGGLSFETTDESLRSHFEQWGTLTDCVVMRDPNTKRSRGFGFVTYATVEEVDAAMNARPHKVDGRVVEPKRAVSREDSQRPGAHLTVKKIFVGGIKEDTEEHHLRDYFEQYGKIEVIEIMTDRGSGKKRGFAFVTFDDHDSVDKIVIQKYHTVNGHNCEVRKALSKQEMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSGPYGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF"
# idr_start = 186 # 186 matches the A1-LCD of 'valence and patterning'
# idr_end = 320 # This is should be the last AA


# %% NPM1
# https://www.uniprot.org/uniprotkb/P06748/entry
# https://alphafold.ebi.ac.uk/entry/P06748
# https://www.nature.com/articles/s41467-018-03255-3
acc = "P06748"
og_sequence = "MEDSMDMDMSPLRPQNYLFGCELKADKDYHFKVDNDENEHQLSLRTVSLGAGAKDELHIVEAEAMNYEGSPIKVTLATLKMSVQPTVSLGGFEITPPVVLRLKCGSGPVHISGQHLVAVEEDAESEDEEEEDVKLLSISGKRSAPGGGSKVPQKKVKLAADEDDDDDDEEDDDEDDDDDDFDDEEAEEKAPVKKSIRDTPAKNAQKSNQNGKDSKPSSTPRSKGQESFKKQEKTPKTPKGPSSVEDIKAKMQASIEKGGSLPKVEAKFINYVKNCFRMTDQEAIQDLWQWRKSL"
idr_start = 119
idr_end = 242  # 242 matches AFDB and Nat Comms reference better

# %% CX43
# # https://www.uniprot.org/uniprotkb/P17302/entry
# # https://alphafold.ebi.ac.uk/entry/P17302
# # https://www.cell.com/current-biology/fulltext/S0960-9822(07)00375-2?script=true
# acc = "P17302"
# og_sequence = "MGDWSALGKLLDKVQAYSTAGGKVWLSVLFIFRILLLGTAVESAWGDEQSAFRCNTQQPGCENVCYDKSFPISHVRFWVLQIIFVSVPTLLYLAHVFYVMRKEEKLNKKEEELKVAQTDGVNVDMHLKQIEIKKFKYGIEEHGKVKMRGGLLRTYIISILFKSIFEVAFLLIQWYIYGFSLSAVYTCKRDPCPHQVDCFLSRPTEKTIFIIFMLVVSLVSLALNIIELFYVFFKGVKDRVKGKSDPYHATSGALSPAKDCGSQKYAYFNGCSSPTAPLSPMSPPGYKLVTGDRNNSSCRNYNKQASEQNWANYSAEQNRMGQAGSTISNSHAQPFDFPDDNQNSKKLAAGHELQPLAIVDQRPSSRASSRASSRPRPDDLEI"
# idr_start = 234 # https://www.mdpi.com/1422-0067/19/5/1428
# idr_end = 382 # C-terminal
# # ~IDR: GVKDRVKGKSDPYHATSGALSPAKDCGSQKYAYFNGCSSPTAPLSPMSPPGYKLVTGDRNNSSCRNYNKQASEQNWANYSAEQNRMGQAGSTISNSHAQPFDFPDDNQNSKKLAAGHELQPLAIVDQRPSSRASSRASSRPRPDDLEI

# %% p53
# # https://www.uniprot.org/uniprotkb/P04637/entry#P04637-1
# # https://alphafold.ebi.ac.uk/entry/AF-A0A2R9A5P4-F1 - exact match to homo sapiens sequence
# # http://elm.eu.org/combined_search?query=P04637
# acc = "P04637"
# og_sequence = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
# idr_start = 1 #
# idr_end = 95 #
# # ~IDR: MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSS


# %% Create FIM prompts for selected protein

# Constants and function for FIM transform based on make_AFDB_FIM.py
SENTINELS = {"prefix": "1", "middle": "2", "suffix": "3"}
RNG_SEED = 42


def fim_transform(seq: str, start: int, end: int) -> str:
    prefix, middle, suffix = seq[:start], seq[start : end + 1], seq[end + 1 :]
    return f"{SENTINELS['prefix']}{prefix}{SENTINELS['suffix']}{suffix}{SENTINELS['middle']}{middle}"


# Create FIM data for selected protein only
random.seed(RNG_SEED)
idr_seq = og_sequence[
    idr_start - 1 : idr_end
]  # Extract IDR sequence (convert to 0-indexed)
fim_seq = fim_transform(og_sequence, idr_start - 1, idr_end - 1)  # Create FIM sequence

# Create prompt by slicing up to and including the '2' sentinel
mid_idx = fim_seq.find(SENTINELS["middle"])
if mid_idx != -1:
    fim_prompt = fim_seq[: mid_idx + 1]
else:
    fim_prompt = fim_seq

# Store as single entry (matching original format)
protein_fim_data = (
    acc,
    0,
    idr_start,
    idr_end,
    idr_seq,
    og_sequence,
    fim_seq,
    fim_prompt,
)

print(f"Created FIM data for protein {acc}")
print(f"IDR sequence: {idr_seq}")
print(f"FIM prompt: {fim_prompt}")


# %% Create protein-based prompts with 1000 duplicates

# Set parameters for protein prompts
num_duplicates = 100000
header = f"{acc.lower()}_prompt_1e5x"
array_filename = f"{header}_array.pkl"
metadata_filename = f"{header}_metadata.pkl"

# Extract the single FIM prompt from protein data
fim_prompt = protein_fim_data[7]  # The fim_prompt is the 8th element (index 7)

# Only add if it's not simply '132' or '312'
if fim_prompt not in ["132", "312"]:
    print(f"Valid protein FIM prompt: {fim_prompt}")

    # Create 1000 duplicates of the same prompt and metadata
    prompts = [fim_prompt] * num_duplicates
    metadata_list = [protein_fim_data] * num_duplicates

    print(
        f"Created {len(prompts)} total prompts (1 unique protein prompt × {num_duplicates} duplicates)"
    )
else:
    print(f"Invalid prompt: {fim_prompt}")
    prompts = []
    metadata_list = []


# %% Tokenize protein prompts

# tokenizer = BasicSmilesTokenizer()
tokenizer = CharTokenizer()  # For sequences

# Tokenize all prompts and convert to numpy arrays
prompt_arrays = []
for p in prompts:
    tokens = tokenizer.tokenize(p)
    indices = [alphabet.index(x) for x in tokens]  # Token indices
    prompt_arrays.append(np.array(indices, dtype=np.int32))

prompt_array = prompt_arrays


# %% Save protein prompts
data_dir = "../prompts"

# Create directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Save prompt_array separately
with open(os.path.join(data_dir, array_filename), "wb") as f:
    pickle.dump(prompt_array, f)

# Save other variables together
with open(os.path.join(data_dir, metadata_filename), "wb") as f:
    data_to_dump = {
        "prompts": prompts,
        "metadata_list": metadata_list,
    }
    pickle.dump(data_to_dump, f)

print(f"Saved protein prompts to {data_dir}")
print(f"Array file: {array_filename}")
print(f"Metadata file: {metadata_filename}")


# %% View protein prompts

# Load the saved protein data
print("Loading saved protein data...")

# Load prompt_array
with open(os.path.join(data_dir, array_filename), "rb") as f:
    loaded_prompt_array = pickle.load(f)

# Load metadata
with open(os.path.join(data_dir, metadata_filename), "rb") as f:
    loaded_data = pickle.load(f)

# Extract variables from loaded data
loaded_prompts = loaded_data["prompts"]
loaded_metadata_list = loaded_data["metadata_list"]

print("\nLoaded protein data summary:")
print(f"- Total prompts: {len(loaded_prompts)}")
print(f"- Total metadata entries: {len(loaded_metadata_list)}")
print(f"- Prompt array shape: {len(loaded_prompt_array)}")

print("\nFirst 5 prompts and metadata:")
for i in range(min(5, len(loaded_prompts))):
    metadata = loaded_metadata_list[i]
    acc, region_idx, start, end, idr_seq, full_seq, fim_seq, fim_prompt = metadata
    print(
        f"{i + 1}. Protein ID: {acc}, Region: {region_idx}, IDR position: {start}-{end}"
    )
    print(f"   Prompt: {loaded_prompts[i]}")
    print(f"   IDR sequence: {idr_seq}")
    print()

print("\nFirst 3 tokenized prompts:")
for i in range(min(3, len(loaded_prompt_array))):
    print(f"Prompt {i + 1}: {loaded_prompts[i]}")
    print(f"Tokenized: {loaded_prompt_array[i]}")
    print()

# Show alphabet information
print("\nAlphabet information:")
print(f"Alphabet size: {len(alphabet)}")
print(f"First 20 alphabet chars: {alphabet[:20]}")
print(
    f"Sentinel positions - '1': {alphabet.index('1')}, '2': {alphabet.index('2')}, '3': {alphabet.index('3')}"
)
print(
    f"Sample amino acids - A: {alphabet.index('A')}, G: {alphabet.index('G')}, P: {alphabet.index('P')}"
)

# Verify all prompts are identical
print("\nVerifying all prompts are identical:")
unique_prompts_count = len(set(loaded_prompts))
print(f"Number of unique prompts: {unique_prompts_count}")
if unique_prompts_count == 1:
    print("✓ All prompts are identical as expected")
else:
    print("⚠ Warning: Found multiple unique prompts")
