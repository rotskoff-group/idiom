import os
import h5py
import numpy as np

for suffix, dataset_key in [("_idrs.h5", "idrs"), ("_residues.h5", "residues")]:
    for src_filename in os.listdir("."):
        if not src_filename.endswith(suffix):
            continue
        with h5py.File(src_filename, "r") as src_file:
            if dataset_key not in src_file:
                print(f"'{dataset_key}' dataset not found in {src_filename}, skipping.")
                continue
            length = len(src_file[dataset_key])

        targs_filename = src_filename.replace(suffix, "_targs.h5")

        with h5py.File(targs_filename, "w") as targs_file:
            targs_file.create_dataset("targets", data=np.zeros(length, dtype="float32"))

        print(f"Created {targs_filename} with {length} zeros.")

os.mkdir("precompute_shards")
