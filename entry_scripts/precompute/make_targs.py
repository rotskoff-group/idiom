import os
import h5py
import numpy as np

# Loop through all files ending with _idrs.h5
for idrs_filename in os.listdir("."):
    if idrs_filename.endswith("_idrs.h5"):
        with h5py.File(idrs_filename, "r") as idrs_file:
            if "idrs" not in idrs_file:
                print(f"'idrs' dataset not found in {idrs_filename}, skipping.")
                continue

            idrs_length = len(idrs_file["idrs"])

        # Generate matching _targs.h5 filename
        targs_filename = idrs_filename.replace("_idrs.h5", "_targs.h5")

        # Write the zero-filled targets dataset
        with h5py.File(targs_filename, "w") as targs_file:
            targs_file.create_dataset(
                "targets", data=np.zeros(idrs_length, dtype="float32")
            )

        print(f"Created {targs_filename} with {idrs_length} zeros.")

os.mkdir("precompute_shards")
