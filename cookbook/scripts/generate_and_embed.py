"""Generate and embed IDRs.

IDiom generates intrinsically disordered regions two ways -- unprompted (de novo, no context) and
prompted (in-filling an IDR between its flanks) -- and exposes the residual stream as embeddings
for downstream models. This script covers both.

    uv run cookbook/scripts/generate_and_embed.py

A GPU is recommended; DEVICE = "auto" falls back to CPU. Edit the parameters below to point at
another model or layer.
"""

import tempfile
from pathlib import Path

from idiom import IDiom

MODEL = "jxliu2/idiom-300M"   # HF repo id, a released directory, or a .ckpt
DEVICE = "auto"               # auto | cpu | cuda
LAYER = 18                    # residual-stream layer to read embeddings from
N = 10                        # sequences per example

model = IDiom.from_pretrained(MODEL, device=DEVICE)
print(model)


# Unprompted: de novo IDRs
# ------------------------
# No context at all -- the model samples from what it learned an IDR looks like.

idrs = model.generate_unprompted(n=N, temperature=1.0, seed=0)

print(f"\n{len(idrs)} IDRs, mean length {sum(map(len, idrs)) / len(idrs):.0f}")
for s in idrs[:3]:
    print(f"  {len(s):>4}  {s[:70]}{'...' if len(s) > 70 else ''}")

# Passing a length_range oversamples and then filters, so you get IDRs in the size range you asked
# for rather than the size the model felt like.

sized = model.generate_unprompted(n=N, length_range=(60, 100), seed=0)

print(f"\n{len(sized)} IDRs, lengths {sorted(len(s) for s in sized)}")


# Prompted: in-fill an IDR between its flanks
# -------------------------------------------
# Give the model a protein and the coordinates of the disordered span (0-based, half-open) and it
# generates replacements for that span, conditioned on the flanking sequence.

protein = "MEDSKVDNRPQACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRST"
idr_start, idr_end = 12, 30

filled = model.generate_prompted(protein, idr_start, idr_end, n=N, seed=0)

print(f"\nleft flank  ...{protein[:idr_start][-8:]}")
print(f"right flank {protein[idr_end:][:8]}...")
print(f"\n{len(filled)} in-fills:")
for s in filled[:3]:
    print(f"  {len(s):>4}  {s[:70]}")

# Both have a FASTA form for generating a set to disk -- generate_unprompted_fasta and
# generate_prompted_fasta. That output is what the SFT and enrichment workflows consume.

out = Path(tempfile.mkdtemp()) / "designed.fasta"
model.generate_unprompted_fasta(out, n=N, seed=0)
print(f"\nwrote {out}\n")
print(out.read_text()[:300])


# Embeddings
# ----------
# embed reads the residual stream at whichever layers you ask for. pool="mean" gives one vector per
# sequence -- the usual input to a downstream predictor.

SEQS = [
    "MEDSKVDNRPQACDEFGHIKLMNPQRSTVWY",
    "GSGSQPQPQPGSGSGSNNNNQQQQGSGSGS",
]

pooled, index = model.embed(SEQS, layers=[LAYER], pool="mean")[LAYER]

print(f"\npooled: {pooled.shape} -- one {pooled.shape[1]}-d vector per sequence")
print("accessions:", [r["accession"] for r in index])

# pool="none" keeps every residue as its own row. The index carries each row's source position, so
# rows stay aligned to residues after any filtering.

per_res, index = model.embed(SEQS[0], layers=[LAYER], pool="none")[LAYER]

print(f"\nper-residue: {per_res.shape} for a {len(SEQS[0])}-residue sequence")
print("first row:", {k: index[0][k] for k in ("accession", "source_pos", "residue")})


# Next
# ----
# - sae_features.py -- read and steer the features behind these embeddings with a sparse
#   autoencoder.
# - cookbook/slurm/ -- pretraining, SFT, and RL templates for real runs.
