# Single-protein prompted GRPO

`P06748.fasta` is one unchanged record copied from
[`../disprot/disprot_len1020_idrs.fasta`](../disprot/disprot_len1020_idrs.fasta), with the same
DisProt provenance and CC BY 4.0 attribution described in the [example data notes](../README.md).
It contains the full 294-residue protein and the header `P06748_IDR_119-259`.
The span is 1-based inclusive: the original IDR is 141 residues long.

Use [`prompted_sae.bash`](../../scripts/grpo/prompted_sae.bash) to optimize replacement IDRs
toward the shipped nucleolus/top30 SAE signature. Edit `REPO` and `OUT`, activate the IDiom
environment, and run:

```bash
bash cookbook/scripts/grpo/prompted_sae.bash --cfg job  # inspect configuration
bash cookbook/scripts/grpo/prompted_sae.bash
```

To use another protein, replace `FASTA` with a single-record FASTA containing its full sequence
and a header ending in `_IDR_x-y`. Set `TARGET_LENGTH` to `y - x + 1`. The length reward is a
soft preference, not an exact-length constraint. Ensure the flanks plus generation budget and
FIM/start markers fit the model context; the sampler clamps generation to the remaining space.

The model sees residues 1–118 and 260–294 as fixed context. The native IDR is omitted from the
prompt; training optimizes the policy's generated replacements rather than directly editing
the input FASTA. Rewards and metapredict score the replacement IDR alone. `prompts.n_per=1000`
repeats this same flank prompt; with one record, all prompts have equal length.

The script saves a final model checkpoint and prints example IDRs. To generate full redesigned
proteins afterward, pass that checkpoint and the same FASTA to:

```bash
idiom_generate prompted --model /path/to/final.ckpt \
    --fasta cookbook/example_data/prompted_grpo/P06748.fasta \
    --out redesigned.fasta --n 32 --return-full --max-new-tokens 256
```

`--return-full` splices replacements into the original flanks and updates the IDR coordinates.
