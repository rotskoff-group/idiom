# Prompted linker Rg redesign

`P45973.fasta` contains full-length HP1α (CBX5), 191 residues, with the annotated
45-residue hinge/linker marked `P45973_IDR_79-123` (1-based inclusive). It is an unchanged
record copied from [`../disprot/disprot_len1020_idrs.fasta`](../disprot/disprot_len1020_idrs.fasta);
see the [example data notes](../README.md) for DisProt provenance and CC BY 4.0 attribution.
HP1α has a hinge connecting its chromodomain and chromoshadow domain
([domain architecture](https://pmc.ncbi.nlm.nih.gov/articles/PMC3365711/)).

Use [`prompted_linker_rg.bash`](../../scripts/grpo/prompted_linker_rg.bash) to redesign the linker
using residues 1–78 and 124–191 as fixed generation context. Edit `REPO` and `OUT`, activate
the IDiom environment, and run:

```bash
bash cookbook/scripts/grpo/prompted_linker_rg.bash --cfg job  # inspect configuration
bash cookbook/scripts/grpo/prompted_linker_rg.bash
```

The objective combines:

- ALBATROSS-predicted linker Rg, via the existing isolated sparrow scorer: illustrative target
  `TARGET_RG=25` Å, quadratic fractional width 0.2, weight 0.5.
- Linker length: target 45 residues, quadratic fractional width 0.1, weight 1. This discourages
  satisfying Rg by simply changing length; it does not enforce exactly 45 residues.
- Composition entropy: target 3.65, quadratic fractional width 0.2, weight 1.

Metapredict V3 disorder is logged every optimizer step as a diagnostic. Both Rg and disorder
are scored on the generated linker alone. ALBATROSS predicts isolated-chain dimensions;
it does not model the attached domains, their separation, or full-protein Rg. The 25 Å target
is an editable demonstration setting, not a calibrated native value or a guarantee of function.
See the [ALBATROSS paper](https://www.nature.com/articles/s41592-023-02159-5) and the
[scorer setup guide](../../rewards/README.md).

To use another linker, set `FASTA` to a single-record FASTA containing the full protein and a
header ending in `_IDR_x-y`. Set `TARGET_LENGTH` to `y - x + 1` and choose `TARGET_RG`.
The native linker is omitted from the prompt. Training optimizes a policy that generates
replacement linkers; it does not directly edit the input FASTA. `prompts.n_per=1000` repeats
this same flank prompt, so all prompts have equal length. Flanks, generated residues, and
FIM/start markers must fit the model context; generation is capped at 96 new tokens here.

The script saves a final model checkpoint and prints example linkers. Generate full redesigned
proteins afterward with that checkpoint and the same input FASTA:

```bash
idiom_generate prompted --model /path/to/final.ckpt \
    --fasta cookbook/example_data/prompted_grpo/P45973.fasta \
    --out redesigned.fasta --n 32 --return-full --max-new-tokens 96
```

`--return-full` splices replacements into the original flanks and updates the IDR coordinates.
