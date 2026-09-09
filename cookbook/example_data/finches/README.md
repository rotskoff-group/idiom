# ProTalpha–H1.0 interaction demo

The two FASTAs reproduce `ProtA` (reference) and `h1_ctd` (fixed partner) from
[Ginell et al., Science 2025, Fig. 5](https://doi.org/10.1126/science.adq8381),
using the exact sequences in the authors' [analysis notebook](https://github.com/holehouse-lab/supportingdata/blob/master/2025/finches_2025/figures/figure_5/Fig_5B_C_D.ipynb).
These are the paper's constructs, not full-length H1.0 or redesigned sequences.

The [bash demo](../../scripts/grpo/finches.bash) generates a synthetic ProTalpha-like
IDR, using hard-coded approximate native targets: FINCHES epsilon -34.8 (Mpipi
at the frontend default salt concentration of 0.150 M), length 111 residues, and
composition entropy 3.14 bits. Recalculate epsilon if changing force field or salt. Disorder is logged as a diagnostic, not rewarded.
Matching epsilon is a demonstration of predicted interaction chemistry, not a
prediction of equal binding affinity or biological function.
