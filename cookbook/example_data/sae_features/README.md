# SAE target signatures

`sae_signatures.json` contains feature IDs for `jxliu2/idiomsae-300M-L18-k32`:

- `top30`: the 30 highest-ranked enriched features per sequence set.
- `private30`: the top 30 private features per sequence set used in the original
  private-feature experiments. Each list is copied verbatim, including order,
  from `topN30` in the archived `rl_targets_specific_lm.json`.

Selecting the top 30 private features differs from removing shared features
from an ordinary top-30 list. The latter can leave fewer than 30 features and
does not reproduce the private-30 experiment.

Both cases include eight sequence sets. The paper's six main compartments are
`nucleolus`, `chromosome`, `nuclear_speckle`, `stress_granule`, `p-body`, and
`nuclear_pore_complex`; the file also retains `pml_body` and
`post_synaptic_density` from the original targets.

To use the private targets, set the SAE reward term in
`cookbook/scripts/training/grpo/sae_features.yaml` to:

```yaml
name: sae_signature
signature: nucleolus
features: ${oc.env:FEATURES}
case: private30
sae: jxliu2/idiomsae-300M-L18-k32
device: null
```

The Bash recipe already points `FEATURES` to this JSON. Selecting this case
reproduces the feature targets; the remaining training and generation settings
must also match the experiment to reproduce its results.

The JSON's `_provenance.private30_source` records the archived source filename,
full-file SHA-256, and source key. The package default at
`src/idiom/train/grpo/reward/sae_signatures.json` and this cookbook copy must
remain byte-identical. On `jxliu2/idiom-db`, publish this JSON and README under
`other/example_data/sae_features/`.
