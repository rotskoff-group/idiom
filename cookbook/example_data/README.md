# Example data

Demo FASTAs and SAE signatures are available in [this directory](.) and on
[jxliu2/idiom-data](https://huggingface.co/datasets/jxliu2/idiom-data). Notebooks download their
inputs; Bash scripts use local paths.

| Directory | Contents |
|---|---|
| [protgps/](protgps/) | IDRs from six subcellular compartments |
| [effector/](effector/) | Activation and repression domain IDRs |
| [disprot/](disprot/) | Held-out proteins with annotated IDR spans and flanking context |
| [sae_features/](sae_features/) | Released SAE signatures as a format reference |
| [prompted_grpo/](prompted_grpo/) | One full-length protein with a marked IDR for prompted GRPO |
| [finches/](finches/) | ProTalpha and H1.0 C-terminal constructs from the FINCHES paper |

```bash
hf download jxliu2/idiom-data --repo-type dataset --include "example_data/*"
```

The download command prints its cache directory. Use that path for local inputs, or use the
copies under `cookbook/example_data/` in a repository clone.

The ProtGPS and effector records mark the whole sequence as the IDR. DisProt records include
flanks; preserve their annotated spans for prompted generation. A protein can have multiple
DisProt records with different IDR spans, so accessions need not be unique.
See [sequence conventions](../../README.md#sequence-conventions).

Sources: [ProtGPS (Kilgore et al.)](https://www.science.org/doi/10.1126/science.adq2634), [DelRosso et al., *Nature* 2023 (effector domains)](https://www.nature.com/articles/s41586-023-05906-y), and [DisProt](https://academic.oup.com/nar/article/54/D1/D383/8325584) (CC BY 4.0).
These are demonstration subsets; cite the original works and follow their licenses when reusing them.
