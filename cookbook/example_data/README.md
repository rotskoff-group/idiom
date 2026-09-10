# Example data

| Directory | Contents |
|---|---|
| [protgps/](protgps) | IDRs with annocated association to six subcellular compartments |
| [effector/](effector) | Experimentally measured activation and repression domain IDRs |
| [disprot/](disprot) | Held-out proteins with annotated IDRs and flanking context |
| [sae_features/](sae_features) | Example SAE signatures |
| [prompted_grpo/](prompted_grpo) | HP1α (P45973), IDR residues 79–123 |
| [finches/](finches) | ProTalpha and H1.0 C-terminal sequences |

Cookbook inputs are included here and on [Hugging Face](https://huggingface.co/datasets/jxliu2/idiom-data).

FASTA headers use `_IDR_x-y` (1-based, inclusive) to denote the IDR span. ProtGPS and effector sequences are isolated
IDRs. DisProt sequences include flanks and may repeat accessions with different spans. For more information on the IDR indexing conventions, see [sequence conventions](../../README.md#sequence-conventions).

Sources: [ProtGPS](https://www.science.org/doi/10.1126/science.adq2634),
[DelRosso et al. (effector domains)](https://www.nature.com/articles/s41586-023-05906-y),
[DisProt](https://academic.oup.com/nar/article/54/D1/D383/8325584), and
[Ginell et al. (FINCHES)](https://doi.org/10.1126/science.adq8381).
