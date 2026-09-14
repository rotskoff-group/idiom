# Example data

See the [cookbook](../README.md#example-data) for details.

| Directory | Contents |
|---|---|
| `cookbook/example_data/protgps/` | IDRs associated with six subcellular compartments |
| `cookbook/example_data/effector/` | Experimentally measured activation and repression domain IDRs |
| `cookbook/example_data/disprot/` | Held-out proteins with annotated IDRs and flanking context |
| `cookbook/example_data/sae_features/` | Example SAE signatures |
| `cookbook/example_data/prompted_grpo/` | HP1α (P45973), IDR residues 79–123 |
| `cookbook/example_data/finches/` | ProTalpha and H1.0 C-terminal reference constructs |


These data are also available on [Hugging Face](https://huggingface.co/datasets/jxliu2/idiom-db).

FASTA headers use `_IDR_x-y` (1-based, inclusive) to denote the IDR span. ProtGPS and transcriptional effector sequences are isolated
IDRs. DisProt sequences include flanks and may have repeat accessions with different spans. For more information on the IDR indexing conventions, see [sequence conventions](../../README.md#sequence-conventions).

Sources: [Kilgore et al.](https://www.science.org/doi/10.1126/science.adq2634),
[DelRosso et al.](https://www.nature.com/articles/s41586-023-05906-y),
[DisProt](https://academic.oup.com/nar/article/54/D1/D383/8325584), and
[Ginell et al.](https://doi.org/10.1126/science.adq8381).
