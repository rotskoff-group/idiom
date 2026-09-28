# Example data

See the [cookbook guide](../README.md#example-data) for input conventions and SAE target selection.

| Directory | Contents |
|---|---|
| `cookbook/example_data/protgps/full_length/` | Full-length ProtGPS proteins, without IDR annotations |
| `cookbook/example_data/protgps/idrs/` | Complete paper IDR sets for six subcellular compartments |
| `cookbook/example_data/effector/` | Experimentally measured activation and repression domain IDRs |
| `cookbook/example_data/disprot/` | Held-out proteins with annotated IDRs and flanking context |
| [`cookbook/example_data/sae_features/`](sae_features/) | Enriched top-30 and original private-30 SAE targets |
| `cookbook/example_data/prompted_grpo/` | HP1α (P45973), IDR residues 79–123 |
| `cookbook/example_data/finches/` | ProTalpha and H1.0 C-terminal reference constructs |

The [IDiom-DB dataset](https://huggingface.co/datasets/jxliu2/idiom-db) is hosted on Hugging Face. These example FASTAs are bundled in this repository.

Annotated FASTA headers use `_IDR_x-y` (1-based, inclusive) to denote the IDR span. ProtGPS sequences in `protgps/idrs/` and transcriptional effector sequences are isolated
IDRs. DisProt sequences include flanks and may have repeat accessions with different spans. For more information on the IDR indexing conventions, see [sequence conventions](../../README.md#sequence-conventions).

Sources: [Kilgore et al.](https://www.science.org/doi/10.1126/science.adq2634),
[DelRosso et al.](https://www.nature.com/articles/s41586-023-05906-y),
[DisProt](https://academic.oup.com/nar/article/54/D1/D383/8325584), and
[Ginell et al.](https://doi.org/10.1126/science.adq8381).

<!-- ProtGPS `full_length/` contains proteins labeled only for each of the six example compartments among the 12 ProtGPS classes. `idrs/` contains the complete corresponding disjoint IDR sets used in the paper, extracted with Metapredict and filtered to 31–1,019 canonical amino-acid residues. Full-length FASTAs use ordinary headers without IDR coordinates. -->

<!-- IDR files use `<compartment>_idrs.fasta`; full-length files use `<compartment>.fasta`. Full-length source sequences are preserved, including a small number with noncanonical residues that the notebook input readers skip. -->
