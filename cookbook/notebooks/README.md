# Notebooks

Start with **[Analyze your sequences](analyze_sequences.ipynb)**. Each notebook runs independently,
with a small demo and one configuration cell for your own data.

| Notebook | Workflow | Colab |
|---|---|---|
| [`analyze_sequences.ipynb`](analyze_sequences.ipynb) | Start here: validate your IDRs, embed, find neighbors, and export results | [Open](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/analyze_sequences.ipynb) |
| [`generate_sequences.ipynb`](generate_sequences.ipynb) | Generate de novo IDRs or replace an annotated protein region | [Open](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/generate_sequences.ipynb) |
| [`inspect_sae_features.ipynb`](inspect_sae_features.ipynb) | Rank features in your sequences and plot residue activations | [Open](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/inspect_sae_features.ipynb) |
| [`feature_enrichment.ipynb`](feature_enrichment.ipynb) | Compare feature prevalence with a length-matched background | [Open](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/feature_enrichment.ipynb) |
| [`compare_sequence_sets.ipynb`](compare_sequence_sets.ipynb) | Compare candidates and references; optionally score perplexity | [Open](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/compare_sequence_sets.ipynb) |
| [`steer_generation.ipynb`](steer_generation.ipynb) | Compare feature steering strengths with an unsteered baseline | [Open](https://colab.research.google.com/github/rotskoff-group/idiom/blob/main/cookbook/notebooks/steer_generation.ipynb) |

## Run locally or in Colab

Install IDiom from the repository root (`pip install -e .`) and a notebook environment
(`pip install jupyterlab pandas`). Launch `jupyter lab` from the repository root or this directory.
Keep `workflow_utils.py` beside the notebooks. In Colab, select a GPU runtime; the setup cell
installs missing dependencies and downloads the companion helper. Local edits to the notebooks
and helper must be published before the Colab links can use them.

The analysis and generation starters default to IDiom-20M; SAE workflows load the released
300M host model. A GPU is recommended; CPU runs are supported but slower. First runs download
weights. Analysis and inspection use six illustrative IDRs, comparison uses two groups of three,
and enrichment samples up to 128 positives and 512 background records. Enrichment downloads the
validation FASTA before sampling. These demo settings do not reproduce published signatures.

Each notebook prints measured elapsed time and saves it in `run.json`, alongside settings and
package versions. Timing includes model loading and analysis, excluding installation. Hardware
and sequence lengths strongly affect runtime; these are not fixed hardware benchmarks.
Reduce sample counts and generation/SAE batch sizes when needed. Generation uses a bounded token
budget and can terminate at that budget; inspect lengths before interpreting candidates. SAE
inspection and comparison use sparse feature datasets, whose construction still requires host RAM.

## Bring your own sequences

- **Isolated IDRs:** choose `"idr"` input mode. Ordinary FASTA headers work. If an `_IDR_x-y`
  suffix is present, it must span the whole sequence (`_IDR_1-L`).
- **Full proteins:** choose `"annotated"` and provide `_IDR_x-y` headers with 1-based inclusive spans.
  Invalid annotations are reported instead of being interpreted as whole-protein IDRs.
- Use uppercase canonical amino acids. Input audits list accepted/excluded records and reasons.
  A unique record ID maps every result back to the original header and accession, even when
  accessions repeat. No sequence is silently truncated to fit model context.

Analysis embeddings average isolated IDR residues by default; set `USE_FLANKS=True` to retain
annotated protein context. The released SAE always reads isolated IDRs. Sequence-set comparison extracts isolated IDRs from both groups
for consistent embeddings and optional likelihood scoring. Perplexity is an aggregate,
token-weighted diagnostic, not a biological quality score.

## Outputs and next steps

Each workflow saves plots and/or tables, input audits, and `run.json` in its configured output
directory. Reusing that directory replaces named files; use a fresh directory per analysis so
older optional exports cannot be mistaken for current results. The notebooks print the output
path; in Colab, download results using the Files panel before disconnecting.

Generation writes IDR-only FASTAs for analysis, comparison, and enrichment, plus redesigned full
proteins with updated spans. Inspection can guide feature selection for steering. Enrichment
keeps signature export and the GRPO handoff optional; a small demo may find no significant features.

`generate_and_embed.ipynb` and `sae_features.ipynb` remain as navigation pages for older links.
Their workflows now live in the focused notebooks above.
See the [cookbook](../README.md) for batch scripts and training workflows.

## Files you can reuse

| Workflow | Main exports |
|---|---|
| Analyze | `sequence_summary.csv`, `embeddings.npy`, `embedding_index.csv`, `nearest_neighbors.csv`, PCA figure and coordinates |
| Generate | `idrs.fasta`, `redesigned_idrs.fasta`, `redesigned_proteins.fasta`, `original_idr.fasta`, `candidates.csv` |
| Inspect | `features/`, `sequence_index.csv`, `feature_ranking.csv`, `residue_traces.csv`, activation figures |
| Enrichment | Selected FASTAs and record indices, input audits, `enrichment.csv`, logos, optional `signature.json` |
| Compare | Sequence summaries, composition figures, `nearest_references.csv`, optional feature prevalence and aggregate perplexity |
| Steer | One FASTA per strength, `candidates.csv` with measured activations, sequence index, comparison figure |

Record IDs join result tables to input audits; the original accession and header are preserved there.
FASTA annotations are 1-based inclusive; exported residue plots use original protein coordinates.
The demo sequences illustrate the workflow and carry no experimental functional labels.
