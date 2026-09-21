# IDiom notebooks

Seven independent workflows, ordered from generation to post-training. Open any notebook
in Colab and run it from top to bottom; no prior notebook or repository checkout is required.

| Notebook | Goal | Colab |
|---|---|---|
| [01 · Generate IDRs](01_generate_idrs.ipynb) | Sample standalone IDRs and redesign an IDR between protein flanks | [Open](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/01_generate_idrs.ipynb) |
| [02 · Explore embeddings](02_explore_embeddings.ipynb) | Extract representations, find similar sequences, and visualize a projection | [Open](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/02_explore_embeddings.ipynb) |
| [03 · Interpret SAE features](03_interpret_sae_features.ipynb) | Rank features, inspect residue traces and logos, and reopen saved activations | [Open](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/03_interpret_sae_features.ipynb) |
| [04 · Discover a feature signature](04_discover_feature_signature.ipynb) | Compare positives and background, inspect enrichment, and export targets | [Open](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/04_discover_feature_signature.ipynb) |
| [05 · Fine-tune and generate](05_finetune_and_generate.ipynb) | Train on your sequences, reload the model, and compare samples | [Open](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/05_finetune_and_generate.ipynb) |
| [06 · Design with custom rewards](06_design_with_custom_rewards.ipynb) | Run GRPO with a transparent sequence-property objective | [Open](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/06_design_with_custom_rewards.ipynb) |
| [07 · Design with RL-SAE](07_design_with_rl_sae.ipynb) | Optimize a feature signature or a union of signatures and evaluate coverage | [Open](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/07_design_with_rl_sae.ipynb) |

## Setup and your own data

Select a GPU runtime in Colab. The setup cell installs `idiom[cookbook]` from the `v1`
release when IDiom is absent. Demo FASTAs come from that same release. An older installed
IDiom must be upgraded and the kernel restarted. Locally, install the cookbook extra and
open the notebook with Jupyter. Shared input and comparison helpers are installed as
`idiom.utils.notebook_helpers`; there is no separately downloaded Python helper.

Each notebook has one input/settings cell. Upload a FASTA using Colab's Files pane or set
an absolute local/Drive path. `INPUT_MODE="idr"` accepts isolated IDRs with ordinary
headers. `"annotated"` requires full proteins with `_IDR_x-y` headers using **1-based,
inclusive** coordinates. Python APIs use **0-based, end-exclusive** coordinates.
These workflows do not predict IDR boundaries. Invalid records are audited, not repaired.
Repeated accessions receive distinct record IDs; audits preserve the original names.

For Drive persistence, mount Drive and set `OUT_DIR` to a directory there. Otherwise,
the final cell creates a ZIP and offers a Colab download. Use a new output directory for
each experiment. Training notebooks accept `RESUME_FROM` and save optimizer checkpoints
as well as reloadable model releases. Archives containing model checkpoints can be large.

## Hardware and scale

Generation, embeddings, SFT, and custom-reward examples default to the 20M model. SAE
workflows use the released 300M host and its matching SAE. Small inference examples can
run on CPU. Training defaults are short demonstrations, not convergence recipes.
RL-SAE defaults to a CPU reward lens to reduce GPU memory pressure, at a speed cost.
A free Colab GPU is not guaranteed to fit every training configuration.

Reduce sequence counts, batch size, and generation length when needed. The feature builder
retains sparse output batches in host RAM. Embedding neighbor comparisons and string-similarity
checks are intended for small sets. Runtime and memory depend on the chosen hardware and inputs;
fresh-kernel CPU tests with tiny models do not establish full-model Colab performance.

## Saved outputs and handoffs

- **01:** generated and redesigned FASTAs, candidate metrics, input audit, plots, and settings.
- **02:** embeddings, row metadata, nearest neighbors, PCA coordinates, and residue embeddings.
- **03:** sparse activations, feature rankings, traces, aligned windows, logo data, and figures.
  Set `FEATURE_DIR` to reopen a dataset from the dataset CLI or notebook 04 without inference.
- **04:** selected input FASTAs and audits, `fd_positive/`, `fd_background/`, `enrichment.npz`,
  `enrichment.tsv`, logo outputs, and optional `signature.json`. Set `RESULT_DIR` to reopen a
  notebook or enrichment CLI run. The NPZ stores numerical statistics and selection diagnostics.
- **05–07:** baseline and adapted FASTAs, comparison tables, training config and CSV logs,
  `training/checkpoints/last.ckpt`, and `model/` for `IDiom.from_pretrained`.
- **07:** also exports the chosen signature, component coverage, and target-feature presence.

Notebook 04's signature can be uploaded to notebook 07. Notebook 07 also includes a bundled
target so it runs independently. Feature IDs belong to a particular SAE: retain its identity
and model revision with your outputs. Legacy signature files without identity remain readable;
the caller is responsible for pairing them with the correct SAE.

Enrichment tests association with the supplied background, not biological function. Training
and steering scores are not experimental validation. SFT's default split separates exact duplicate
IDRs but does not separate homologs. Use clustered splits and independent evaluation for substantive studies.

Pretraining, SAE training, larger runs, and external predictor integrations remain in
[the script cookbook](../scripts/README.md).
