# IDiom notebooks

| Notebook | Colab |
|---|---|
| [01 · Generate IDRs](01_generate_idrs.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/01_generate_idrs.ipynb) |
| [02 · Predict IDRs and generate replacements](02_predict_idrs.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/02_predict_idrs.ipynb) |
| [03 · Extract embeddings](03_extract_embeddings.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/03_extract_embeddings.ipynb) |
| [04 · Interpret SAE features](04_interpret_sae_features.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/04_interpret_sae_features.ipynb) |
| [05 · Discover a feature signature](05_discover_feature_signature.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/05_discover_feature_signature.ipynb) |
| [06 · Fine-tune and generate](06_finetune_and_generate.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/06_finetune_and_generate.ipynb) |
| [07 · Design with custom rewards](07_design_with_custom_rewards.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/07_design_with_custom_rewards.ipynb) |
| [08 · Design with SAE rewards](08_design_with_rl_sae.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rotskoff-group/idiom/blob/v1/cookbook/notebooks/08_design_with_rl_sae.ipynb) |

Open a notebook with its Colab button and run the cells in order. Each notebook
installs its dependencies and downloads its own examples; no repository clone is needed.
Parameter comments explain the defaults and input formats.

Notebooks 01–03 default to CPU. For 04–08, a GPU is recommended; 08 also loads a
frozen SAE host model and may exceed the memory available in some runtimes.
GPU availability and runtime limits vary; see the [Colab FAQ](https://research.google.com/colaboratory/faq.html#resource-limits).

Each run uses a timestamped output folder. Saved previews contain only a few rows,
values, or useful plots; full tables, arrays, and models are written to that folder.
Download files from Colab's Files pane before the runtime ends, or copy them to
mounted Drive. Saved notebook outputs do not include these files.

Default sequence inputs come from [`cookbook/example_data`](../example_data/):

| Notebook | Example input |
|---|---|
| 01 · Prompted generation | DisProt proteins with annotated IDRs (`disprot/disprot_len1020_idrs.fasta`) |
| 02 · Prediction and prompted generation | HP1α protein (`prompted_grpo/P45973.fasta`); existing IDR annotations are ignored for prediction |
| 03 · Embeddings, 04 · SAE interpretation, 06 · Fine-tuning | ProtGPS nucleolus IDRs (`protgps/nucleolus.fasta`) |
| 05 · Signature discovery | Activation and repression domains (`effector/ad.fasta`, `effector/rd.fasta`) |

Keep the FASTA setting as `None` to download the example, or supply your own path.
Unprompted generation and reward-design notebooks generate their own sequences.
