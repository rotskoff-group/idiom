# Usage reference

See the [notebooks](notebooks/) for complete walkthroughs and
[sequence conventions](../README.md#sequence-conventions) for FASTA headers and IDR coordinates.
Examples below assume the indicated input files exist.

## Loading, generation, and embeddings

`IDiom.from_pretrained` accepts a Hub model ID or a released directory.
Use `IDiom.load` to also accept a Lightning `.ckpt` file.

```python
from idiom import IDiom

model = IDiom.from_pretrained("jxliu2/idiom-300M")
sequences = model.generate_unprompted(n=10, length_range=(80, 120))
embeddings = model.embed(sequences, layers=[18], pool="mean")[18]
```

Use `pool="none"` for per-residue embeddings. Length-filtered generation may return fewer sequences
if it reaches its sampling limit.

Given a protein sequence and its IDR coordinates, `generate_prompted(seq, idr_start, idr_end, n=10)`
generates replacement IDRs conditioned on the flanks. To write redesigned proteins from a record FASTA:

```python
model.generate_prompted_fasta(
    "proteins.fasta", "redesigned.fasta", n=10, return_full=True
)
```

`return_full=True` inserts each generated IDR between its flanks and updates the FASTA span.
For de novo output, use `generate_unprompted_fasta` or:

```bash
idiom_generate unprompted --model jxliu2/idiom-300M --n 100 --out idrs.fasta
```

## Perplexity

Score a record FASTA under the model's fill-in-the-middle objective:

```python
from idiom.utils.perplexity import perplexity

scores = perplexity(model.model, "heldout.fasta", device=model.device)
# scores contains nll, perplexity, and n_tokens
```

## SAE features and steering

`IDiomSAE` loads the SAE with its recorded host model, layer, and prompt format.

```python
from idiom import IDiomSAE

sae = IDiomSAE.from_pretrained("jxliu2/idiomsae-300M-L18-k32")
features, accessions = sae.encode("proteins.fasta", pool="mean")
sequences = sae.steer_generate(feature=1234, strength=0.5, n=10)
```

Use `pool="none"` for per-residue features. The released SAE was trained on unprompted IDRs;
its only valid `region` is `"idr"` (the default). SAEs trained with flanking context can also
select `"all"` or `"non_idr"`.

Steering supports `add_direction`, `clamp`, and `ablate`. Options `normalize`, `relative`,
and `preserve_norm` control how strength is applied; see [sae_features.ipynb](notebooks/sae_features.ipynb).

Build and inspect a feature dataset:

```bash
idiom_feature_dataset --sae jxliu2/idiomsae-300M-L18-k32 --fasta proteins.fasta --out features/
# Run from the repository clone.
streamlit run src/idiom/sae/features/feature_viewer.py -- --features features/
```

```python
from idiom.sae.features import FeatureDataset

dataset = FeatureDataset("features/")
ids, frequency, mean_activation = dataset.feature_ranking()
sequence_ids, scores = dataset.top_sequences(ids[0], n=20)
```

## Exporting and publishing

Export a training checkpoint to a directory containing `config.json` and `model.safetensors`:

```python
from idiom import IDiom

model = IDiom.load("/path/to/model.ckpt")
model.save_pretrained("my-idiom")
```

To upload the model, authenticate with Hugging Face and call:

```python
model.push_to_hub("your-account/my-idiom", private=True, model_card="# My IDiom model")
```

`IDiomSAE` also provides `save_pretrained` and `push_to_hub`, recording its host model so the
pair can be reloaded. Its exported filenames are `sae_config.json` and `sae.safetensors`.
