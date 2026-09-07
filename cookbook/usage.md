# Usage reference

See [core usage](../README.md#quickstart) for generation, embeddings, SAE steering, and model export,
and the [notebooks](notebooks/) for complete walkthroughs. Examples below assume the indicated
input files exist and follow the [sequence conventions](../README.md#sequence-conventions).

## Perplexity

Score a record FASTA under the model's fill-in-the-middle objective:

```python
from idiom import IDiom
from idiom.utils.perplexity import perplexity

model = IDiom.from_pretrained("jxliu2/idiom-300M")
scores = perplexity(model.model, "heldout.fasta", device=model.device)
# scores contains nll, perplexity, and n_tokens
```

## Feature datasets

Build and inspect a feature dataset:

```bash
idiom_feature_dataset --sae jxliu2/idiomsae-300M-L18-k32 --fasta proteins.fasta --out features/
# Run from the repository clone.
streamlit run src/idiom/sae/features/feature_viewer.py -- --features features/
```

```python
from idiom.sae.features import FeatureDataset

dataset = FeatureDataset("features/")
max_activation, total_activation, firing_count = dataset.feature_ranking()
feature_id = int(max_activation.argmax())
sequence_ids, scores = dataset.top_sequences(feature_id, n=20)
```
