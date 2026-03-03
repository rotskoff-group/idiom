# IDR-PLM
Official repository for IDR-PLM. 

All necessary data and model checkpoints can be found on our HuggingFace Model and Dataset repositories here: 

Model checkpoints: https://huggingface.co/jxliu2/idr-plm

Dataset: https://huggingface.co/datasets/jxliu2/idr-plm-dataset

## Environment setup (uv) 

```bash
git clone https://github.com/rotskoff-group/idr-plm.git
cd idr-plm 
uv sync
uv pip install -e .
```

## Data and checkpoints 
After downloading, place dataset and checkpoint files under the repository `data/` subdirectory. The configs read from:
```
data/<dataset>/scale2max/<dataset>.csv
data/<dataset>/<dataset>.fasta
```

## Entrypoints/commands

Entrypoints in `entrypoints/`

Commands are under:
```
entrypoints/infer
entrypoints/train/post-train

entrypoints/train/pre-train
entrypoints/precompute
```

For typical usage, `entrypoints/infer` and `entrypoints/train/post-train` for generating sequences and post-training the model with an external reward model or oracle. 

### Generating unprompted IDPs

To generate unprompted IDRs, 

### Generating prompted IDRs

To generate prompted IDRs, 

### Post-training the base model with your own reward function

To post-train, 

After that, generating unprompted IDPs or prompted IDRs can be performed following the above sections. 
