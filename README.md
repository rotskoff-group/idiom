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

## Entrypoints/commands

Entrypoints in `entrypoints`

Commands are under:
```
entrypoints/infer
entrypoints/train/post-train

entrypoints/train/pre-train
entrypoints/precompute
```

For typical usage, `entrypoints/infer` and `entrypoints/train/post-train` for generating sequences and post-training the model with an external reward model or oracle. 

## Generating unprompted IDRs 

To generate unprompted IDRs

## Generating prompted IDRs

To generate prompted IDRs
