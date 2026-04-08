import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
from tqdm import tqdm
import pandas as pd
import torch
from pytorch_lightning.utilities.cloud_io import load as pl_load
from typing import List
from protgps.utils.registry import get_object

COMPARTMENTS = [
    "nuclear_speckle",
    "p-body",
    "pml-bdoy",
    "post_synaptic_density",
    "stress_granule",
    "chromosome",
    "nucleolus",
    "nuclear_pore_complex",
    "cajal_body",
    "rna_granule",
    "cell_junction",
    "transcriptional",
]


def load_model(model_path: str, esm_dir: str) -> torch.nn.Module:
    """
    Load model from checkpoint

    Parameters
    ----------
    model_path : str
        Path to the model checkpoint
    esm_dir : str
        Path to ESM-2 directory

    Returns
    -------
    torch.nn.Module
        Model instance loaded from the checkpoint
    """
    checkpoint = pl_load(model_path, map_location=lambda storage, loc: storage)
    args = checkpoint["hyper_parameters"]["args"]
    args.pretrained_hub_dir = esm_dir
    model = get_object(args.lightning_name, "lightning")(args)
    model = model.load_from_checkpoint(
        checkpoint_path=model_path,
        **{"args": args},
    )
    return model


def predict_condensates(
    model: torch.nn.Module, sequences: List[str], batch_size: int, round: bool = True
) -> torch.Tensor:
    """
    Predict condensate ID for the given sequences

    Parameters
    ----------
    model : torch.nn.Module
        protGPS
    sequences : list
        List of sequences
    batch_size : int
        Batch size for inference
    round : bool, optional
        whether to round scores, by default True

    Returns
    -------
    torch.Tensor
        Predicted scores for each condensate
    """
    scores = []
    for i in tqdm(range(0, len(sequences), batch_size), ncols=100):
        batch = sequences[i : (i + batch_size)]
        with torch.no_grad():
            out = model.model({"x": batch})
        s = torch.sigmoid(out["logit"]).to("cpu")
        scores.append(s)
    scores = torch.vstack(scores)
    if round:
        scores = torch.round(scores, decimals=3)

    scores = scores.cpu()  # move to cpu
    return scores


def get_valid_rows(df: pd.DataFrame, cols: list) -> list:
    """
    Get rows with valid sequence length

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    cols : list
        Column name of the sequences

    Returns
    -------
    list
        List of row indices with valid sequence length
    """
    rows_with_valid_seq_len = []
    for i in range(len(df)):
        if all([len(df.iloc[i][c]) < 1800 for c in cols]):
            rows_with_valid_seq_len.append(i)
    return rows_with_valid_seq_len


parser = argparse.ArgumentParser(description="Inference script")
parser.add_argument("--model_path", "-m", type=str, help="Input file path")
parser.add_argument(
    "--esm_dir", type=str, help="Path to ESM2 directory", default="esm_checkpoint"
)
parser.add_argument(
    "--device", type=str, help="Device to run inference on", default="cpu"
)
parser.add_argument("--input", "-i", type=str, help="Input file path")
parser.add_argument(
    "--colname", type=str, help="Column name of the sequences", default="Sequence"
)
parser.add_argument("--output", "-o", type=str, help="Output file path")

if __name__ == "__main__":
    args = parser.parse_args()
    # load model
    model = load_model(args.model_path, args.esm_dir)
    model.eval()
    print()

    # move model to device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # read input data
    data = pd.read_excel(args.input)
    assert args.colname in data.columns, (
        f"Column name {args.colname} not found in the input file"
    )

    # get valid rows (sequences with length < 1800)
    rows_with_valid_seq_len = get_valid_rows(data, [args.colname])
    data = data.loc[rows_with_valid_seq_len]

    sequences = [s.upper() for s in list(data[args.colname])]

    # predict condensates
    scores = predict_condensates(model, sequences, batch_size=1, round=False)
    for j, condensate in enumerate(COMPARTMENTS):
        data[f"{condensate.upper()}_Score"] = scores[:, j].tolist()

    # save output
    data.to_csv(args.output, index=False)
