# dataset utils
import warnings
from typing import Literal, List
from protgps.datasets.abstract import AbstractDataset
from protgps.utils.registry import register_object, get_object
from tqdm import tqdm
import argparse
import torch


@register_object("disprot", "dataset")
class Disprot(AbstractDataset):
    """A pytorch Dataset for the classifying protein intrinsically disordered regions from Disprot DB."""

    def init_class(self, args: argparse.ArgumentParser, split_group: str) -> None:
        """Perform Class-Specific init methods
           Default is to load JSON dataset

        Args:
            args (argparse.ArgumentParser)
            split_group (str)
        """
        self.load_dataset(args)
        if args.assign_splits:
            self.assign_splits(
                self.metadata_json, split_probs=args.split_probs, seed=args.split_seed
            )

        if args.precomputed_protein_embeddings:
            self.protein_encoder = get_object(self.args.protein_encoder_name, "model")(
                args
            ).to("cuda")
            self.protein_encoder.eval()

    def skip_sample(self, sample, split_group) -> bool:
        """
        Return True if sample should be skipped and not included in data
        """
        if sample["split"] != split_group:
            return True
        return False

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        sequences = []
        dataset = []
        for protein_dict in tqdm(self.metadata_json["data"]):
            if self.skip_sample(protein_dict, split_group):
                continue

            item = {
                "x": protein_dict["sequence"],
                "y": self.get_label(sample_dict),
                "sample_id": protein_dict["disprot_id"],
            }
            sequences.append(protein_dict["sequence"])
            dataset.append(item)

        if args.precomputed_protein_embeddings:
            # this batches protein sequences and then converts to features
            batch_size = 10
            hiddens = []
            for i in tqdm(range(0, len(ids), batch_size)):
                preds = self.protein_encoder(sequences[i : i + batch_size])
                hiddens.append(preds["hidden"].cpu())
            hiddens = torch.stack(hiddens)

            for i, h in enumerate(hiddens):
                dataset[i]["sequence"] = dataset[i]["x"]
                dataset[i]["x"] = h

        return dataset

    def get_label(self, protein_dict):
        """
        Get task specific label for a given sample
        """
        y = torch.zeros(len(protein_dict["sequence"]))
        for disordered_region in protein_dict["regions"]:
            start = disordered_region["start"] - 1
            end = disordered_region["end"]
            y[start:end] = 1
        return y

    def __getitem__(self, index):
        try:
            return self.dataset[index]

        except Exception:
            warnings.warn("Could not load sample")

    @property
    def SUMMARY_STATEMENT(self) -> None:
        """
        Prints summary statement with dataset stats
        """
        return f"{len(self.dataset)} Proteins."

    @staticmethod
    def set_args(args) -> None:
        args.num_classes = 1

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(Disprot, Disprot).add_args(parser)
        parser.add_argument(
            "--precomputed_protein_embeddings",
            default=False,
            action="store_true",
            help="whether to use precomputed embeddings",
        )


@register_object("protein_compartment_precomputed", "dataset")
class Protein_Compartments_Precomputed(Protein_Compartments):
    """A pytorch Dataset for the classifying proteins into compartment."""

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        dataset = []
        for sample_dict in tqdm(self.metadata_json):
            if self.skip_sample(sample_dict, split_group):
                continue

            item = {
                "sequence": sample_dict["Sequence"],
                "x": torch.tensor(sample_dict["esm2_embedding"]),
                "y": self.get_label(sample_dict),
                "sample_id": sample_dict["Entry"],
            }
            dataset.append(item)

        return dataset

    @staticmethod
    def set_args(args) -> None:
        args.num_classes = 5
        args.mlp_input_dim = args.protein_hidden_dim

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(
            Protein_Compartments_Precomputed, Protein_Compartments_Precomputed
        ).add_args(parser)
        parser.add_argument(
            "--protein_hidden_dim",
            type=int,
            default=1280,
            help="hidden dimension of the protein",
        )
