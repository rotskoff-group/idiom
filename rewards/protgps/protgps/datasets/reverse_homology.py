# dataset utils
import warnings
from typing import Literal, List
from protgps.datasets.abstract import AbstractDataset
from protgps.utils.registry import register_object, get_object
from tqdm import tqdm
import argparse
import torch
import os
import glob
import re
import numpy as np
import copy


@register_object("reverse_homology", "dataset")
class ReverseHomology(AbstractDataset):
    """A pytorch Dataset for the classifying proteins into compartment."""

    def load_homology_dataset(self, args: argparse.ArgumentParser) -> None:
        """Loads fasta files from dataset folder
        Args:
            args (argparse.ArgumentParser)
        Raises:
            Exception: Unable to load
        """
        data_folders = args.homology_dataset_folder.split(",")
        fasta_paths = []
        for folder_path in data_folders:
            fasta_paths.extend(glob.glob(os.path.join(folder_path, "*.fasta")))
        print("Loading fasta files...")
        for fasta in tqdm(fasta_paths):
            idrs = []
            f = open(fasta, "r")
            lines = f.readlines()
            for line in lines:
                outh = re.search(">", line)
                if outh:
                    pass
                else:
                    s = line.replace("-", "").strip()
                    if len(s) <= self.args.max_idr_len:  # skip long sequences
                        idrs.append(s)
            if len(idrs) >= self.args.pos_samples + 1:
                self.homology_sets.append(np.array(idrs))

    def init_class(self, args: argparse.ArgumentParser, split_group: str) -> None:
        """Perform Class-Specific init methods
           Default is to load JSON dataset

        Args:
            args (argparse.ArgumentParser)
            split_group (str)
        """
        self.homology_sets = []
        if self.args.homology_dataset_folder and self.args.use_homology_dataset:
            self.load_homology_dataset(args)
        if self.args.compartment_dataset_file and self.args.use_compartment_dataset:
            self.load_compartment_dataset(copy.deepcopy(args))

    def load_compartment_dataset(self, args: argparse.ArgumentParser) -> None:
        """Loads dataset from json file
        Args:
            args (argparse.ArgumentParser)
        """
        if self.args.compartment_dataset_name:
            args.dataset_file_path = self.args.compartment_dataset_file
            args.drop_multilabel = False
            args.max_prot_len = np.inf
            dataset = get_object(self.args.compartment_dataset_name, "dataset")(
                args, "train"
            )
            comp_dict = {}
            for sample_dict in tqdm(dataset.metadata_json):
                idrs = "".join(sample_dict["idrs"])
                if len(idrs) <= self.args.max_idr_len:
                    label = dataset.get_label(sample_dict)
                    for l in torch.argwhere(label == 1).T[0]:
                        l = l.item()
                        if l in comp_dict:
                            comp_dict[l].append(idrs)
                        else:
                            comp_dict[l] = [idrs]

            for label in comp_dict:
                if len(comp_dict[label]) >= self.args.pos_samples + 1:
                    self.homology_sets.append(np.array(comp_dict[label]))
        else:
            raise Exception("No compartment dataset name provided")

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        dataset = []
        print(f"Creating '{split_group}' dataset...")
        if split_group == "train":
            hom_mult = self.args.homology_multiple * self.args.split_probs[0]
            rng = np.random.default_rng(self.args.dataset_seed)
        elif split_group == "dev":
            hom_mult = self.args.homology_multiple * self.args.split_probs[1]
            rng = np.random.default_rng(self.args.dataset_seed + 1)
        elif split_group == "test":
            hom_mult = self.args.homology_multiple * self.args.split_probs[2]
            rng = np.random.default_rng(self.args.dataset_seed + 2)

        for _ in tqdm(range(int(hom_mult * len(self.homology_sets)))):
            sample, rng = self.generate_sample(rng)
            dataset.append(sample)
        return dataset

    def generate_sample(self, rng) -> dict:
        """Generates sample for contrastive learning of homology sets
        Args:
            rng: numpy random generator
        Returns:
            list: list of strings
        """
        if len(self.homology_sets) < self.args.neg_samples + 1:
            self.args.neg_samples = len(self.homology_sets) - 1
        neg_idx = rng.choice(
            len(self.homology_sets), size=self.args.neg_samples + 1, replace=False
        )
        pos_idx, neg_idx = neg_idx[0], neg_idx[1:]
        pos_samples = rng.choice(
            self.homology_sets[pos_idx], size=self.args.pos_samples + 1, replace=False
        )
        anchor, pos_samples = pos_samples[0], pos_samples[1:]
        neg_samples = np.array(
            [
                rng.choice(self.homology_sets[i], size=self.args.neg_multiple)
                for i in neg_idx
            ]
        ).flatten()
        return {"x": [anchor, *pos_samples, *neg_samples]}, rng

    def __getitem__(self, index):
        # rng = np.random.default_rng(self.args.dataset_seed)
        try:
            return self.dataset[index]
        except Exception:
            warnings.warn("Could not load sample")

    @property
    def SUMMARY_STATEMENT(self) -> None:
        """
        Prints summary statement with dataset stats
        """
        try:
            return (
                f"Reverse Homology Dataset with {len(self.dataset)} samples\n"
                + f"Using Homology sets: {len(self.homology_sets)}\n"
                + f"Using {self.args.pos_samples} positive samples and {self.args.neg_samples * self.args.neg_multiple} negative samples\n"
            )
        except:
            return "Could not produce summary statement"

    @staticmethod
    def set_args(args) -> None:
        args.num_classes = 2

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(ReverseHomology, ReverseHomology).add_args(parser)
        parser.add_argument(
            "--homology_dataset_folder",
            type=str,
            help="folders containing fasta files seperated by comma",
        )
        parser.add_argument(
            "--dataset_seed",
            type=int,
            help="seed for dataset generation",
        )
        parser.add_argument(
            "--homology_multiple",
            type=float,
            default=1,
            help="the expected number of times to use each homology set as a positive example",
        )
        parser.add_argument(
            "--pos_samples",
            type=int,
            help="number of positive samples to use from the anchor homology set",
        )
        parser.add_argument(
            "--neg_samples",
            type=int,
            help="number of homology sets to draw negative samples from",
        )
        parser.add_argument(
            "--max_idr_len",
            type=int,
            help="max total length of idrs in a protein",
        )
        parser.add_argument(
            "--compartment_dataset_file",
            type=str,
            help="json file containing compartment dataset",
        )
        parser.add_argument(
            "--compartment_dataset_name",
            type=str,
            help="protgps name of compartment dataset object",
        )
        parser.add_argument(
            "--use_compartment_dataset",
            action="store_true",
            default=False,
            help="use compartment dataset to generate homology sets",
        )
        parser.add_argument(
            "--use_homology_dataset",
            action="store_true",
            default=False,
            help="use homology dataset to generate homology sets",
        )
        parser.add_argument(
            "--neg_multiple",
            type=int,
            default=1,
            help="number of negative samples to draw from each negative homology set",
        )
