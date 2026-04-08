# dataset utils
import warnings
from typing import Literal, List
from protgps.datasets.abstract import AbstractDataset
from protgps.utils.registry import register_object
from tqdm import tqdm
import argparse
import torch


@register_object("protein_compartment", "dataset")
class Protein_Compartments(AbstractDataset):
    """A pytorch Dataset for the classifying proteins into compartment."""

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

    @property
    def COMPARTMENTS(self):
        return ["cytosol", "nucleoli", "nucleoplasm", "ER", "mitochondria"]

    def skip_sample(self, sample, split_group) -> bool:
        """
        Return True if sample should be skipped and not included in data
        """
        if sample["split"] != split_group:
            return True
        if "Sequence" in sample and len(sample["Sequence"]) < 10:
            return True
        if "sequence" in sample and len(sample["sequence"]) < 10:
            return True

        if "Sequence" in sample and len(sample["Sequence"]) > self.args.max_prot_len:
            return True
        if "sequence" in sample and len(sample["sequence"]) > self.args.max_prot_len:
            return True

        return False

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        dataset = []
        for sample_dict in tqdm(self.metadata_json):
            if self.skip_sample(sample_dict, split_group):
                continue

            item = {
                "x": sample_dict["Sequence"],
                "y": self.get_label(sample_dict),
                "sample_id": sample_dict["Entry"],
            }
            dataset.append(item)
        return dataset

    def get_label(self, sample):
        """
        Get task specific label for a given sample
        """
        try:
            return torch.tensor([sample[c] for c in self.COMPARTMENTS])
        except:
            return None

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
        try:
            compartment_counts = (
                torch.stack([d["y"] for d in self.dataset]).sum(0).tolist()
            )
            compartment_str = ""
            for i, (c, count) in enumerate(zip(self.COMPARTMENTS, compartment_counts)):
                compartment_str += f"{count} {c.upper()}"
                if i < len(self.COMPARTMENTS) - 1:
                    compartment_str += " -- "
            return f"* {len(self.dataset)} Proteins.\n* {compartment_str}"
        except:
            return "Could not produce summary statement"

    @staticmethod
    def set_args(args) -> None:
        args.num_classes = 5

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        AbstractDataset.add_args(parser)
        parser.add_argument(
            "--max_prot_len",
            type=int,
            default=2000,
            help="len above which to skip prots",
        )


@register_object("protein_compartment_guy", "dataset")
class ProteinCompartmentsGuy(AbstractDataset):
    """A pytorch Dataset for the classifying proteins into compartment."""

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

    @property
    def COMPARTMENTS(self):
        return [
            "Nucleus",
            "Cytoplasm",
            "Secreted",
            "Mitochondrion",
            "Membrane",
            "Endoplasmic",
            "Plastid",
            "Golgi_apparatus",
            "Lysosome",
            "Peroxisome",
        ]

    @property
    def esm_tokens(self):
        return [
            "L",
            "A",
            "G",
            "V",
            "S",
            "E",
            "R",
            "T",
            "I",
            "D",
            "P",
            "K",
            "Q",
            "N",
            "F",
            "Y",
            "M",
            "H",
            "W",
            "C",
            "X",
            "B",
            "U",
            "Z",
            "O",
            ".",
            "-",
        ]

    def target_index(self, target):
        return self.COMPARTMENTS.index(target)

    def skip_sample(self, sample, split_group) -> bool:
        """
        Return True if sample should be skipped and not included in data
        """
        if sample is None:
            return True
        if self.get_label(sample) is None:
            # print("Skipped because no label")
            return True
        if self.args.drop_multilabel:
            if self.get_label(sample).sum() > 1:  # skip multi-compartment samples
                print("Skipped because multi label")
                return True
        if split_group in ["train", "dev", "test"]:
            if sample["split"] != split_group:
                return True
        if "sequence" in sample and len(sample["sequence"]) < 10:
            return True
        if "sequence" in sample and len(sample["sequence"]) > self.args.max_prot_len:
            return True
        if "Sequence" in sample and len(sample["Sequence"]) < 10:
            return True
        if "Sequence" in sample and len(sample["Sequence"]) > self.args.max_prot_len:
            return True
        if "sequence" in sample and not set(sample["sequence"]).issubset(
            self.esm_tokens
        ):
            return True
        if "Sequence" in sample and not set(sample["Sequence"]).issubset(
            self.esm_tokens
        ):
            return True
        return False

    def skip_idr_sample(self, sample, split_group) -> bool:
        if self.skip_sample(sample, split_group):
            return True

        if all([len(s) < 10 for s in sample["idrs"]]):  # if all IDRs are small
            print("Skipped because all IDRs are len 10 or less")
            return True

        if len(sample["idrs"]) == 0:  # if there are no idrs
            print("Skipped because no IDRs")
            return True

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        dataset = []
        for sample_dict in tqdm(self.metadata_json):
            if self.skip_sample(sample_dict, split_group):
                continue
            sss = "sequence" if "sequence" in sample_dict else "Sequence"
            eid = "entry" if "entry" in sample_dict else "Entry"
            item = {
                "x": sample_dict[sss],
                "y": self.get_label(sample_dict),
                "entry_id": sample_dict[eid],
                # "sample_id": sample_dict["Entry"],
            }
            dataset.append(item)
        return dataset

    def get_label(self, sample):
        """
        Get task specific label for a given sample
        """
        try:
            return torch.tensor([sample["labels"][c] for c in self.COMPARTMENTS])
        except:
            return None

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
        try:
            compartment_counts = (
                torch.stack([d["y"] for d in self.dataset]).sum(0).tolist()
            )
            compartment_str = ""
            for i, (c, count) in enumerate(zip(self.COMPARTMENTS, compartment_counts)):
                compartment_str += f"{count} {c.upper()}"
                if i < len(self.COMPARTMENTS) - 1:
                    compartment_str += " -- "
            return f"* {len(self.dataset)} Proteins.\n* {compartment_str}"
        except:
            return "Could not produce summary statement"

    @staticmethod
    def set_args(args) -> None:
        args.num_classes = 10

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        AbstractDataset.add_args(parser)
        parser.add_argument(
            "--max_prot_len",
            type=int,
            default=2000,
            help="len above which to skip prots",
        )

        parser.add_argument(
            "--drop_multilabel",
            type=bool,
            default=False,
            help="whether to drop multilabel samples",
        )


@register_object("protein_compartment_uniprot_combined", "dataset")
class ProteinCompartmentsUniprotCombined(ProteinCompartmentsGuy):
    def get_label(self, sample):
        """
        Get task specific label for a given sample
        """
        try:
            label = []
            for c in self.COMPARTMENTS:
                if isinstance(c, str):
                    if c in sample["labels"]:
                        label.append(sample["labels"][c])
                    else:
                        label.append(0)
                else:
                    l = 0
                    for c_ in c:
                        if c_ in sample["labels"]:
                            if sample["labels"][c_] == 1:
                                l = 1
                                break
                        else:
                            continue
                    label.append(l)
            if sum(label) > 0:
                return torch.tensor(label)
            else:
                return None
        except:
            return None

    def target_index(self, target):
        for i, c in enumerate(self.COMPARTMENTS):
            if isinstance(c, str):
                if isinstance(target, str):
                    if c == target:
                        return i
            else:
                if target in c:
                    return i
                elif next(iter(target)) in c:
                    return i
        return None

    @property
    def COMPARTMENTS(self):
        return [
            "nuclear_membrane",
            "rough_endoplasmic_reticulum",
            "vacuole",
            "nucleus",
            "inflammasome",
            {"endplasmic_reticulum", "endoplasmic_reticulum"},
            "cytoplasm",
            "nuclear_gem",
            {"membrane", "cell_membrane"},
            "mitochondrion",
            {"vesicle", "vesicles"},
            "cell_projection",
            "lipid_droplet",
            "sarcoplasmic_reticulum",
            "endosome",
            "centromere",
            "nuclear_body",
            "nucleoplasm",
            "golgi_apparatus",
            {"excretion_vesicles", "excretion_vesicle"},
            "peroxisome",
            "lysosome",
        ]

    @staticmethod
    def set_args(args) -> None:
        args.num_classes = 22


# USE THIS
@register_object("protein_condensates_combined", "dataset")
class ProteinCondensatesCombined(ProteinCompartmentsUniprotCombined):
    @property
    def COMPARTMENTS(self):
        return [
            {"nuclear_speckles", "nuclear_speckle"},
            {"pbody", "p-body"},
            {"pml_body", "pml-bdoy"},
            "post_synaptic_density",
            "stress_granule",
            {"chromosomes", "chromosome"},
            "nucleolus",
            "nuclear_pore_complex",
            "cajal_body",
            "rna_granule",
            "cell_junction",
            "transcriptional",
        ]

    @staticmethod
    def set_args(args) -> None:
        args.num_classes = 12
