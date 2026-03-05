import torch
import torch.nn as nn
import copy
from protgps.models.abstract import AbstractModel
from protgps.utils.classes import set_protgps_type
from protgps.utils.registry import register_object, get_object
import functools


@register_object("fair_esm", "model")
class FairEsm(AbstractModel):
    """
    Refer to https://github.com/facebookresearch/esm#available-models
    """

    def __init__(self, args):
        super(FairEsm, self).__init__()
        self.args = args
        torch.hub.set_dir(args.pretrained_hub_dir)
        self.model, self.alphabet = torch.hub.load(
            "facebookresearch/esm:main", args.esm_name
        )
        self.batch_converter = (
            self.alphabet.get_batch_converter()
        )  # TODO: Move to dataloader, so that we can batch in parallel
        self.register_buffer("devicevar", torch.zeros(1, dtype=torch.int8))
        if args.freeze_esm:
            self.model.eval()

        self.repr_layer = args.esm_hidden_layer
        print("Using ESM hidden layers", self.repr_layer)

    def forward(self, x, tokens=False, soft=False):
        """
        x: list of str (protein sequences)
        tokens: tokenized or tensorized input
        soft: embeddings precomputed
        """
        output = {}
        if tokens:
            batch_tokens = x.unsqueeze(0)
        else:
            fair_x = self.truncate_protein(x, self.args.max_prot_len)
            batch_labels, batch_strs, batch_tokens = self.batch_converter(fair_x)

        batch_tokens = batch_tokens.to(self.devicevar.device)

        # use partial for cleanness
        model_func = functools.partial(
            self.model,
            repr_layers=[self.repr_layer],
            return_contacts=False,
        )
        if soft:
            model_func = functools.partial(model_func, soft=soft)

        if self.args.freeze_esm:
            with torch.no_grad():
                result = model_func(batch_tokens)
        else:
            result = model_func(batch_tokens)

        # Generate per-sequence representations via averaging
        hiddens = []
        for sample_num, sample in enumerate(x):
            # breakpoint()
            hiddens.append(
                result["representations"][self.repr_layer][
                    sample_num, 1 : len(sample) + 1
                ].mean(0)
            )
        if self.args.output_residue_hiddens:
            output["residues"] = result["representations"][self.repr_layer]

        output["hidden"] = torch.stack(hiddens)

        return output

    def truncate_protein(self, x, max_length=None):
        # max length allowed is 1024
        return [
            (i, s[: max_length - 2])
            if not isinstance(x[0], list)
            else (i, s[0][: max_length - 2])
            for i, s in enumerate(x)
        ]

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--pretrained_hub_dir",
            type=str,
            default="/home/protgps/esm_models",
            help="directory to torch hub where pretrained models are saved",
        )
        parser.add_argument(
            "--esm_name",
            type=str,
            default="esm2_t12_35M_UR50D",
            help="directory to torch hub where pretrained models are saved",
        )
        parser.add_argument(
            "--freeze_esm",
            action="store_true",
            default=False,
            help="do not update encoder weights",
        )
        parser.add_argument(
            "--esm_hidden_layer",
            type=int,
            default=12,
            help="do not update encoder weights",
        )
        parser.add_argument(
            "--output_residue_hiddens",
            action="store_true",
            default=False,
            help="do not return residue-level hiddens, only sequence average",
        )


@register_object("fair_esm2", "model")
class FairEsm2(FairEsm):
    # def forward(self, x):
    #     """
    #     x: list of str (protein sequences)
    #     """
    #     output = {}
    #     fair_x = self.truncate_protein(x)
    #     batch_labels, batch_strs, batch_tokens = self.batch_converter(fair_x)
    #     batch_tokens = batch_tokens.to(self.devicevar.device)

    #     if self.args.freeze_esm:
    #         with torch.no_grad():
    #             result = self.model(
    #                 batch_tokens, repr_layers=[self.repr_layer], return_contacts=False
    #             )
    #     else:
    #         result = self.model(
    #             batch_tokens, repr_layers=[self.repr_layer], return_contacts=False
    #         )

    #     # Generate per-sequence representations via averaging
    #     hiddens = []
    #     for sample_num, sample in enumerate(x):
    #         hiddens.append(
    #             result["representations"][self.repr_layer][
    #                 sample_num, 1 : len(sample) + 1
    #             ]
    #         )
    #     if self.args.output_residue_hiddens:
    #         output["residues"] = result["representations"][self.repr_layer]

    #     output["hidden"] = hiddens
    #     return output

    def truncate_protein(self, x, max_length=torch.inf):
        return [
            (i, s) if not isinstance(x[0], list) else (i, s[0]) for i, s in enumerate(x)
        ]


@register_object("fair_esm_fast", "model")
class FairEsmFast(FairEsm):
    def forward(self, x, tokens=False, soft=False):
        """
        x: list of str (protein sequences)
        """
        output = {}
        if tokens:
            batch_tokens = x.unsqueeze(0)
        else:
            fair_x = [(i, v) for i, v in enumerate(x)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(fair_x)
        batch_tokens = batch_tokens.to(self.devicevar.device)

        # use partial for cleanness
        model_func = functools.partial(
            self.model,
            repr_layers=[self.repr_layer],
            return_contacts=False,
        )
        if soft:
            model_func = functools.partial(model_func, soft=soft)

        if self.args.freeze_esm:
            with torch.no_grad():
                result = model_func(batch_tokens)
        else:
            result = model_func(batch_tokens)

        if self.args.output_residue_hiddens:
            output["residues"] = result["representations"][self.repr_layer]

        output["hidden"] = result["representations"][self.repr_layer].mean(axis=1)
        return output


import numpy as np


@register_object("reverse_hom", "model")
class ReverseHomology(FairEsm):
    def forward(self, batch):
        """
        x: list of str (protein sequences)
        """
        output = {}
        x = np.array(batch["x"]).reshape(-1, order="F")
        fair_x = [(i, v) for i, v in enumerate(x)]
        _, _, batch_tokens = self.batch_converter(fair_x)
        batch_tokens = batch_tokens.to(self.devicevar.device)
        if self.args.freeze_esm:
            with torch.no_grad():
                result = self.model(
                    batch_tokens, repr_layers=[self.repr_layer], return_contacts=False
                )
        else:
            result = self.model(
                batch_tokens, repr_layers=[self.repr_layer], return_contacts=False
            )
        if self.args.output_residue_hiddens:
            output["residues"] = result["representations"][self.repr_layer]

        # NOTE: works for batch size of 1 only (otherwise need to reshape)
        output["hidden"] = result["representations"][self.repr_layer].mean(axis=1)

        return output


@register_object("protein_encoder", "model")
class ProteinEncoder(AbstractModel):
    def __init__(self, args):
        super(ProteinEncoder, self).__init__()
        self.args = args
        self.encoder = get_object(args.protein_encoder_type, "model")(args)
        cargs = copy.deepcopy(args)
        cargs.mlp_input_dim = args.protein_hidden_dim
        args.freeze_esm = args.freeze_encoder
        self.mlp = get_object(args.protein_classifer, "model")(cargs)
        if self.args.freeze_encoder:
            self.encoder.eval()

    def forward(self, batch, tokens=False, soft=False):
        output = {}
        if self.args.freeze_encoder:
            with torch.no_grad():
                output_esm = self.encoder(batch["x"], tokens=tokens, soft=soft)
        else:
            output_esm = self.encoder(batch["x"], tokens=tokens, soft=soft)
        # output["protein_hidden"] = output_esm["hidden"]
        output.update(self.mlp({"x": output_esm["hidden"]}))
        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--protein_encoder_type",
            type=str,
            default="fair_esm2",
            help="name of the protein encoder",
            action=set_protgps_type("model"),
        )
        parser.add_argument(
            "--freeze_encoder",
            action="store_true",
            default=False,
            help="do not update encoder weights",
        )
        parser.add_argument(
            "--protein_hidden_dim",
            type=int,
            default=480,
            help="hidden dimension of the protein",
        )
        parser.add_argument(
            "--protein_classifer",
            type=str,
            default="mlp_classifier",
            help="name of classifier",
            action=set_protgps_type("model"),
        )


@register_object("protein_encoder_attention", "model")
class ProteinEncoderAttention(ProteinEncoder):
    def __init__(self, args):
        super(ProteinEncoder, self).__init__()
        self.args = args
        self.encoder = get_object(args.protein_encoder_type, "model")(args)
        cargs = copy.deepcopy(args)
        cargs.mlp_input_dim = args.protein_hidden_dim
        args.freeze_esm = args.freeze_encoder
        self.mlp = get_object(args.protein_classifer, "model")(cargs)
        if self.args.freeze_encoder:
            self.encoder.eval()

        heads = 8
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.protein_hidden_dim, nhead=heads
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, batch):
        output = {}
        if self.args.freeze_encoder:
            with torch.no_grad():
                output_esm = self.encoder(batch["x"])
        else:
            output_esm = self.encoder(batch["x"])

        v_attention = []
        for v in output_esm["hidden"]:
            v = self.transformer_encoder(v)
            v_attention.append(v.mean(0))

        output.update(self.mlp({"x": torch.stack(v_attention)}))
        return output


@register_object("protein_encoder_esm_embeddings", "model")
class ProteinEncoderESMEmbeddings(ProteinEncoder):
    def forward(self, batch):
        output = {}

        fair_x = self.encoder.truncate_protein(batch["x"])
        _, _, batch_tokens = self.encoder.batch_converter(fair_x)
        batch_tokens = batch_tokens.to(self.encoder.devicevar.device)
        esm_embedded = self.encoder.model.embed_tokens(batch_tokens).mean(1)

        # output["protein_hidden"] = output_esm["hidden"]
        output.update(self.mlp({"x": esm_embedded}))
        return output


@register_object("idr_encoder", "model")
class IDREncoder(ProteinEncoder):
    def forward(self, batch):
        output = {}

        if self.args.freeze_encoder:
            with torch.no_grad():
                idr_embeddings = self._forward_function(batch)
        else:
            idr_embeddings = self._forward_function(batch)

        output.update(self.mlp({"x": torch.stack(idr_embeddings)}))
        return output

    def _forward_function(self, batch) -> list:
        output_esm = self.encoder(batch["x"])
        # mask out non-idr residues and average
        B, N, H = output_esm["residues"].shape
        mask = torch.zeros(B, N)
        for i in range(B):
            mask[i, batch["start_idx"][i] : batch["end_idx"][i]] = 1

        idr_residue_embeddings = output_esm["residues"] * mask.unsqueeze(-1).to(
            output_esm["residues"].device
        )
        idr_embeddings = []
        for idx, sample in enumerate(idr_residue_embeddings):
            avg_sample = sample.sum(0) / mask[idx].sum()
            idr_embeddings.append(avg_sample)

        return idr_embeddings

    @staticmethod
    def set_args(args) -> None:
        args.output_residue_hiddens = True


@register_object("all_idr_encoder", "model")
class AllIDREncoder(ProteinEncoder):
    def forward(self, batch):
        output = {}

        if self.args.freeze_encoder:
            with torch.no_grad():
                idr_embeddings = self._forward_function(batch)
        else:
            idr_embeddings = self._forward_function(batch)

        output.update(self.mlp({"x": torch.stack(idr_embeddings)}))
        return output

    def _forward_function(self, batch) -> list:
        output_esm = self.encoder(batch["x"])

        # mask out non-idr residues and average
        B, N, H = output_esm["residues"].shape
        mask = torch.zeros(B, N)

        for i in range(B):
            start_indices = [int(n) for n in batch["start_indices"][i].split("_")]
            end_indices = [int(n) for n in batch["end_indices"][i].split("_")]
            for idr_idx in range(len(start_indices)):
                mask[i, start_indices[idr_idx] : end_indices[idr_idx]] = 1

        idr_residue_embeddings = output_esm["residues"] * mask.unsqueeze(-1).to(
            output_esm["residues"].device
        )
        idr_embeddings = []
        for idx, sample in enumerate(idr_residue_embeddings):
            avg_sample = sample.sum(0) / mask[idx].sum()
            idr_embeddings.append(avg_sample)

        return idr_embeddings

    @staticmethod
    def set_args(args) -> None:
        args.output_residue_hiddens = True


@register_object("all_idr_esm_embeddings_encoder", "model")
class AllIDRESMEmbeddingsEncoder(ProteinEncoder):
    def forward(self, batch):
        output = {}

        fair_x = self.encoder.truncate_protein(batch["x"])
        _, _, batch_tokens = self.encoder.batch_converter(fair_x)
        batch_tokens = batch_tokens.to(self.encoder.devicevar.device)
        esm_embedded = self.encoder.model.embed_tokens(batch_tokens)

        # mask out non-idr residues and average
        B, N, H = esm_embedded.shape
        mask = torch.zeros(B, N)

        for i in range(B):
            start_indices = [int(n) for n in batch["start_indices"][i].split("_")]
            end_indices = [int(n) for n in batch["end_indices"][i].split("_")]
            for idr_idx in range(len(start_indices)):
                mask[i, start_indices[idr_idx] : end_indices[idr_idx]] = 1

        idr_residue_embeddings = esm_embedded * mask.unsqueeze(-1).to(
            esm_embedded.device
        )
        idr_embeddings = []
        for idx, sample in enumerate(idr_residue_embeddings):
            avg_sample = sample.sum(0) / mask[idx].sum()
            idr_embeddings.append(avg_sample)

        output.update(self.mlp({"x": torch.stack(idr_embeddings)}))
        return output


@register_object("all_not_idr_esm_embeddings_encoder", "model")
class AllNotIDRESMEmbeddingsEncoder(ProteinEncoder):
    def forward(self, batch):
        output = {}

        fair_x = self.encoder.truncate_protein(batch["x"])
        _, _, batch_tokens = self.encoder.batch_converter(fair_x)
        batch_tokens = batch_tokens.to(self.encoder.devicevar.device)
        esm_embedded = self.encoder.model.embed_tokens(batch_tokens)

        # mask out non-idr residues and average
        B, N, H = esm_embedded.shape
        mask = torch.ones(B, N)

        for i in range(B):
            start_indices = [int(n) for n in batch["start_indices"][i].split("_")]
            end_indices = [int(n) for n in batch["end_indices"][i].split("_")]
            for idr_idx in range(len(start_indices)):
                mask[i, start_indices[idr_idx] : end_indices[idr_idx]] = 0

        idr_residue_embeddings = esm_embedded * mask.unsqueeze(-1).to(
            esm_embedded.device
        )
        idr_embeddings = []
        for idx, sample in enumerate(idr_residue_embeddings):
            avg_sample = sample.sum(0) / mask[idx].sum()
            idr_embeddings.append(avg_sample)

        output.update(self.mlp({"x": torch.stack(idr_embeddings)}))
        return output


@register_object("all_not_idr_encoder", "model")
class AllNotIDREncoder(AllIDREncoder):
    def _forward_function(self, batch) -> list:
        output_esm = self.encoder(batch["x"])

        # mask out non-idr residues and average
        B, N, H = output_esm["residues"].shape
        mask = torch.ones(B, N)

        for i in range(B):
            start_indices = [int(n) for n in batch["start_indices"][i].split("_")]
            end_indices = [int(n) for n in batch["end_indices"][i].split("_")]
            for idr_idx in range(len(start_indices)):
                mask[i, start_indices[idr_idx] : end_indices[idr_idx]] = 0

        idr_residue_embeddings = output_esm["residues"] * mask.unsqueeze(-1).to(
            output_esm["residues"].device
        )
        idr_embeddings = []
        for idx, sample in enumerate(idr_residue_embeddings):
            avg_sample = sample.sum(0) / mask[idx].sum()
            idr_embeddings.append(avg_sample)

        return idr_embeddings


@register_object("context_idr_hiddens", "model")
class ContextIDREncoder(ProteinEncoder):
    def forward(self, batch):
        output = {}

        if self.args.freeze_encoder:
            with torch.no_grad():
                idr_embeddings = self._forward_function(batch)
        else:
            idr_embeddings

        output["hidden"] = torch.stack(idr_embeddings)
        return output

    def _forward_function(self, batch) -> list:
        output_esm = self.encoder(batch["x"])
        # mask out non-idr residues and average
        B, N, H = output_esm["residues"].shape
        mask = torch.zeros(B, N)
        for i in range(B):
            mask[i, batch["start_idx"][i] : batch["end_idx"][i]] = 1

        idr_residue_embeddings = output_esm["residues"] * mask.unsqueeze(-1).to(
            output_esm["residues"].device
        )
        idr_embeddings = []
        for idx, sample in enumerate(idr_residue_embeddings):
            avg_sample = sample.sum(0) / mask[idx].sum()
            idr_embeddings.append(avg_sample)

        return idr_embeddings

    @staticmethod
    def set_args(args) -> None:
        args.output_residue_hiddens = True


@register_object("fair_esm_hiddens", "model")
class FairEsmHiddens(AbstractModel):
    def __init__(self, args):
        super(FairEsmHiddens, self).__init__()
        self.args = args
        self.encoder = get_object(args.fair_esm_type, "model")(args)
        if self.args.freeze_esm:
            self.encoder.eval()

    def forward(self, batch):
        output = {}
        if self.args.freeze_esm:
            with torch.no_grad():
                output_esm = self.encoder(batch["x"])
        else:
            output_esm = self.encoder(batch["x"])

        return output_esm

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--fair_esm_type",
            type=str,
            default="fair_esm2",
            help="name of the protein encoder",
            action=set_protgps_type("model"),
        )
