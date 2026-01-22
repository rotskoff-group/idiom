import torch
import torch.nn as nn
from clm.layers import TransformerStack, RegressionHead
from clm.layers import TransfusionEmbedding
from torch_scatter import segment_csr


# FH: This transformer model is essentially only useful for SMILES training and SMILES-based tasks.
#   It is not sufficient for mixed token training
class GeometricMolTransformer(nn.Module):
    def __init__(
        self, dim_model: int, token_info: dict[str, int], unified_transformer_args
    ):
        """Generic geometric molecular transformer

        Args:
            dim_model: int
                The dimension of the model
            num_tokens: Number of tokens in the vocabulary
            dim_model: Dimension of the model
            num_heads: Number of heads in the multi-head attention
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dropout_p: Dropout probability

            For molecules, we have two types of tokens:
                "smiles tokens": Here, considered embedding for all tokens in the SMILES sequence
                "structural tokens": Tokens obtained from the VQVAE
            For now, only using the smi token embedding, not the structural token embedding which
                requires re-training a VQVAE model
        """
        super().__init__()
        # INFO
        # LAYERS
        # At least one of the token embeddings should be present
        # Uses fact that empty dictionary evaluates to a False boolean
        contains_smi = bool(token_info["input"]["TOK"])
        contains_struct = bool(token_info["input"]["STRUCT"])

        assert contains_smi or contains_struct, (
            "At least one of the token embeddings should be present!"
        )

        if contains_smi:
            self.smi_token_embedding = nn.Embedding(
                embedding_dim=dim_model,
                num_embeddings=token_info["TOTAL"],
                padding_idx=token_info["input"]["TOK"]["TOK_PAD"],
            )
        else:
            self.smi_token_embedding = None
        if contains_struct:
            self.structural_token_embedding = nn.Embedding(
                embedding_dim=dim_model,
                num_embeddings=token_info["TOTAL"],
                padding_idx=token_info["input"]["STRUCT"]["STRUCT_PAD"],
            )
        else:
            self.structural_token_embedding = None
        self.transformer = TransformerStack(
            d_model=dim_model, **unified_transformer_args
        )
        total = token_info["TOTAL"]
        self.out = RegressionHead(dim_model, total)

    def forward(
        self,
        smi_tokens,
        structural_tokens,
        sequence_id,
        batch_access_indices=None,
        use_cache_here=False,
        inference_iteration=None,
    ):
        """Args:
        src: Source sequence - (batch_size, sequence length)
        tgt: Target sequence - (batch_size, sequence length)
        batch_access_indices: Optional tensor mapping current batch positions to original cache positions
        use_cache_here: Whether to use KV-caching for this forward pass
        inference_iteration: Current inference iteration for debugging purposes
        """
        # (batch_size, sequence_length, 1)
        if self.smi_token_embedding is not None:
            smi_token_embedding = self.smi_token_embedding(smi_tokens)
        else:
            smi_token_embedding = 0
        if self.structural_token_embedding is not None:
            structural_token_embedding = self.structural_token_embedding(
                structural_tokens
            )
        else:
            structural_token_embedding = 0
        embedding = smi_token_embedding + structural_token_embedding
        # embedding.shape = [B, L, D]
        x, _ = self.transformer(
            embedding,
            sequence_id,
            affine=None,
            affine_mask=None,
            # batch_access_indices=batch_access_indices,
            # use_cache_here=use_cache_here,
            # inference_iteration=inference_iteration,
        )
        # Here, x.shape = [B, L, D]
        x = self.out(
            x
        )  # out() is RegressionHead from embeddings_dim to vocabulary size
        # Here, x.shape = [B, L, E] where E is vocab size
        return x  # These are the logits returned to the self.model() call in module.py


class TransfusionMolTransformer(nn.Module):
    """
    Alternative transformer implementation based on Transfusion process
    """

    def __init__(
        self,
        dim_model,
        token_info,
        unified_transformer_args,
        structure_embedding_args,
        structure_out_dim,
    ):
        """Args:
        num_tokens: Number of tokens in the vocabulary
        dim_model: Dimension of the model
        num_heads: Number of heads in the multi-head attention
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        dropout_p: Dropout probability

        For molecules, we have two types of tokens:
            "smiles tokens": Here, considered embedding for all tokens in the SMILES sequence
            "structural tokens": Tokens obtained from the VQVAE
        For now, only using the smi token embedding, not the structural token embedding which
            requires re-training a VQVAE model
        """
        super().__init__()
        # Uses fact that empty dictionary evaluates to a False boolean
        contains_smi = bool(token_info["input"]["TOK"])
        contains_struct = bool(token_info["input"]["STRUCT"])

        assert contains_smi and contains_struct, (
            "Both the token and structure embedding information should be present!"
        )

        self.embed = TransfusionEmbedding(
            d_model=dim_model,
            token_metadata=token_info,
            struct_embedding_opts=structure_embedding_args,
        )

        self.transformer = TransformerStack(
            d_model=dim_model, **unified_transformer_args
        )
        total = token_info["TOTAL"]
        self.token_out_head = RegressionHead(dim_model, total)
        # The output dimension of the final regression head should be the dimensionality of the structural
        #   information passed into the model, in this case the number of dihedral angles
        self.struct_out_head = RegressionHead(dim_model, structure_out_dim)
        self.token_info = token_info

    def forward(self, token_input, struct_input, struct_mask, sequence_id, ts):
        embedding = self.embed(token_input, struct_input, struct_mask, ts)
        x, _ = self.transformer(
            embedding, sequence_id, affine=None, affine_mask=None
        )  # (B, T, E)

        token_out = self.token_out_head(x)

        if struct_input is not None:
            b, t, e = x.shape
            x_flat = x.reshape(b * t, e)
            token_input_flat = token_input.reshape(b * t)
            batch_vector = (
                torch.arange(b).unsqueeze(1).expand(b, t).reshape(b * t).to(x.device)
            )
            # Only select out the structure elements for averaging over
            structure_elements = (
                token_input_flat == self.token_info["input"]["STRUCT"]["STRUCT"]
            )
            struct_batch_indices = batch_vector[structure_elements]
            # Select out the structural tokens only if structural information was present
            #   in the forward pass
            struct_out = x_flat[structure_elements]  # (B * T, 2)
            indptr = torch.diff(struct_batch_indices, dim=0).nonzero() + 1
            # Prepend 0, append the length of the struct out tensor
            indptr = torch.cat(
                [
                    torch.tensor([0], device=struct_out.device),
                    indptr.squeeze(),
                    torch.tensor([struct_out.shape[0]], device=struct_out.device),
                ]
            )
            struct_out = segment_csr(struct_out, indptr, reduce="mean")
            struct_out = self.struct_out_head(struct_out)
        else:
            struct_out = None
        # Return the token output as is, extract the token values in the loss calculation
        return (token_out, struct_out)


class SequenceConcatenator(nn.Module):
    def __init__(self, token_info: dict):
        super().__init__()
        self.token_info = token_info
        self.struct_pad_token = self.token_info["input"]["STRUCT"]["STRUCT_PAD"]
        self.struct_token = self.token_info["input"]["STRUCT"]["STRUCT"]

    def forward(
        self,
        smiles_tokens: torch.Tensor,
        structure_tokens: torch.Tensor,
        smiles_embedding: torch.Tensor,
        structural_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Combines the embeddings from the smiles and structure tracks together into a concatenated sequence
        Args:
            smiles_tokens: The unembedded SMILES tokens, (N, T)
            structure_tokens: The unembedded structure tokens, (N, S) for regular structure sequence only
                or (N, 4, S) for structure sequence with atom, valency, and hybridization information
            smiles_embedding: The embedded SMILES tokens, (N, T, E)
            structural_embedding: The embedded structure tokens, (N, S, E), S < T
        Returns:
            The combined embeddings, (N, T, E) where the structure token positions in the SMILES embeddings
                have been replaced with the corresponding structure embeddings
        """
        # FH: Sequence composition should alwyas be done based on the structure token sequence in the
        #   case where atom, valency, and hybridization information is present.
        if structure_tokens.ndim == 3:
            structure_tokens = structure_tokens[:, 0, :]
        n_smi, t_smi, e_smi = smiles_embedding.shape
        n_struct, t_struct, e_struct = structural_embedding.shape
        assert smiles_tokens.shape == (n_smi, t_smi)
        smiles_tokens = smiles_tokens.reshape(-1)
        structure_tokens = structure_tokens.reshape(-1)
        smiles_embeddings = smiles_embedding.reshape(n_smi * t_smi, e_smi)
        structural_embeddings = structural_embedding.reshape(
            n_struct * t_struct, e_struct
        )
        non_padding_struct_embed = structural_embeddings[
            structure_tokens != self.struct_pad_token
        ]
        structure_embedding_indices = smiles_tokens == self.struct_token

        smiles_embeddings[structure_embedding_indices] = non_padding_struct_embed
        return smiles_embeddings.reshape(n_smi, t_smi, e_smi)


class SequenceSummator(nn.Module):
    def __init__(self, token_info: dict):
        super().__init__()
        self.token_info = token_info

    def forward(
        self,
        smiles_tokens: torch.Tensor,
        structure_tokens: torch.Tensor,
        smiles_embedding: torch.Tensor,
        structural_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Combines the embeddings from the smiles and structure tracks together into a summed sequence
        Args:
            smiles_tokens: The unembedded SMILES tokens, (N, T)
            smiles_embedding: The embedded SMILES tokens, (N, T, E)
            structural_embedding: The embedded structure tokens, (N, S, E), S < T
        Returns:
            The combined embeddings, (N, T, E) where the structure token positions in the SMILES embeddings
                have been replaced with the corresponding structure embeddings
        """
        return smiles_embedding + structural_embedding


class StructAVH(nn.Module):
    """Structural embedding that combines the atom, valency, and hybridization information"""

    def __init__(self, d_model: int, token_info: dict, structure_embed_args: dict):
        super().__init__()
        self.token_info = token_info
        self.d_model = d_model
        self.structure_embed_args = structure_embed_args
        assert self.structure_embed_args["embedding_form"] in [
            "struct_avh_sum",
            "struct_avh_concat",
        ], "Invalid embedding form specified!"
        assert "avh_spec" in self.structure_embed_args, (
            "Must specify the AVH specification for the structure embedding!"
        )

        embed_form = self.structure_embed_args["embedding_form"]
        if embed_form == "struct_avh_sum":
            embedding_dimension = self.d_model
        elif embed_form == "struct_avh_concat":
            assert self.d_model % 4 == 0, (
                "The model dimension must be divisible by 4 for the concatenation embedding form!"
            )
            embedding_dimension = self.d_model // 4

        n_atom_class = self.structure_embed_args["avh_spec"]["n_atom_class"]
        atom_pad_idx = self.structure_embed_args["avh_spec"]["atom_pad_idx"]
        n_valency_class = self.structure_embed_args["avh_spec"]["n_valency_class"]
        valency_pad_idx = self.structure_embed_args["avh_spec"]["valency_pad_idx"]
        n_hybrid_class = self.structure_embed_args["avh_spec"]["n_hybrid_class"]
        hybrid_pad_idx = self.structure_embed_args["avh_spec"]["hybrid_pad_idx"]

        self.structural_embedding = nn.Embedding(
            embedding_dim=embedding_dimension,
            num_embeddings=self.token_info["input"]["STRUCT"]["STRUCT_MAX_SIZE"],
            padding_idx=self.token_info["input"]["STRUCT"]["STRUCT_PAD"],
        )
        self.atom_embedding = nn.Embedding(
            embedding_dim=embedding_dimension,
            num_embeddings=n_atom_class
            + 1,  # 0 - n_atom_class for actual embeddings, then padding
            padding_idx=atom_pad_idx,
        )
        self.valency_embedding = nn.Embedding(
            embedding_dim=embedding_dimension,
            num_embeddings=n_valency_class
            + 1,  # 0 - n_valency_class for actual embeddings, then padding
            padding_idx=valency_pad_idx,
        )
        self.hybridization_embedding = nn.Embedding(
            embedding_dim=embedding_dimension,
            num_embeddings=n_hybrid_class
            + 1,  # 0 - n_hybrid_class for actual embeddings, then padding
            padding_idx=hybrid_pad_idx,
        )

    def forward(self, x):
        """
        Args:
            x: The input tensor, (N, 4, T). The four parallel tracks are as follows:
                - structural tokens
                - atom tokens
                - valency tokens
                - hybridization tokens
        Returns:
            The structural embedding, (N, T, E)
        """
        assert x.ndim == 3, "The input tensor must be 3D!"
        assert x.shape[1] == 4, "The input tensor must have 4 parallel tracks!"
        struct_embed = self.structural_embedding(x[:, 0, :])
        atom_embed = self.atom_embedding(x[:, 1, :])
        valency_embed = self.valency_embedding(x[:, 2, :])
        hybridization_embed = self.hybridization_embedding(x[:, 3, :])
        # import pickle
        # with open('debug_dict.pkl', 'wb') as f:
        #     pickle.dump({'original_tokens' : x.detach().cpu(),
        #                  'struct_embed': struct_embed.detach().cpu(),
        #                  'atom_embed': atom_embed.detach().cpu(),
        #                  'valency_embed': valency_embed.detach().cpu(),
        #                  'hybridization_embed': hybridization_embed.detach().cpu()}, f)
        # import pdb; pdb.set_trace()
        if self.structure_embed_args["embedding_form"] == "struct_avh_sum":
            return struct_embed + atom_embed + valency_embed + hybridization_embed
        elif self.structure_embed_args["embedding_form"] == "struct_avh_concat":
            return torch.cat(
                (struct_embed, atom_embed, valency_embed, hybridization_embed), dim=-1
            )


class SeqStructMixedTransformer(nn.Module):
    def __init__(
        self,
        dim_model: int,
        token_info: dict[str, int],
        unified_transformer_args: dict,
        embedding_args: dict,
        forward_mode: str,
    ):
        """Generic transformer model for mixed token training"""
        super().__init__()
        # Check some assertions
        contains_smi = bool(token_info["input"]["TOK"])
        contains_struct = bool(token_info["input"]["STRUCT"])
        assert contains_smi or contains_struct, (
            "At least one of the token embeddings should be present!"
        )
        assert "embedding_combination" in embedding_args, (
            "Must specify how to combine embeddings to use!"
        )
        assert "struct_embedding_args" in embedding_args, (
            "Must specify arguments on how to embed the structure tokens!"
        )

        # Save information
        self.forward_mode = forward_mode
        self.embedding_args = embedding_args
        self.token_info = token_info
        self.dim_model = dim_model

        if self.embedding_args["embedding_combination"] == "sum":
            # Always assumed the first two arguments are the smiles embeddings and structure embeddings
            self.connector = SequenceSummator(token_info)
        elif self.embedding_args["embedding_combination"] == "concat":
            self.connector = SequenceConcatenator(token_info)
        else:
            raise ValueError("Invalid embedding combination method specified!")

        # FH: Typical embedding and associated regression head for the smiles tokens
        if contains_smi:
            self.smi_token_embedding = nn.Embedding(
                embedding_dim=dim_model,
                num_embeddings=token_info["input"]["TOK"]["TOK_MAX_SIZE"],
                padding_idx=token_info["input"]["TOK"]["TOK_PAD"],
            )
            # FH: Update token aggregation to include MAX_SIZE
            self.smi_regression_head = RegressionHead(
                dim_model, token_info["input"]["TOK"]["TOK_MAX_SIZE"]
            )
        else:
            self.smi_token_embedding = None
            self.smi_regression_head = None

        # FH: The structure embedding depends on the arguments passed in
        if contains_struct:
            self.structural_token_embedding = self._get_structural_embedding()
            self.structural_regression_head = RegressionHead(
                dim_model, token_info["input"]["STRUCT"]["STRUCT_MAX_SIZE"]
            )
        else:
            self.structural_token_embedding = None
            self.structural_regression_head = None

        self.transformer = TransformerStack(
            d_model=dim_model, **unified_transformer_args
        )

    def _get_structural_embedding(self):
        structure_embedding_args = self.embedding_args["struct_embedding_args"]
        embedding_form = structure_embedding_args["embedding_form"]

        if embedding_form == "struct_only":
            structural_token_embedding = nn.Embedding(
                embedding_dim=self.dim_model,
                num_embeddings=self.token_info["input"]["STRUCT"]["STRUCT_MAX_SIZE"],
                padding_idx=self.token_info["input"]["STRUCT"]["STRUCT_PAD"],
            )

            # FH: Load the weights for the structural token embedding from a pre-defined tensor
            if ("struct_embedding_init_weight" in structure_embedding_args) and (
                structure_embedding_args["struct_embedding_init_weight"] is not None
            ):
                embed_weight_path = structure_embedding_args[
                    "struct_embedding_init_weight"
                ]
                embed_weight = torch.load(embed_weight_path)
                if embed_weight.shape == self.structural_token_embedding.weight.shape:
                    structural_token_embedding.weight.data = embed_weight
                else:
                    # Still partially load the weights for the embedding
                    structural_token_embedding.weight.data[: embed_weight.shape[0]] = (
                        embed_weight
                    )

            return structural_token_embedding

        elif embedding_form in ["struct_avh_sum", "struct_avh_concat"]:
            return StructAVH(
                d_model=self.dim_model,
                token_info=self.token_info,
                structure_embed_args=structure_embedding_args,
            )

    def forward(self, batch):
        """Forward pass with differing behavior depending on batch structure"""
        mode = batch[0]
        if mode == "smiles_only":
            _, smiles_tokens, sequence_id = batch
            embedding = self.smi_token_embedding(smiles_tokens)
            x, _ = self.transformer(
                embedding,
                # Here, just have a different number between tokens and mask (e.g. 0 for tokens, -1 for padding)
                sequence_id,
                affine=None,
                affine_mask=None,
            )
            x = self.smi_regression_head(x)
            return (x, self.token_info["input"]["TOK"]["TOK_PAD"])
        elif mode in ["smiles_and_struct", "smiles_struct_avh"]:
            _, smiles_tokens, structural_tokens, sequence_id = batch
            smi_embedding = self.smi_token_embedding(smiles_tokens)
            struct_embedding = self.structural_token_embedding(structural_tokens)
            embedding = self.connector(
                smiles_tokens, structural_tokens, smi_embedding, struct_embedding
            )
            # (N, T, E)
            x, _ = self.transformer(
                embedding, sequence_id, affine=None, affine_mask=None
            )
            if self.embedding_args["embedding_combination"] == "sum":
                smi_out = self.smi_regression_head(x)
                struct_out = self.structural_regression_head(x)
            elif self.embedding_args["embedding_combination"] == "concat":
                n, t, e = x.shape
                x_flat = x.reshape(n * t, e)
                smi_token_flat = smiles_tokens.reshape(-1)

                # FH: Need to do an OR selection in the autoregressive case because the smiles stop token
                #   should map to the first structural token due to the right-shifting
                # This autoregressive formulation relies implicitly on the fact that smiles sequence stop tokens
                #   always occur right before structure tokens in the sequence
                if self.forward_mode == "masking":
                    structure_selection_mask = (
                        smi_token_flat == self.token_info["input"]["STRUCT"]["STRUCT"]
                    )
                elif self.forward_mode == "autoregressive":
                    structure_selection_mask = (
                        smi_token_flat == self.token_info["input"]["STRUCT"]["STRUCT"]
                    ) | (smi_token_flat == self.token_info["input"]["TOK"]["TOK_STOP"])

                struct_out = x_flat[structure_selection_mask]
                smi_out = self.smi_regression_head(x)  # (N, T, E)
                struct_out = self.structural_regression_head(struct_out)  # (N*S, E)

            return (
                smi_out,
                struct_out,
                self.token_info["input"]["TOK"]["TOK_PAD"],
                self.token_info["input"]["STRUCT"]["STRUCT_PAD"],
            )

        elif mode == "ida":
            _, smiles_tokens, structural_tokens, coords, sequence_id = batch

            smi_embedding = self.smi_token_embedding(smiles_tokens)
            struct_embedding = self.structural_token_embedding(structural_tokens)
            embedding = self.connector(
                smiles_tokens, structural_tokens, smi_embedding, struct_embedding
            )

            # Generate coordinate mask based on forward mode
            if coords is not None:
                if self.forward_mode == "autoregressive":
                    # Autoregressive: only allow coordinates for structural tokens
                    # Causal masking is handled by the transformer's attention mechanism
                    coords_mask_BL = (
                        smiles_tokens != self.token_info["input"]["STRUCT"]["STRUCT"]
                    )
                    # Shape (B, L, L) of all pairwise logical or (true when i and j both are not structure tokens)
                    coords_mask_BLL = coords_mask_BL.unsqueeze(
                        -1
                    ) & coords_mask_BL.unsqueeze(-2)
                    # Shape (B, H, L, L)
                    coords_mask = coords_mask_BLL.unsqueeze(1)
                else:
                    raise ValueError(
                        "Coordinates are only supported in autoregressive mode!"
                    )
            else:
                coords_mask = None

            # (N, T, E)
            x, _ = self.transformer(
                embedding,
                sequence_id,
                coords=coords,
                coords_mask=coords_mask,
                affine=None,
                affine_mask=None,
            )

            # Handle different embedding combination strategies
            if self.embedding_args["embedding_combination"] == "sum":
                smi_out = self.smi_regression_head(x)
                struct_out = self.structural_regression_head(x)
            elif self.embedding_args["embedding_combination"] == "concat":
                n, t, e = x.shape
                x_flat = x.reshape(n * t, e)
                smi_token_flat = smiles_tokens.reshape(-1)

                # FH: Need to do an OR selection in the autoregressive case because the smiles stop token
                #   should map to the first structural token due to the right-shifting
                # This autoregressive formulation relies implicitly on the fact that smiles sequence stop tokens
                #   always occur right before structure tokens in the sequence
                if self.forward_mode == "masking":
                    structure_selection_mask = (
                        smi_token_flat == self.token_info["input"]["STRUCT"]["STRUCT"]
                    )
                elif self.forward_mode == "autoregressive":
                    structure_selection_mask = (
                        smi_token_flat == self.token_info["input"]["STRUCT"]["STRUCT"]
                    ) | (smi_token_flat == self.token_info["input"]["TOK"]["TOK_STOP"])

                struct_out = x_flat[structure_selection_mask]
                smi_out = self.smi_regression_head(x)  # (N, T, E)
                struct_out = self.structural_regression_head(struct_out)  # (N*S, E)

            return (
                smi_out,
                struct_out,
                self.token_info["input"]["TOK"]["TOK_PAD"],
                self.token_info["input"]["STRUCT"]["STRUCT_PAD"],
            )
