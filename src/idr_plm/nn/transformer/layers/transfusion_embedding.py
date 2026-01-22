import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, feat_dim: int) -> None:
        """
        Generates an embedding of time steps through a linear transformation

        Args:
            feat_dim: int
                The dimension to embed the time step into
        """
        super().__init__()

        self.dim = feat_dim
        self.embed = nn.Linear(1, feat_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: torch.Tensor
                (B, 1) tensor of time steps

        Returns:
            embedded: torch.Tensor
                (B, E) tensor of embedded tiem steps
        """
        embedded = self.embed(t)
        return embedded


class DiscreteDihedralEmbedding(nn.Module):
    """
    Bins dihedral angles and embeds based on bin indices

    Dihedral angles tend to span from -pi to pi radians, so we can bin them and pass through a
    nn.embed layer

    Args:
        d_model: int
            The dimension of the embedding
        num_bins: int
            The number of bins to use for discretizing the dihedral range
    """

    def __init__(self, d_model: int, num_bins: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_bins + 1, d_model)
        # Add a buffer to the minimum and maximum boundaries
        self.bins = torch.linspace(-torch.pi - 0.1, torch.pi + 0.1, num_bins)

    def forward(
        self, dihedrals: torch.Tensor, structure_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dihedrals: torch.Tensor
                (B, T), tensor of dihedral angles
            structure_mask: torch.Tensor
                (B, T), mask for removing padding in dihedral values

        Returns:
            embdded: torch.Tensor
                (B, T, E) tensor of embedded dihedral angles
        """
        # Bin the dihedrals
        binned_indices = torch.bucketize(dihedrals, self.bins) - 1
        embedded = self.embed(binned_indices)
        embedded[structure_mask] = 0
        return embedded  # (B, T, E)


class ContinuousDihedralEmbedding(nn.Module):
    """
    Featurizes the dihedral angles as sin and cosines, as done in https://pubs.acs.org/doi/full/10.1021/acs.jpcb.3c08195

    Args:
        d_model: int
            The dimension of the embedding
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Linear(2, d_model)
        self.time_embedding = TimeEmbedding(2)

    def forward(
        self, dihedrals: torch.Tensor, structure_mask: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dihedrals: torch.Tensor
                (B, T, 2), tensor of dihedral angles in sine and cosine format
            structure_mask: torch.Tensor
                (B, T), mask for removing padding in dihedral values
            t: torch.Tensor
                (B, 1) tensor of time steps

        Returns:
            embedded: torch.Tensor
                (B, T, E) tensor of embedded dihedral angles
        """
        t = t.reshape(-1, 1, 1)
        time_embedding = self.time_embedding(t)  # (B, T, 2)
        angle_feat = dihedrals + time_embedding
        embedded = self.embed(angle_feat)  # (B, T, E)
        embedded[structure_mask] = 0
        return embedded


class MultiheadDihedralEmbedding(nn.Module):
    """
    Featurizes the dihedral angles using a set of linear transformations

    Args:
        d_model: int
            The dimension of the embedding
        d_input: int
            The input dimension of the data, i.e. the number of dihedral angles
            before the sine-cosine transformation
        nhead: int
            The number of separate linear transformations (heads) that go from
            d_input to d_model. Default is 16

    Notes:
        This featurization was found to work well when tested in isolation with a transformer encoder
        model trying to learn the dihedral angles from conformations of a single molecule. Based on testing,
        the MultiheadDihedralEmbedding also performs a sine-cosine transformation on the angles to enforce
        periodicity. The input dimension is therefore 2 * num_dihedrals + 1 for concatenating the time step
    """

    def __init__(self, d_model: int, d_input: int, nhead: int = 16) -> None:
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.heads = nn.ModuleList(
            [nn.Linear(d_input * 2 + 1, d_model) for _ in range(nhead)]
        )
        self.w = nn.Linear(nhead, d_input)

    def forward(
        self, x: torch.Tensor, structure_mask: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor
                (B, d_input) tensor of dihedral angles
            structure_mask: torch.Tensor
                (B, d_input) tensor indicating where to mask out dihedrals. Not actually used here since
                the input features have been transformed through a sine-cosine transformation and all
                features are used for the MLP heads
            t: torch.Tensor
                (B, 1) tensor of time steps for each example in the batch

        Returns:
            out: torch.Tensor
                The concatenated output of all the heads (B, d_input, d_model)
        """
        # Construct the sine-cosine pre-tensors
        sine_pre = x.clone()
        sine_pre[structure_mask] = torch.pi
        cosine_pre = x.clone()
        cosine_pre[structure_mask] = torch.pi / 2
        x = torch.cat(
            [torch.sin(sine_pre), torch.cos(cosine_pre)], dim=-1
        )  # (B, d_input * 2)
        # Concatenate the time step
        t = t.reshape(-1, 1)
        x = torch.cat([x, t], dim=1)  # (B, d_input * 2 + 1)
        out = [head(x) for head in self.heads]
        out = torch.stack(out, dim=1)  # (B, nhead, d_model)
        out = out.permute(0, 2, 1)  # (B, d_model, nhead)
        out = self.w(out)  # (B, d_model, d_input)
        return out.permute(0, 2, 1)  # (B, d_input, d_model)


class TransfusionEmbedding(nn.Module):
    """
    Combines the token and structure embeddings for use in one transformer model

    The SMILES tokens are embedded using a standard embedding layer, dihedral structures
    are embedded using a continuous linear transformation, similar to what is done in
    https://pubs.acs.org/doi/full/10.1021/acs.jpcb.3c08195

    The token_metadata contains information about which token is used to indicate
    structural information which is necessary for assembling the final input to the transformer
    stack when training transfusion

    Args:
        d_model: int
            The dimension of the embedding
        num_tokens: int
            The number of tokens in the vocabulary
        token_metadata: dict
            TODO: update docstring
        struct_embedding_opts: dict
            TODO: update docstring
    """

    def __init__(
        self, d_model: int, token_metadata: dict, struct_embedding_opts: dict
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.struct_embedding_opts = struct_embedding_opts
        self.struct_embedding_mode = self.struct_embedding_opts["mode"]
        structure_embedding_args = struct_embedding_opts["args"]
        if self.struct_embedding_mode == "continuous":
            self.dihedral_embedding = ContinuousDihedralEmbedding(
                d_model, **structure_embedding_args
            )
        elif self.struct_embedding_mode == "discrete":
            self.dihedral_embedding = DiscreteDihedralEmbedding(
                d_model, **structure_embedding_args
            )
        elif self.struct_embedding_mode == "multihead":
            self.dihedral_embedding = MultiheadDihedralEmbedding(
                d_model, **structure_embedding_args
            )
        self.token_metadata = token_metadata
        num_tokens = self.token_metadata["TOTAL"]
        self.token_embedding = nn.Embedding(num_tokens, d_model)

    def forward(
        self,
        tokenized_input: torch.Tensor,
        structure_input: torch.Tensor,
        structure_mask: torch.Tensor,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokenized_input: torch.Tensor
                (B, N) tensor of tokenized inputs
            structure_input: torch.Tensor
                tensor of dihedral angles, shape (B, T)
            structure_mask: torch.Tensor
                (B, T) mask tensor for dihedral angles
            ts: torch.Tensor
                (B, 1) tensor of time steps for the diffusion process

        Notes:
            The tokenized input has sections blocked out for the structural information,
            just have to assemble them correctly
        """
        token_embedding = self.token_embedding(tokenized_input)  # (B, N, E)
        if structure_input is not None:
            structure_embedding = self.dihedral_embedding(
                structure_input, structure_mask, ts
            )  # (B, T, E)

            # Replace the locations in token embedding corresponding to structure with structure embedding
            # Select out the valid structure embeddings
            # import pdb; pdb.set_trace()
            b, n, e = token_embedding.shape
            struct_reshape = structure_embedding.reshape(-1, e)  # (B * T, E)
            struct_mask_reshape = structure_mask.reshape(-1)  # (B * T)
            valid_struct_embeddings = struct_reshape[~struct_mask_reshape]

            # Select out the locations in the tokenized input that need to be replaced
            struct_token_select = (
                tokenized_input == self.token_metadata["input"]["STRUCT"]["STRUCT"]
            )  # (B, N)
            struct_token_select_reshape = struct_token_select.reshape(-1)  # (B * N)
            token_emb_reshape = token_embedding.reshape(-1, e)  # (B * N, E)
            # import pdb; pdb.set_trace()
            # Put the structure embeddings into the correct locations
            assert struct_token_select_reshape.sum() == valid_struct_embeddings.shape[0]
            token_emb_reshape[struct_token_select_reshape] = (
                valid_struct_embeddings.float()
            )  # Hard-coded data type for now

            token_emb_fixed = token_emb_reshape.reshape(b, n, e)  # (B, N, E)
        else:
            # If no structural information is provided, just use next-token prediction
            #    used only for inference
            token_emb_fixed = token_embedding
        return token_emb_fixed
