import torch
import torch.nn as nn

from .rpe import RelativePositionEmbedding
from .ppe import PairwisePredictionHead
from .structure_proj import Dim6RotStructureHead
from .regression_head import RegressionHead
from .transformer_stack import TransformerStack
from .codebook import EMACodebook

from clm.utils import Affine3D, build_affine3d_from_coordinates


def knn_graph(ca_coords, coord_mask, padding_mask, sequence_id, num_knn, max_dist=1e6):
    """Args:
        ca_coords: (B, S, 3)
        coord_mask: (B, S)
        padding_mask: (B, S)
        sequence_id: (B, S)
        num_knn: int
    Returns:
        chosen_edges: (B, S, num_knn)
        chosen_mask: (B, S, num_knn)
    """
    device = ca_coords.device
    S = ca_coords.shape[-2]
    num_knn = min(num_knn, S)

    ca_coords = ca_coords.nan_to_num()
    coord_mask = ~(coord_mask[..., None, :] & coord_mask[..., :, None])
    padding_pairwise_mask = padding_mask[..., None, :] | padding_mask[..., :, None]
    if sequence_id is not None:
        padding_pairwise_mask |= sequence_id.unsqueeze(1) != sequence_id.unsqueeze(2)

    # Mask has shape (B, S, S)

    dists = (ca_coords.unsqueeze(-2) - ca_coords.unsqueeze(-3)).norm(dim=-1)
    arange = torch.arange(S, device=device)
    seq_dists = (arange.unsqueeze(-1) - arange.unsqueeze(-2)).abs()
    # We only support up to a certain distance, above that, we use sequence distance
    # instead. This is so that when a large portion of the structure is masked out,
    # the edges are built according to sequence distance.

    torch._assert_async((dists[~coord_mask] < max_dist).all())

    # (B, S, S)
    struct_then_seq_dist = (
        seq_dists.to(dists.dtype)
        .mul(1e2)
        .add(max_dist)
        .where(coord_mask, dists)
        .masked_fill(padding_pairwise_mask, torch.inf)
    )
    dists, edges = struct_then_seq_dist.sort(dim=-1, descending=False)
    # This is a S x S tensor, where we index by rows first,
    # and columns are the edges we should pick.
    # (B, S, num_knn)
    chosen_edges = edges[..., :num_knn]
    chosen_mask = dists[..., :num_knn].isfinite()
    return chosen_edges, chosen_mask


def batched_gather(data, inds, dim=0, num_batch_dims=0):
    """
    This function is a generalization of the gather operation
    """
    ranges = []
    # Reshape the indices to match the data shape
    # (-1, 1, 1)
    # (1, -1, 1)
    # (1, 1, -1)
    # (1, 1, 1, -1)
    for i, s in enumerate(data.shape[:num_batch_dims]):
        r = torch.arange(s)

        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - num_batch_dims)]
    remaining_dims[dim - num_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def node_gather(s, edges):
    """
    Args:
        s: (B, S, 12)
        edges: (B, S, E)
    Returns:
        (B, S, E, 12)
    """
    return batched_gather(
        data=s.unsqueeze(-3),  # (B, 1, S, 12)
        inds=edges,
        dim=-2,
        num_batch_dims=len(s.shape) - 1,
    )


class StructureTokenEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_out,
        n_codes,
        num_knn,
        max_dist,
        unified_transformer_args,
        rpe_args,
    ):
        super().__init__()
        # We only support fully-geometric structure token encoders for now...
        # setting n_layers_geom to something that's not n_layers won't work because
        # sequence ID isn't supported fully in this repo for plain-old transformers
        self.transformer = TransformerStack(d_model=d_model, **unified_transformer_args)
        self.pre_vq_proj = nn.Linear(d_model, d_out)
        self.codebook = EMACodebook(n_codes, d_out)
        self.relative_positional_embedding = RelativePositionEmbedding(
            d_model=d_model, **rpe_args
        )
        self.num_knn = num_knn
        self.max_dist = max_dist

    def find_knn_edges(
        self, coords, coord_mask, padding_mask, sequence_id=None, num_knn=None
    ):
        # Coords are N, CA, C
        coords = coords.clone()
        coords[~coord_mask] = 0

        if sequence_id is None:
            sequence_id = torch.zeros(
                (coords.shape[0], coords.shape[1]), device=coords.device
            ).long()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):  # type: ignore
            ca_coords = coords[..., 1, :]
            edges, edge_mask = knn_graph(
                ca_coords=ca_coords,
                coord_mask=coord_mask,
                padding_mask=padding_mask,
                sequence_id=sequence_id,
                num_knn=num_knn,
                max_dist=self.max_dist,
            )

        return edges, edge_mask

    def encode_local_structure(
        self, coords, affine, affine_mask, attention_mask, sequence_id, residue_index
    ):
        """This function allows for a multi-layered encoder to encode tokens with a local receptive fields. The implementation is as follows:

        1. Starting with (B, L) frames, we find the KNN in structure space. This now gives us (B, L, K) where the last dimension is the local
        neighborhood of all (B, L) residues.
        2. We reshape these frames to (B*L, K) so now we have a large batch of a bunch of local neighborhoods.
        3. Pass the (B*L, K) local neighborhoods through a stack of geometric reasoning blocks, effectively getting all to all communication between
        all frames in the local neighborhood.
        4. This gives (B*L, K, d_model) embeddings, from which we need to get a single embedding per local neighborhood. We do this by simply
        taking the embedding corresponding to the query node. This gives us (B*L, d_model) embeddings.
        5. Reshape back to (B, L, d_model) embeddings
        Args:
            coords: (B, S, 3, 3) coordinates
            affine (Affine3D): contains:
                trans (torch.Tensor): The translational portion of the affine transform
                rot (torch.Tensor): The rotational portion of the affine transform
            affine_mask (torch.Tensor): The affine mask tensor or None. (B, S)
            sequence_id (torch.Tensor): The sequence ID tensor of shape (B, S)
            attention_mask (B, S)
            residue_index: (B, S)
        """
        assert coords.size(-1) == 3 and coords.size(-2) == 3, "need N, CA, C"
        with torch.no_grad():
            knn_edges, _ = self.find_knn_edges(
                coords,
                coord_mask=affine_mask,
                padding_mask=~attention_mask,
                sequence_id=sequence_id,
                num_knn=self.num_knn,
            )
            # E = min(num_knn, L)
            B, S, E = knn_edges.shape

            # (B, S, ..., 12)
            affine_tensor = affine.get_T()  # for easier manipulation
            T_D = affine_tensor.size(-1)  # 12
            # (B, S, ..., E, 12)
            knn_affine_frame_tensor = node_gather(affine_tensor, knn_edges)
            # (-1, E, 12)
            knn_affine_frame_tensor = knn_affine_frame_tensor.view(
                -1, E, T_D
            ).contiguous()

            affine = Affine3D.from_frame(T=knn_affine_frame_tensor)

            if sequence_id is not None:
                # (B, S, E, 1)
                knn_sequence_id = node_gather(sequence_id.unsqueeze(-1), knn_edges)
                # (-1, E)
                knn_sequence_id = knn_sequence_id.view(-1, E)
            else:
                knn_sequence_id = torch.zeros(
                    S, E, dtype=torch.int64, device=coords.device
                )

            # (B, S, E, 1)
            knn_affine_mask = node_gather(affine_mask.unsqueeze(-1), knn_edges)
            # (- 1, E)
            knn_affine_mask = knn_affine_mask.view(-1, E)

            knn_chain_id = torch.zeros(S, E, dtype=torch.int64, device=coords.device)

            if residue_index is None:
                res_idxs = knn_edges.view(-1, E)
            else:
                # (B, S, E, 1)
                res_idxs = node_gather(residue_index.unsqueeze(-1), knn_edges)
                # (-1, E)
                res_idxs = res_idxs.view(-1, E)

        z = self.relative_positional_embedding(res_idxs[:, 0], res_idxs)

        z, _ = self.transformer.forward(
            x=z, sequence_id=knn_sequence_id, affine=affine, affine_mask=knn_affine_mask
        )

        # Unflatten the output and take the query node embedding, which will always be the first one because
        # a node has distance 0 with itself and the KNN are sorted.
        z = z.view(B, S, E, -1)
        z = z[:, :, 0, :]
        return z

    def encode(self, coords, attention_mask=None, sequence_id=None, residue_index=None):
        """
        coords: (B, S, 3, 3) coordinates
        sequence_id (torch.Tensor): The sequence ID tensor of shape (B, S)
        attention_mask (B, S)
        residue_index: (B, S)
        """
        coords = coords[..., :3, :]

        affine, affine_mask = build_affine3d_from_coordinates(coords=coords)

        if attention_mask is None:
            attention_mask = torch.ones_like(affine_mask, dtype=torch.bool)
        attention_mask = attention_mask.bool()

        if sequence_id is None:
            sequence_id = torch.zeros_like(affine_mask, dtype=torch.int64)

        z = self.encode_local_structure(
            coords=coords,
            affine=affine,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            affine_mask=affine_mask,
            residue_index=residue_index,
        )

        z = z.masked_fill(~affine_mask.unsqueeze(2), 0)
        z = self.pre_vq_proj(z)

        z_q, min_encoding_indices, _ = self.codebook(z)

        return z_q, min_encoding_indices


class StructureTokenDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        n_codes,
        special_tokens,
        max_pae_bin,
        direction_loss_bins,
        pae_bins,
        unified_transformer_args,
        output_proj_args,
        ppe_args,
        rh_args,
    ):
        super().__init__()
        self.decoder_channels = d_model

        self.n_codes = n_codes
        self.special_tokens = special_tokens
        self.max_pae_bin = max_pae_bin

        self.embed = nn.Embedding(self.n_codes + len(self.special_tokens), d_model)

        self.decoder_stack = TransformerStack(
            d_model=d_model, **unified_transformer_args
        )

        self.affine_output_projection = Dim6RotStructureHead(
            input_dim=d_model, **output_proj_args
        )

        self.pairwise_bins = [64, direction_loss_bins * 6, pae_bins]

        self.pairwise_classification_head = PairwisePredictionHead(
            input_dim=d_model, n_bins=sum(self.pairwise_bins), **ppe_args
        )

        self.plddt_head = RegressionHead(d_model=d_model, **rh_args)

    def decode(self, structure_tokens, attention_mask=None, sequence_id=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(structure_tokens, dtype=torch.bool)
        attention_mask = attention_mask.bool()
        if sequence_id is None:
            sequence_id = torch.zeros_like(structure_tokens, dtype=torch.int64)

        # check that BOS and EOS are set correctly
        assert structure_tokens[:, 0].eq(self.special_tokens["BOS"]).all(), (
            "First token in structure_tokens must be BOS token"
        )
        assert (
            structure_tokens[
                torch.arange(structure_tokens.shape[0]), attention_mask.sum(1) - 1
            ]
            .eq(self.special_tokens["EOS"])
            .all()
        ), "Last token in structure_tokens must be EOS token"
        assert (structure_tokens < 0).sum() == 0, (
            "All structure tokens set to -1 should be replaced with BOS, EOS, PAD, or MASK tokens by now, but that isn't the case!"
        )

        x = self.embed(structure_tokens)
        # !!! NOTE: Attention mask is actually unused here so watch out (use sequence_id for padding)
        x, _ = self.decoder_stack(
            x, sequence_id=sequence_id, affine=None, affine_mask=None
        )

        # The make here doesn't really matter because each frame gets generated independently
        tensor7_affine, bb_pred = self.affine_output_projection(
            x, affine=None, affine_mask=~attention_mask
        )

        pae, ptm = None, None
        pairwise_logits = self.pairwise_classification_head(x)

        # #Distogram logita
        # distogram_logits, direction_logits, pae_logits = [(o if o.numel() > 0 else None)
        #                                                   for o in pairwise_logits.split(self.pairwise_bins, dim=-1)]

        # special_tokens_mask = (structure_tokens
        #                        >= min(self.special_tokens.values()))
        # pae = compute_predicted_aligned_error(pae_logits,
        #                                       aa_mask=~special_tokens_mask,
        #                                       max_bin=self.max_pae_bin)
        # # This might be broken for chainbreak tokens? We might align to the chainbreak
        # ptm = compute_tm(pae_logits,
        #                  aa_mask=~special_tokens_mask,
        #                  max_bin=self.max_pae_bin)

        # plddt_logits = self.plddt_head(x)
        # plddt_value = CategoricalMixture(plddt_logits,
        #                                  bins=plddt_logits.shape[-1]).mean()

        return dict(
            tensor7_affine=tensor7_affine,
            pairwise_logits=pairwise_logits,
            bb_pred=bb_pred,
        )
        # plddt=plddt_value,
        # ptm=ptm,
        # predicted_aligned_error=pae)


class VQVAE(nn.Module):
    def __init__(self, encoder_args, decoder_args):
        super().__init__()
        self.encoder = StructureTokenEncoder(**encoder_args)
        self.decoder = StructureTokenDecoder(**decoder_args)

    def forward(
        self, coords, attention_mask=None, sequence_id=None, residue_index=None
    ):
        z_q, min_encoding_indices = self.encoder.encode(
            coords, attention_mask, sequence_id, residue_index
        )
        return self.decoder.decode(z_q, attention_mask, sequence_id)
