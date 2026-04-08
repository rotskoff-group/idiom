import torch
import torch.nn as nn


class RelativePositionEmbedding(nn.Module):
    """
    Embedding layer for relative position embeddings. `bins` is the number of positions relative
    to the query position that are considered before clipping. For instance, if `bins=10`, then
    the relative position embedding will have 21 positions, [-10, 10].
    """

    def __init__(self, d_model, bins, init_std=0.02):
        super().__init__()
        self.bins = bins

        self.embedding = torch.nn.Embedding(2 * bins + 2, d_model)
        self.embedding.weight.data.normal_(0, init_std)

    def forward(self, query_residue_index, key_residue_index):
        """
        Input:
          query_residue_index: (B, ) tensor of source indices
          key_residue_index: (B, L) tensor of target indices (dytpe=torch.long)
        Output:
          embeddings: B x L x d_model tensor of embeddings
        """

        diff = key_residue_index - query_residue_index.unsqueeze(1)
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # add 1 to adjust for padding index
        output = self.embedding(diff)
        return output
