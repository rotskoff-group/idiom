import torch.nn as nn
from idr_plm.nn.layers import TransformerStack, RegressionHead


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
