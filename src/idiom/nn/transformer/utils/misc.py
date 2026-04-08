import torch
from idiom.nn.transformer.nn import GeometricMolTransformer


def compute_policy_logps(model, tokens, structure, masks):
    """Calculate per-token logps under a provided model

    Args:
        model: The model to compute logps with
        tokens: Token indices
        structure: Structural information
        masks: Attention masks

    Returns:
        policy_logps: Per-token log probabilities (1 shorter than len(tokens) because start token doesn't have a logp)
    """
    if isinstance(model, GeometricMolTransformer):
        policy_logits = model(tokens, structure, masks)

    policy_logps = policy_logits.log_softmax(dim=-1)
    policy_logps = torch.gather(
        policy_logps[:, :-1, :], dim=-1, index=tokens[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    # policy_logps is going to be 1 shorter than len(tokens) because start token doesn't have a logp
    return policy_logps
