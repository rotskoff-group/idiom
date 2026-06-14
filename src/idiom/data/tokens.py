"""IDiom token-id semantics: which positions an SAE was actually trained on.

IDiom's activation extraction (``idiom .../extract_activations.py``) drops the control
tokens (PAD / START / STOP / MASK) **and** the FIM markers ``'1'`` / ``'2'`` / ``'3'``,
keeping only real amino-acid residue positions. SAEs are therefore trained purely on
residue activations.

Substituting or steering the residual stream at the dropped positions feeds the SAE
out-of-distribution inputs, and because IDiom's attention is causal, that corruption at
the front of a sequence (START / FIM markers) propagates to every downstream token. So
fidelity eval and feature steering should both confine their edits to residue positions.
These helpers identify those positions from ``token_info``.
"""

from __future__ import annotations

import torch

# FIM (fill-in-the-middle) marker characters, dropped from SAE training data alongside
# the control tokens. The 20 amino-acid characters are everything else in the alphabet.
FIM_MARKERS: tuple[str, ...] = ("1", "2", "3")


def _alpha_char(a: object) -> str:
    """Decode one alphabet entry (stored as utf-8 bytes in idiom h5 files) to a str."""
    return a.decode() if isinstance(a, (bytes, bytearray)) else str(a)


def residue_token_ids(token_info: dict) -> list[int]:
    """Token ids of real amino-acid residues — the positions the SAE was trained on.

    These are the ``token_info["alphabet"]`` entries that are not FIM markers. Control
    tokens (PAD/START/STOP/MASK) have ids beyond the alphabet range, so they are excluded
    automatically.
    """
    alphabet = token_info["alphabet"]
    return [i for i, a in enumerate(alphabet) if _alpha_char(a) not in FIM_MARKERS]


def residue_position_mask(res_tokens: torch.Tensor, token_info: dict) -> torch.Tensor:
    """Boolean ``[B, L]`` mask, ``True`` where the input token is a real residue.

    Args:
        res_tokens: ``[B, L]`` long tensor of input token ids (idiom ``res_tokens``).
        token_info: idiom token metadata (provides the residue alphabet).
    """
    ids = torch.tensor(residue_token_ids(token_info), device=res_tokens.device)
    return torch.isin(res_tokens, ids)
