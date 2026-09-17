"""Fixed-alphabet character tokenizer.

Id map (vocab size 27):

    0..19     amino acids, in the order of RESIDUES
    20,21,22  FIM markers 1, 2, 3   (1=prefix, 2=middle/IDR, 3=suffix)
    23 PAD    24 START    25 STOP    26 MASK
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

# Residue order fixes the trained vocabulary; changing it invalidates model weights
RESIDUES = "ACDEFGHIKLMNPQRSTVWY"
RESIDUE_SET = frozenset(RESIDUES)
FIM = "123"
SPECIALS = ("<pad>", "<start>", "<stop>", "<mask>")


class Tokenizer:
    """Character-level tokenizer over the fixed RESIDUES + FIM + SPECIALS alphabet.

    Attributes:
        n_residues (int): Number of amino-acid tokens (20).
        n_fim (int): Number of FIM marker tokens (3).
        vocab_size (int): Total number of tokens (27).
        pad_id (int): Token id of <pad>.
        start_id (int): Token id of <start>.
        stop_id (int): Token id of <stop>.
        mask_id (int): Token id of <mask>.
        fim_prefix_id (int): Token id of the "1" marker.
        fim_middle_id (int): Token id of the "2" marker.
        fim_suffix_id (int): Token id of the "3" marker.
    """

    def __init__(self) -> None:
        """Build residue and special-token lookup tables and expose their token IDs."""
        self._itos: list[str] = list(RESIDUES) + list(FIM) + list(SPECIALS)
        self._stoi: dict[str, int] = {c: i for i, c in enumerate(self._itos)}
        self._char2id: dict[str, int] = {c: self._stoi[c] for c in RESIDUES + FIM}

        self.n_residues = len(RESIDUES)
        self.n_fim = len(FIM)
        self.vocab_size = len(self._itos)
        self.pad_id = self._stoi["<pad>"]
        self.start_id = self._stoi["<start>"]
        self.stop_id = self._stoi["<stop>"]
        self.mask_id = self._stoi["<mask>"]
        self.fim_prefix_id = self._stoi["1"]
        self.fim_middle_id = self._stoi["2"]
        self.fim_suffix_id = self._stoi["3"]

    def is_canonical(self, seq: str) -> bool:
        """Return whether seq is non-empty and contains only uppercase canonical amino acids."""
        return bool(seq) and all(c in RESIDUE_SET for c in seq)

    def encode(self, s: str) -> list[int]:
        """Map a residue/FIM string to token ids, adding no control tokens.

        Args:
            s: A string of residue and/or FIM-marker characters.

        Returns:
            The token id for each character, in order.

        Raises:
            ValueError: If s contains a character that is neither a residue nor a FIM marker.
        """
        try:
            return [self._char2id[c] for c in s]
        except KeyError as e:
            raise ValueError(
                f"non-tokenizable character {e.args[0]!r} in sequence; non-canonical "
                f"sequences must be dropped before encoding (see Tokenizer.is_canonical)"
            ) from None

    def decode(self, ids: Iterable[int]) -> str:
        """Decode token ids to a residue/FIM string, omitting control tokens."""
        n_seq = self.n_residues + self.n_fim
        return "".join(self._itos[int(i)] for i in ids if int(i) < n_seq)

    def is_residue(self, i: int) -> bool:
        """Check whether a token ID lies below the residue vocabulary boundary."""
        return int(i) < self.n_residues

    def is_fim(self, i: int) -> bool:
        """Check whether a token ID belongs to the FIM marker range."""
        return self.n_residues <= int(i) < self.n_residues + self.n_fim

    def residue_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask of residue positions, with the same shape as ids."""
        return self.region_mask(ids)

    def region_mask(
        self, ids: torch.Tensor, *, region: str = "all", drop_markers: bool = True
    ) -> torch.Tensor:
        """Return a boolean [B, L] mask of positions selected by token class and region.

        Control tokens are always dropped. region then restricts by position relative to the FIM
        "2" marker that opens the IDR.

        Args:
            ids: Token ids, shape [B, L].
            region: Which kept positions to select: "all" for every kept position, "idr" for only
                those after the "2" marker, "non_idr" for only those before it. A row with no "2"
                marker contributes nothing to "idr" or "non_idr".
            drop_markers: If True, keep only real residues; if False, also keep FIM markers.

        Returns:
            A boolean [B, L] mask of the selected positions.

        Raises:
            ValueError: If region is not "all", "idr", or "non_idr".
        """
        cutoff = self.n_residues if drop_markers else self.n_residues + self.n_fim
        keep = ids < cutoff
        if region == "all":
            return keep
        if region not in ("idr", "non_idr"):
            raise ValueError(f"region must be 'all', 'idr', or 'non_idr', got {region!r}")
        middle = ids == self.fim_middle_id
        has_mid = middle.any(dim=1)
        mid_pos = torch.where(  # index of the '2' per row; sentinel L (no IDR boundary) if absent
            has_mid,
            middle.int().argmax(dim=1),
            torch.full((ids.size(0),), ids.size(1), device=ids.device, dtype=torch.long),
        )
        after = torch.arange(ids.size(1), device=ids.device)[None, :] > mid_pos[:, None]
        return keep & (after if region == "idr" else (~after & has_mid[:, None]))
