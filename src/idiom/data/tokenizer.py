"""Fixed-alphabet character tokenizer.

The alphabet is fixed rather than data-derived, and tokenization is a per-character lookup.

Id map (vocab size 27):

    0..19     amino acids, in the order of RESIDUES
    20,21,22  FIM markers 1, 2, 3   (1=prefix, 2=middle/IDR, 3=suffix)
    23 PAD    24 START    25 STOP    26 MASK
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

# Character-level: one residue/marker == one token (no BPE).
# Alphabetical order fixes residue ids 0..19 — must not change once a model is trained on it.
RESIDUES = "ACDEFGHIKLMNPQRSTVWY"
RESIDUE_SET = frozenset(RESIDUES)
# FIM markers in 1{prefix}3{suffix}2{IDR}: 1=prefix, 3=suffix, 2=middle(IDR). Real tokens,
# but dropped for SAE/extraction.
FIM = "123"
# Control tokens — never inside a string, added structurally (START→input, STOP→target end,
# PAD→batch fill / loss ignore_index, MASK→reserved).
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
        """Build the id maps and expose the token-class ids as attributes."""
        self._itos: list[str] = list(RESIDUES) + list(FIM) + list(SPECIALS)  # id -> token
        self._stoi: dict[str, int] = {c: i for i, c in enumerate(self._itos)}  # token -> id
        # Encodable chars only: specials never appear in a string, so they're not in here.
        self._char2id: dict[str, int] = {c: self._stoi[c] for c in RESIDUES + FIM}

        self.n_residues = len(RESIDUES)
        self.n_fim = len(FIM)
        self.vocab_size = len(self._itos)
        self.pad_id = self._stoi["<pad>"]
        self.start_id = self._stoi["<start>"]
        self.stop_id = self._stoi["<stop>"]
        self.mask_id = self._stoi["<mask>"]
        # FIM marker ids (1=prefix, 2=middle/IDR opener, 3=suffix). The MIDDLE marker splits a
        # FIM string into flanks (residues before it) and the IDR (residues after it).
        self.fim_prefix_id = self._stoi["1"]
        self.fim_middle_id = self._stoi["2"]
        self.fim_suffix_id = self._stoi["3"]

    # --- validation ---
    def is_canonical(self, seq: str) -> bool:
        """Return whether seq contains only the 20 canonical amino acids.

        FIM markers and control characters are not canonical, so a FIM-formatted string fails
        this check.

        Args:
            seq (str): The sequence to check.

        Returns:
            bool: True if seq is non-empty and every character is a canonical amino acid.
        """
        return bool(seq) and all(c in RESIDUE_SET for c in seq)

    # --- encode / decode ---
    def encode(self, s: str) -> list[int]:
        """Map a residue/FIM string to token ids, adding no control tokens.

        Args:
            s (str): A string of residue and/or FIM-marker characters.

        Returns:
            list[int]: The token id for each character, in order.

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
        """Map token ids back to a string, dropping control tokens.

        Args:
            ids (Iterable[int]): Token ids to decode.

        Returns:
            str: The residue/FIM string.
        """
        n_seq = self.n_residues + self.n_fim
        return "".join(self._itos[int(i)] for i in ids if int(i) < n_seq)

    # --- token-class predicates (used by the activation extractor) ---
    def is_residue(self, i: int) -> bool:
        """Return True if token id i is a real amino-acid residue."""
        return int(i) < self.n_residues

    def is_fim(self, i: int) -> bool:
        """Return True if token id i is a FIM marker."""
        return self.n_residues <= int(i) < self.n_residues + self.n_fim

    def residue_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask that is True at real-residue positions.

        FIM markers and control tokens are masked out. Equivalent to region_mask with region="all".

        Args:
            ids (torch.Tensor): Token ids, shape [B, L].

        Returns:
            torch.Tensor: A boolean [B, L] mask, True at residue positions.
        """
        return self.region_mask(ids)

    def region_mask(
        self, ids: torch.Tensor, *, region: str = "all", drop_markers: bool = True
    ) -> torch.Tensor:
        """Return a boolean [B, L] mask of positions selected by token class and region.

        Control tokens (START/STOP/PAD/MASK) are always dropped. region then restricts by position
        relative to the FIM MIDDLE ("2") marker that opens the IDR in "1{prefix}3{suffix}2{IDR}".

        Args:
            ids (torch.Tensor): Token ids, shape [B, L].
            region (str): Which kept positions to select: "all" for every kept position, "idr" for
                only those after the "2" marker, "non_idr" for only those before it. A row with no
                "2" marker contributes nothing to "idr" or "non_idr".
            drop_markers (bool): If True, keep only real residues; if False, also keep FIM markers.

        Returns:
            torch.Tensor: A boolean [B, L] mask of the selected positions.

        Raises:
            ValueError: If region is not "all", "idr", or "non_idr".
        """
        cutoff = self.n_residues if drop_markers else self.n_residues + self.n_fim
        keep = ids < cutoff
        if region == "all":
            return keep
        if region not in ("idr", "non_idr"):
            raise ValueError(f"region must be 'all', 'idr', or 'non_idr', got {region!r}")
        middle = ids == self.fim_middle_id  # the '2' that opens the IDR
        has_mid = middle.any(dim=1)
        mid_pos = torch.where(  # index of the '2' per row; sentinel L (no IDR boundary) if absent
            has_mid,
            middle.int().argmax(dim=1),
            torch.full((ids.size(0),), ids.size(1), device=ids.device, dtype=torch.long),
        )
        after = torch.arange(ids.size(1), device=ids.device)[None, :] > mid_pos[:, None]
        return keep & (after if region == "idr" else (~after & has_mid[:, None]))
