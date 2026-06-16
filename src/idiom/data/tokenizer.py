"""Fixed-alphabet character tokenizer for IDiom (v2).

No data-derived vocabulary: the alphabet is fixed and documented here, so token ids are
stable across datasets, models, and releases. Tokenization is a per-character lookup done
on the fly in the dataset — there is no precompute / token-h5 step.

Id map (vocab size 27)::

    0..19     amino acids, in the order of ``RESIDUES`` below
    20,21,22  FIM markers '1','2','3'   (1=prefix, 3=suffix, 2=middle/IDR)
    23 PAD    24 START    25 STOP    26 MASK

The residues occupy the *lowest* ids on purpose: ``is_residue(i)`` is just ``i < 20`` and
the non-residue ids (FIM markers + control tokens) are exactly what the activation extractor
drops to keep residue-only positions (see :func:`residue_mask`).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

# Character-level: one residue/marker == one token (no BPE).
# Alphabetical order fixes residue ids 0..19 — must not change once a model is trained on it.
RESIDUES = "ACDEFGHIKLMNPQRSTVWY"
RESIDUE_SET = frozenset(RESIDUES)
# FIM markers in `1{prefix}3{suffix}2{IDR}`: 1=prefix, 3=suffix, 2=middle(IDR). Real tokens,
# but dropped for SAE/extraction.
FIM = "123"
# Control tokens — never inside a string, added structurally (START→input, STOP→target end,
# PAD→batch fill / loss ignore_index, MASK→reserved).
SPECIALS = ("<pad>", "<start>", "<stop>", "<mask>")


class Tokenizer:
    """Character-level tokenizer over the fixed ``RESIDUES + FIM + SPECIALS`` alphabet."""

    def __init__(self) -> None:
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
        """True if ``seq`` is only the 20 canonical amino acids (no markers, no junk).

        The ingestion check for raw incoming sequences: any sequence that fails this is
        **dropped whole** (no UNK token, no substitution) — see the FASTA reader and
        curation. FIM markers are intentionally *not* canonical, since this gates raw
        protein/IDR sequences before any FIM assembly.
        """
        return bool(seq) and all(c in RESIDUE_SET for c in seq)

    # --- encode / decode ---
    def encode(self, s: str) -> list[int]:
        """Map a residue/FIM string to token ids (no control tokens added).

        Hard-fails on any non-tokenizable character so a non-canonical residue can never
        slip through silently — such sequences must be dropped upstream (``is_canonical``).
        """
        try:
            return [self._char2id[c] for c in s]
        except KeyError as e:
            raise ValueError(
                f"non-tokenizable character {e.args[0]!r} in sequence; non-canonical "
                f"sequences must be dropped before encoding (see Tokenizer.is_canonical)"
            ) from None

    def decode(self, ids: Iterable[int]) -> str:
        """Inverse of :meth:`encode`; control tokens are skipped."""
        n_seq = self.n_residues + self.n_fim
        return "".join(self._itos[int(i)] for i in ids if int(i) < n_seq)

    # --- token-class predicates (used by the activation extractor) ---
    def is_residue(self, i: int) -> bool:
        return int(i) < self.n_residues

    def is_fim(self, i: int) -> bool:
        return self.n_residues <= int(i) < self.n_residues + self.n_fim

    def residue_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """Boolean mask, ``True`` at real-residue positions (drops FIM markers + controls).

        This is the residue-only selector shared by SAE training and activation export, so
        what the SAE sees and what users export are the same positions.
        """
        return ids < self.n_residues
