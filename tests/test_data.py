"""P1 data tests (CPU-only): tokenizer, FIM transforms, and marker-drop alignment."""

import torch

from idiom.data.fim import fim_prompted, fim_unprompted, residue_source_positions
from idiom.data.tokenizer import FIM, RESIDUES, Tokenizer

TOK = Tokenizer()


def test_id_map_layout():
    assert TOK.vocab_size == len(RESIDUES) + len(FIM) + 4 == 27
    # residues occupy the lowest ids; markers next; controls last.
    assert [TOK.encode(c)[0] for c in "AY"] == [0, 19]
    assert [TOK.encode(c)[0] for c in "123"] == [20, 21, 22]
    assert (TOK.pad_id, TOK.start_id, TOK.stop_id, TOK.mask_id) == (23, 24, 25, 26)


def test_encode_decode_roundtrip():
    s = "1MEDS3KVDN2RPQNYLF"  # a FIM-shaped string
    assert TOK.decode(TOK.encode(s)) == s


def test_predicates_and_mask():
    assert TOK.is_residue(0) and TOK.is_residue(19)
    assert not TOK.is_residue(20)
    assert TOK.is_fim(20) and TOK.is_fim(22) and not TOK.is_fim(19)
    ids = torch.tensor(TOK.encode("1AC3D2EF"))  # 1 A C 3 D 2 E F
    assert TOK.residue_mask(ids).tolist() == [False, True, True, False, True, False, True, True]


def test_canonical_and_drop_policy():
    import pytest

    assert TOK.is_canonical("MEDSKVDN")
    # non-canonical residues (X, B, Z, U, O, *, lowercase, ...) -> not canonical -> drop upstream
    for bad in ("MEDSX", "meds", "MEDS*", "MED1S"):
        assert not TOK.is_canonical(bad)
    assert not TOK.is_canonical("")  # empty is not a usable sequence
    # encode hard-fails rather than silently dropping/mapping a non-canonical char
    with pytest.raises(ValueError, match="non-canonical"):
        TOK.encode("MEDSX")


def test_fim_transforms():
    seq, start, end = "MEDSKVDNRPQ", 4, 8  # IDR = seq[4:8] = "KVDN" (half-open)
    assert seq[start:end] == "KVDN"
    assert fim_prompted(seq, start, end) == "1MEDS3RPQ2KVDN"   # 1 prefix 3 suffix 2 idr
    assert fim_unprompted(seq, start, end) == "132KVDN"


def test_fim_legacy_aliases():
    # old vocabulary kept as aliases: fim_idr == fim_prompted, fim_idp == fim_unprompted
    from idiom.data.fim import fim_idp, fim_idr, normalize_mode

    assert fim_idr is fim_prompted and fim_idp is fim_unprompted
    assert normalize_mode("idr") == "prompted" and normalize_mode("denovo") == "unprompted"


def test_marker_drop_alignment_prompted():
    seq, start, end = "MEDSKVDNRPQ", 4, 8
    fim = fim_prompted(seq, start, end)
    # residues = FIM string with the 1/3/2 markers removed
    residues = "".join(c for c in fim if c not in "123")
    pos = residue_source_positions(len(seq), start, end, "prompted")
    # each residue aligns to its source index in full_seq
    assert "".join(seq[p] for p in pos) == residues
    # IDR residues are exactly those with start <= pos < end
    assert [p for p in pos if start <= p < end] == list(range(start, end))


def test_marker_drop_alignment_unprompted():
    seq, start, end = "MEDSKVDNRPQ", 4, 8
    pos = residue_source_positions(len(seq), start, end, "unprompted")
    assert "".join(seq[p] for p in pos) == "KVDN"
