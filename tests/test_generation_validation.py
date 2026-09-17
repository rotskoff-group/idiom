"""Public generation rejects invalid inputs before any model forward or download."""

import pytest

from idiom import IDiom, IDiomSAE, ModelConfig
from idiom.model import IDiomTransformer
from idiom.sae import SparseCoder


@pytest.fixture
def model():
    host = IDiom(IDiomTransformer(ModelConfig(n_layers=1, d_model=16, n_heads=2, max_seq_len=32)))

    def fail_forward(*args, **kwargs):
        pytest.fail("invalid or empty request executed the model")

    host.model.forward = fail_forward
    return host


@pytest.mark.parametrize(
    "kw",
    [
        {"n": -1},
        {"n": 1.5},
        {"n": True},
        {"max_new_tokens": 0},
        {"max_new_tokens": 1.5},
        {"batch_size": 0},
        {"batch_size": -2},
        {"batch_size": 1.5},
        {"max_oversample": 0},
        {"max_oversample": 1.5},
        {"temperature": -1},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"top_k": -1},
        {"top_k": 0},
        {"top_k": 1.5},
        {"top_p": 0},
        {"top_p": 1.1},
        {"top_p": float("nan")},
        {"length_range": (5, 2)},
        {"length_range": (0, 2)},
        {"length_range": (1.5, 2)},
        {"length_range": (1,)},
        {"length_range": (10, 20), "max_new_tokens": 5},
        {"seed": -1},
        {"seed": 2**64},
    ],
)
def test_invalid_options_rejected_by_generation_and_steering(model, kw):
    sae = IDiomSAE(SparseCoder(16, num_latents=32, k=4), model, layer=0)
    for call in (model.generate_unprompted, lambda **opts: sae.steer_generate(0, 0.5, **opts)):
        with pytest.raises(ValueError):
            call(**kw)


@pytest.mark.parametrize("start,end", [(-1, 2), (2, 1), (2, 2), (0, 7), (0.5, 2), (False, 2)])
def test_invalid_spans(model, start, end):
    with pytest.raises(ValueError):
        model.generate_prompted("ACDEFG", start, end)


@pytest.mark.parametrize("seq", ["", "ACXEFG", "AC1EFG", "acdefg"])
def test_noncanonical_sequence_rejected_even_inside_replaced_region(model, seq):
    with pytest.raises(ValueError, match="canonical"):
        model.generate_prompted(seq, 0, len(seq))


def test_zero_count_is_empty_and_full_sequence_span_is_valid(model):
    assert model.generate_unprompted(n=0) == []
    assert model.generate_prompted("ACDEFG", 0, 6, n=0) == []
    sae = IDiomSAE(SparseCoder(16, num_latents=32, k=4), model, layer=0)
    assert sae.steer_generate(0, 0.5, n=0) == []


@pytest.mark.parametrize("args", [["--min-len", "0"], ["--top-p", "nan"], ["--n", "-1"]])
def test_cli_validates_before_loading(monkeypatch, args):
    from idiom.api.cli import main

    monkeypatch.setattr(IDiom, "load", lambda *a, **kw: pytest.fail("loaded model"))
    with pytest.raises(SystemExit) as exc:
        main(["unprompted", "--model", "unused", "--out", "unused", *args])
    assert exc.value.code == 2


def test_empty_fasta_still_validates_options(model, tmp_path):
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("")
    with pytest.raises(ValueError, match="temperature"):
        model.generate_prompted_fasta(fasta, tmp_path / "out.fasta", temperature=-1)
