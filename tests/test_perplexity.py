import math

import pytest
import torch

from idiom.data.tokenizer import Tokenizer
from idiom.utils.perplexity import perplexity


@pytest.mark.parametrize("prompted_prob", [0.0, 1.0])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_completion_perplexity_excludes_context_and_padding(tmp_path, prompted_prob, batch_size):
    fasta = tmp_path / "records.fasta"
    fasta.write_text(">first_IDR_3-4\nCCAACC\n>second_IDR_4-6\nCCCAAACC\n")
    tok = Tokenizer()

    class ConstantModel(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros(*x.shape, tok.vocab_size)
            logits[..., tok.encode("A")[0]] = 2.0
            logits[..., tok.stop_id] = 2.0
            return logits

    kwargs = dict(device="cpu", num_workers=0, batch_size=batch_size,
                  prompted_prob=prompted_prob)
    scores = perplexity(ConstantModel(), str(fasta), completion_only=True, **kwargs)
    # Five IDR residues plus two STOP tokens; flanks, markers and padding are unscored.
    expected_nll = math.log(2 * math.exp(2) + tok.vocab_size - 2) - 2
    assert scores["n_tokens"] == 7
    assert scores["nll"] == pytest.approx(expected_nll)
    assert scores["perplexity"] == pytest.approx(math.exp(expected_nll))

    full = perplexity(ConstantModel(), str(fasta), **kwargs)
    assert full["n_tokens"] > scores["n_tokens"]
    assert full["nll"] > scores["nll"]
