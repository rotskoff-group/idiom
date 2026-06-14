"""IDiom — generative design of intrinsically disordered protein regions.

Public API:

    from idiom import IDiom
    model = IDiom.from_pretrained("jxliu2/idiom-medium")
    model.generate_idp(n=100)
    model.generate_idr(protein_seq, idr_start, idr_end, n=100)
    model.embed("proteins.fasta", layers=[8])
"""

from idiom.api import IDiom
from idiom.data import fim
from idiom.data.tokenizer import Tokenizer
from idiom.model.config import ModelConfig
from idiom.sae import SparseCoder

__all__ = ["IDiom", "ModelConfig", "SparseCoder", "Tokenizer", "fim"]
