"""IDiom — generative design of intrinsically disordered protein regions.

Public API:

    from idiom import IDiom, IDiomSAE
    model = IDiom.from_pretrained("jxliu2/idiom-medium")
    model.generate_idp(n=100)
    model.generate_idr(protein_seq, idr_start, idr_end, n=100)
    model.embed("proteins.fasta", layers=[8])
    sae = IDiomSAE.from_pretrained("jxliu2/idiom-medium-sae-L8", model=model)
    sae.encode("proteins.fasta")
"""

from idiom.api import IDiom, IDiomSAE
from idiom.data import fim
from idiom.data.tokenizer import Tokenizer
from idiom.model.config import ModelConfig
from idiom.sae import SparseCoder

__all__ = ["IDiom", "IDiomSAE", "ModelConfig", "SparseCoder", "Tokenizer", "fim"]
