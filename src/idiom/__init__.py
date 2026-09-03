"""IDiom: generative design of intrinsically disordered protein regions.

Exports IDiom and IDiomSAE, the two public wrappers, along with ModelConfig, SparseCoder,
Tokenizer, and the fim module.

Example:
    from idiom import IDiom, IDiomSAE

    model = IDiom.from_pretrained("jxliu2/idiom-300M")
    model.generate_unprompted(n=100)
    model.generate_prompted(protein_seq, idr_start, idr_end, n=100)
    model.embed("proteins.fasta", layers=[18])

    sae = IDiomSAE.from_pretrained("jxliu2/idiomsae-300M-L18-k32")
    sae.encode("proteins.fasta")
"""

from idiom.utils.api import IDiom, IDiomSAE
from idiom.data import fim
from idiom.data.tokenizer import Tokenizer
from idiom.model.config import ModelConfig
from idiom.sae import SparseCoder

__all__ = ["IDiom", "IDiomSAE", "ModelConfig", "SparseCoder", "Tokenizer", "fim"]
