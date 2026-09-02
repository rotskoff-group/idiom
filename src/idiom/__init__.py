"""IDiom — generative design of intrinsically disordered protein regions.

Public API:

    from idiom import IDiom, IDiomSAE
    model = IDiom.from_pretrained("jxliu2/idiom-300M")
    model.generate_unprompted(n=100)                                 # de novo IDRs
    model.generate_prompted(protein_seq, idr_start, idr_end, n=100)  # IDR in flanking context
    model.embed("proteins.fasta", layers=[18])
    sae = IDiomSAE.from_pretrained("jxliu2/idiomsae-300M-L18-k32")   # host model auto-loaded
    sae.encode("proteins.fasta")
"""

from idiom.api import IDiom, IDiomSAE
from idiom.data import fim
from idiom.data.tokenizer import Tokenizer
from idiom.model.config import ModelConfig
from idiom.sae import SparseCoder

__all__ = ["IDiom", "IDiomSAE", "ModelConfig", "SparseCoder", "Tokenizer", "fim"]
