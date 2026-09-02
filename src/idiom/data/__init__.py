"""Data layer: tokenizer, FIM formatting, record I/O, datasets, and the record store.

Modules:
    tokenizer: the fixed-alphabet character tokenizer and its position masks.
    fim: fill-in-the-middle string formatting.
    io: FASTA reading and the Record type.
    record_store: a memory-mapped columnar store built from a record FASTA.
    dataset: the map-style RecordDataset and its collate function.
    datamodule: the Lightning DataModule over per-split FASTAs.
"""
