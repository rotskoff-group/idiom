"""idiom.data — tokenizer, FIM transform, datasets, and curation.

v2: char tokenization and FIM-formatting happen **on the fly** in the dataset; the
precompute/token-h5 pipeline is removed. Curation writes raw records
``(accession, full_seq, idr_start, idr_end)`` which the dataset turns into
``1{prefix}3{suffix}2{IDR}`` / ``132{IDR}`` strings at load time.

Status: P0 stub. See REFACTOR_PLAN.md (P1). Migration source: legacy
``idiom.scripts.data`` and ``idiom.nn.transformer.{dataset,generators}``.
"""
