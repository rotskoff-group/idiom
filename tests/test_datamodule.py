"""P1 DataModule test (CPU-only): per-split record FASTAs -> padded batches."""

import torch

from idiom.data.datamodule import RecordDataModule
from idiom.data.tokenizer import Tokenizer

TOK = Tokenizer()

TRAIN = ">A_IDR_3-6\nMEDSKVDNRPQ\n>B_IDR_2-5\nACDEFGHIKL\n>C_IDR_1-4\nWYFGSTNQAA\n"
VAL = ">V_IDR_2-4\nMKLVGQHACD\n"


def test_datamodule_batches(tmp_path):
    tr = tmp_path / "train.fasta"
    va = tmp_path / "val.fasta"
    tr.write_text(TRAIN)
    va.write_text(VAL)

    dm = RecordDataModule(tr, va, tokenizer=TOK, batch_size=2, max_len=64, num_workers=0)
    dm.setup()
    assert len(dm.train_set) == 3 and len(dm.val_set) == 1

    x, y, m = next(iter(dm.train_dataloader()))
    assert x.shape == y.shape == m.shape
    assert x.size(0) == 2 and x.dtype == torch.long and m.dtype == torch.bool
    # every row starts with START; padding uses pad_id.
    assert (x[:, 0] == TOK.start_id).all()
    assert x.max().item() <= TOK.mask_id
