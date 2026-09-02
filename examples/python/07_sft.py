"""Supervised fine-tuning (SFT): warm-start a released model and specialize it on one IDR set.

Runs a real (tiny) SFT loop end to end. It warm-starts from a released model straight off the Hub
(init_from also accepts a local .ckpt or released dir), fine-tunes on a curated set with the
completion-only loss, and generates before/after so you can see it moved. Defaults are sized to run
anywhere in a minute or two; a real run uses more steps, a bigger base, and the CLI:

    idiom_train --config-name sft init_from=jxliu2/idiom-300M \
      data.train_fasta=examples/example_data/protgps/nucleolus.fasta

    uv run python examples/python/07_sft.py --steps 30

A GPU is recommended; pass --device cpu (and keep --init-from small) to force CPU.
"""

from __future__ import annotations

import argparse

import lightning as L

from idiom import IDiom
from idiom.data.datamodule import RecordDataModule
from idiom.train.lit_autoregressive import LitAutoregressive
from idiom.utils.device import resolve_device


def sample(lit, device, n=3):
    """Generate a few unprompted IDRs from the current policy, for a before/after peek."""
    model = IDiom(lit.model, tokenizer=None, device=device)
    idrs = model.generate_unprompted(n=n, temperature=1.0, seed=0)
    return [s[:60] for s in idrs]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init-from", default="jxliu2/idiom-20M",
                   help="base to fine-tune: a HF repo id, a released dir, or a .ckpt")
    p.add_argument("--train-fasta", default="examples/example_data/protgps/nucleolus.fasta",
                   help="the (usually small, curated) SFT set")
    p.add_argument("--steps", type=int, default=30, help="optimizer steps (small for a demo)")
    p.add_argument("--lr", type=float, default=1e-5, help="gentle, to not wash out pretraining")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--out", default=None, help="optional .ckpt to save the fine-tuned model to")
    args = p.parse_args()

    accelerator = {"cpu": "cpu", "cuda": "gpu"}.get(args.device, "auto")
    gen_device = resolve_device(None if args.device == "auto" else args.device)

    # Warm start: architecture and weights come from the base; SFT never re-declares the model.
    print(f"warm-starting from {args.init_from} ...")
    lit = LitAutoregressive.init_from_checkpoint(
        args.init_from, lr=args.lr, warmup_steps=max(1, args.steps // 10), max_steps=args.steps)

    print("base model, before SFT:")
    for s in sample(lit, gen_device):
        print("  ", s)

    # SFT data, loss on the IDR completion only. These sets are whole-sequence IDRs (no flanks), so
    # train the de-novo/unprompted form (prompted_prob=0.0) — for a full IDR the prompted "1{}3{}2{IDR}"
    # and unprompted "132{IDR}" strings are identical anyway. Raise prompted_prob only when your
    # records carry real flanking context you want to condition on.
    dm = RecordDataModule(
        args.train_fasta, None, max_len=lit.cfg.max_seq_len,
        prompted_prob=0.0, completion_only=True, batch_size=args.batch_size, num_workers=0, seed=0)

    trainer = L.Trainer(
        max_steps=args.steps, accelerator=accelerator, devices=1, logger=False,
        enable_checkpointing=False, limit_val_batches=0, num_sanity_val_steps=0,
        gradient_clip_val=1.0, precision="bf16-mixed" if accelerator == "gpu" else "32-true")
    trainer.fit(lit, datamodule=dm)

    print("after SFT:")
    for s in sample(lit, gen_device):
        print("  ", s)

    if args.out:
        trainer.save_checkpoint(args.out)
        print(f"saved fine-tuned checkpoint to {args.out} (feed it to 08_grpo.py or idiom_grpo)")


if __name__ == "__main__":
    main()
