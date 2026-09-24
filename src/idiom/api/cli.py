"""Command-line IDR generation and FASTA export."""

from __future__ import annotations

import argparse

from idiom.api.idiom import IDiom
from idiom.data.fim import UNPROMPTED
from idiom.utils.validation import validate_generation


def main(argv: list[str] | None = None) -> None:
    """Run the idiom_generate CLI: generate IDRs and write them to a FASTA.

    Args:
        argv: Argument list; sys.argv[1:] if None.
    """
    p = argparse.ArgumentParser(description="Generate IDRs with IDiom (writes a FASTA).")
    p.add_argument(
        "mode",
        choices=["unprompted", "prompted"],
        help="unprompted = de novo (no flanks); prompted = in-filled in flanking context",
    )
    p.add_argument(
        "--model",
        required=True,
        help="HF repo id (e.g. jxliu2/idiom-300M), a released dir, or a training .ckpt",
    )
    p.add_argument("--out", required=True, help="output FASTA")
    p.add_argument("--n", type=int, default=1000, help="sequences (unprompted) or per protein (prompted)")
    p.add_argument("--fasta", help="prompted mode: input proteins with _IDR_x-y headers")
    p.add_argument(
        "--return-full",
        action="store_true",
        help="prompted mode: splice each IDR back into its flanks and write the whole protein",
    )
    p.add_argument("--max-new-tokens", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--min-len", type=int, default=None, help="oversample until len >= this (inclusive)")
    p.add_argument("--max-len", type=int, default=None, help="oversample until len <= this (inclusive)")
    p.add_argument(
        "--max-oversample",
        type=int,
        default=20,
        help="cap on total draws as a multiple of n when a length range is set",
    )
    p.add_argument("--batch-size", type=int, default=8, help="max sequences per model forward (default: 8)")
    args = p.parse_args(argv)

    kw = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        batch_size=args.batch_size,
        max_oversample=args.max_oversample,
    )
    if args.seed is not None:
        kw["seed"] = args.seed
    if args.min_len is not None or args.max_len is not None:
        kw["length_range"] = (
            1 if args.min_len is None else args.min_len,
            10**9 if args.max_len is None else args.max_len,
        )
        kw["max_oversample"] = args.max_oversample

    try:
        validate_generation(args.n, **kw)
    except ValueError as exc:
        p.error(str(exc))
    if args.mode != UNPROMPTED and not args.fasta:
        p.error("prompted mode requires --fasta")
    model = IDiom.load(args.model)
    if args.mode == UNPROMPTED:
        model.generate_unprompted_fasta(args.out, n=args.n, **kw)
    else:
        model.generate_prompted_fasta(args.fasta, args.out, n=args.n, return_full=args.return_full, **kw)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
