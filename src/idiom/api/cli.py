"""Command-line IDR generation and FASTA export."""

from __future__ import annotations

import argparse

from idiom.api.idiom import IDiom
from idiom.data.fim import UNPROMPTED


def main(argv: list[str] | None = None) -> None:
    """Run the idiom_generate CLI: generate IDRs and write them to a FASTA.

    Args:
        argv: Argument list; sys.argv[1:] if None.
    """
    p = argparse.ArgumentParser(description="Generate IDRs with IDiom (writes a FASTA).")
    p.add_argument("mode", choices=["unprompted", "prompted"],
                   help="unprompted = de novo (no flanks); prompted = in-filled in flanking context")
    p.add_argument("--model", required=True,
                   help="HF repo id (e.g. jxliu2/idiom-300M), a released dir, or a training .ckpt")
    p.add_argument("--out", required=True, help="output FASTA")
    p.add_argument("--n", type=int, default=1000, help="sequences (unprompted) or per protein (prompted)")
    p.add_argument("--fasta", help="prompted mode: input proteins with _IDR_x-y headers")
    p.add_argument("--return-full", action="store_true",
                   help="prompted mode: splice each IDR back into its flanks and write the whole protein")
    p.add_argument("--max-new-tokens", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--min-len", type=int, default=None, help="oversample until len >= this (inclusive)")
    p.add_argument("--max-len", type=int, default=None, help="oversample until len <= this (inclusive)")
    p.add_argument("--max-oversample", type=int, default=20,
                   help="cap on total draws as a multiple of n when a length range is set")
    p.add_argument("--batch-size", type=int, default=None,
                   help="max sequences per model forward (chunks each draw to bound memory)")
    args = p.parse_args(argv)

    model = IDiom.load(args.model)  # a .ckpt from a training run, a released dir, or a Hub id
    kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
              top_k=args.top_k, top_p=args.top_p, batch_size=args.batch_size)
    if args.seed is not None:
        kw["seed"] = args.seed
    if args.min_len is not None or args.max_len is not None:
        kw["length_range"] = (args.min_len or 1, args.max_len or 10**9)
        kw["max_oversample"] = args.max_oversample

    if args.mode == UNPROMPTED:
        model.generate_unprompted_fasta(args.out, n=args.n, **kw)
    else:
        if not args.fasta:
            p.error("prompted mode requires --fasta")
        model.generate_prompted_fasta(args.fasta, args.out, n=args.n, return_full=args.return_full, **kw)
    print(f"wrote {args.out}")



if __name__ == "__main__":
    main()
