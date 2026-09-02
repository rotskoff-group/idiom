"""Generate IDRs, unprompted (de novo) and prompted (conditioned on flanks).

    uv run python examples/python/01_generate.py --model jxliu2/idiom-300M --n 10
"""

import argparse

from idiom import IDiom


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="jxliu2/idiom-300M", help="HF repo id or local directory")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="optional FASTA to write")
    args = p.parse_args()

    model = IDiom.from_pretrained(args.model, device=args.device)

    # 1. Unprompted: de-novo IDRs with no flanking context.
    idrs = model.generate_unprompted(n=args.n, temperature=1.0, seed=0)
    print(f"unprompted: {len(idrs)} IDRs, mean length {sum(map(len, idrs)) / len(idrs):.0f}")
    print("  example:", idrs[0][:80])

    # 2. Unprompted within a target length range (oversamples, then length-filters).
    sized = model.generate_unprompted(n=args.n, length_range=(60, 100), seed=0)
    print(f"length-filtered: {len(sized)} IDRs, all in [60, 100]:",
          all(60 <= len(s) <= 100 for s in sized))

    # 3. Prompted: in-fill an IDR given its flanks. Coords are 0-based, half-open.
    protein = "MEDSKVDNRPQACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRST"
    idr_start, idr_end = 12, 30
    filled = model.generate_prompted(protein, idr_start, idr_end, n=args.n, seed=0)
    print(f"prompted: {len(filled)} IDRs in-filled between "
          f"{protein[:idr_start][-8:]}... and ...{protein[idr_end:][:8]}")
    print("  example:", filled[0][:80])

    if args.out:
        model.generate_unprompted_fasta(args.out, n=args.n, seed=0)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
