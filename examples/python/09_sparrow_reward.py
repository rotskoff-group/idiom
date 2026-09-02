"""RL toward a biophysical target with an EXTERNAL reward model (sparrow), run in its own env.

Same GRPO loop as 08, but the reward is a separate program instead of an in-process function: a
sparrow (https://github.com/idptools/sparrow) scorer that predicts a biophysical property of each
IDR. It imports nothing from IDiom and runs in its own environment, which uv builds on demand from
the cmd — no install step. The scorer returns a raw value (e.g. radius of gyration in A); the term's
target/width shape it with a quadratic penalty toward the target (the same form as the entropy/length
guardrails), so the policy is pushed toward IDRs near that target.

    uv run python examples/python/09_sparrow_reward.py --property radius_of_gyration --target 25 --width 0.2

The first step builds sparrow (~30s, needs a C compiler and network); every run after is a uv cache
hit. Point UV_CACHE_DIR at scratch if your home directory is small. A GPU is recommended for the
policy; pass --device cpu to force CPU.
"""

from __future__ import annotations

import argparse
import sys

import lightning as L
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from idiom import IDiom
from idiom.train.grpo.data import collate_prompts
from idiom.train.grpo.reward import build_reward_terms
from idiom.train.grpo.reward.external_reward import check
from idiom.train.grpo.train_grpo import build
from idiom.utils.device import resolve_device


def mean_reward(reward_terms, model, device, n, group_size):
    """Generate n IDRs from model and return the mean total reward the objective assigns them."""
    idrs = IDiom(model, tokenizer=None, device=device).generate_unprompted(n=n, temperature=1.0, seed=0)
    totals, _ = reward_terms(idrs, group_size)
    return sum(totals) / len(totals)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init-from", default="jxliu2/idiom-20M",
                   help="base to optimize: a HF repo id, a released dir, or a .ckpt")
    p.add_argument("--property", default="radius_of_gyration",
                   help="sparrow property to target (radius_of_gyration, asphericity, FCR, kappa, ...)")
    p.add_argument("--target", type=float, default=25.0, help="target value in the property's units")
    p.add_argument("--width", type=float, default=0.2, help="tolerance as a fraction of the target")
    p.add_argument("--sparrow-spec", default="sparrow @ git+https://github.com/idptools/sparrow.git",
                   help="uv --with spec for sparrow (pin @<commit> for reproducibility)")
    p.add_argument("--steps", type=int, default=5, help="optimizer steps (small for a demo)")
    p.add_argument("--group-size", type=int, default=4, help="completions sampled per prompt")
    p.add_argument("--max-new-tokens", type=int, default=48, help="tokens per completion (small)")
    p.add_argument("--skip-check", action="store_true", help="skip the scorer pre-flight")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = p.parse_args()

    accelerator = {"cpu": "cpu", "cuda": "gpu"}.get(args.device, "auto")
    gen_device = resolve_device(None if args.device == "auto" else args.device)

    # The scorer command: uv builds sparrow in an isolated env from the spec, then runs our scorer,
    # which speaks the newline-delimited JSON protocol on stdin/stdout (see rewards/external_scorers/).
    cmd = (f"uv run --isolated --no-project --with '{args.sparrow_spec}' "
           f"python rewards/external_scorers/sparrow.py --property {args.property}")

    # Pre-flight: build the scorer once and print raw values + shaped rewards, so a broken command
    # fails here (seconds) rather than after warm-starting the policy. This is the same check the
    # `idiom.train.grpo.reward.external_reward` CLI runs.
    if not args.skip_check:
        print(f"checking sparrow scorer ({args.property}, target={args.target}, width={args.width}) ...")
        if check(cmd, args.target, args.width) != 0:
            print("scorer check failed; fix the command before running GRPO", file=sys.stderr)
            raise SystemExit(1)

    # Reward: sparrow as the external objective, with an entropy guardrail to keep IDRs natural.
    cfg = OmegaConf.create({
        "init_from": args.init_from,
        "prompts": {"mode": "unprompted", "n": 16, "n_per": 1, "fasta": None, "batch_size": 2},
        "grpo": {
            "group_size": args.group_size, "max_new_tokens": args.max_new_tokens, "lr": 1e-5,
            "beta_kl": 0.02, "eps_clip": 0.2, "temperature": 1.0, "top_k": None, "top_p": None,
            "normalize_advantage": True, "log_samples_every": 1, "n_log_samples": 2,
        },
        "reward": {
            "entropy": {"enabled": True, "weight": 0.5, "target_entropy": 3.65, "width": 0.2},
            "external": [{"enabled": True, "weight": 1.0, "name": args.property,
                          "target": args.target, "width": args.width, "cmd": cmd}],
        },
    })

    print(f"warm-starting from {args.init_from} ...")
    lit, ds = build(cfg)
    reward_terms = build_reward_terms(cfg.reward)  # reused for the before/after readout

    print(f"mean reward before RL: {mean_reward(reward_terms, lit.model, gen_device, 16, args.group_size):+.3f}")

    dl = DataLoader(ds, batch_size=cfg.prompts.batch_size, shuffle=True, collate_fn=collate_prompts)
    trainer = L.Trainer(
        max_steps=args.steps, accelerator=accelerator, devices=1, logger=False,
        enable_checkpointing=False, limit_val_batches=0, num_sanity_val_steps=0,
        precision="bf16-mixed" if accelerator == "gpu" else "32-true")
    trainer.fit(lit, dl)

    print(f"mean reward after RL:  {mean_reward(reward_terms, lit.model, gen_device, 16, args.group_size):+.3f}")
    print(f"(reward is a quadratic penalty on sparrow's {args.property} toward {args.target}; entropy keeps it natural)")


if __name__ == "__main__":
    main()
