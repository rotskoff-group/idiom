"""RL post-training (GRPO): optimize a released model toward a reward over the IDRs it generates.

Runs a real (tiny) GRPO loop end to end. It warm-starts from a released model (init_from also takes
a local .ckpt or released dir — e.g. the checkpoint 07_sft.py saves), then optimizes a weighted-sum
reward: the entropy/length guardrails plus a custom "bring your own" reward registered right here.
Each step samples group_size completions per prompt, scores them, and takes a GRPO step; the reward
should climb over a handful of steps. Defaults run anywhere in a couple of minutes.

    idiom_grpo init_from=jxliu2/idiom-300M reward.rl_sae.enabled=true reward.rl_sae.signature=nucleolus

    uv run python examples/08_grpo.py --steps 5

A GPU is recommended (each step generates many completions); pass --device cpu to force CPU.
"""

from __future__ import annotations

import argparse

import lightning as L
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from idiom import IDiom
from idiom.train.grpo.data import collate_prompts
from idiom.train.grpo.reward import build_reward_terms, register_reward
from idiom.train.grpo.train_grpo import build
from idiom.utils.device import resolve_device


@register_reward("aromatic_fraction")
def aromatic_fraction(idr: str) -> float:
    """Fraction of aromatic residues (F/W/Y) — 'stickers' that drive condensate formation."""
    return sum(idr.count(a) for a in "FWY") / len(idr) if idr else 0.0


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
    p.add_argument("--steps", type=int, default=5, help="optimizer steps (small for a demo)")
    p.add_argument("--group-size", type=int, default=4, help="completions sampled per prompt")
    p.add_argument("--max-new-tokens", type=int, default=48, help="tokens per completion (small)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = p.parse_args()

    accelerator = {"cpu": "cpu", "cuda": "gpu"}.get(args.device, "auto")
    gen_device = resolve_device(None if args.device == "auto" else args.device)

    # A weighted-sum reward: entropy + length guardrails, plus our custom aromatic term (already
    # registered above, so the external entry just names it — no module import needed).
    cfg = OmegaConf.create({
        "init_from": args.init_from,
        "prompts": {"mode": "unprompted", "n": 16, "n_per": 1, "fasta": None, "batch_size": 2},
        "grpo": {
            "group_size": args.group_size, "max_new_tokens": args.max_new_tokens, "lr": 1e-5,
            "beta_kl": 0.02, "eps_clip": 0.2, "temperature": 1.0, "top_k": None, "top_p": None,
            "normalize_advantage": True, "log_samples_every": 1, "n_log_samples": 2,
        },
        "reward": {
            "entropy": {"enabled": True, "weight": 1.0, "target_entropy": 3.65, "width": 0.2},
            "length": {"enabled": True, "weight": 1.0, "target_length": 60, "width": 1.0},
            "external": [{"enabled": True, "weight": 3.0, "name": "aromatic_fraction"}],
        },
    })

    print(f"warm-starting from {args.init_from} ...")
    lit, ds = build(cfg)
    reward_terms = build_reward_terms(cfg.reward)  # same objective, for the before/after readout

    print(f"mean reward before RL: {mean_reward(reward_terms, lit.model, gen_device, 16, args.group_size):+.3f}")

    dl = DataLoader(ds, batch_size=cfg.prompts.batch_size, shuffle=True, collate_fn=collate_prompts)
    trainer = L.Trainer(
        max_steps=args.steps, accelerator=accelerator, devices=1, logger=False,
        enable_checkpointing=False, limit_val_batches=0, num_sanity_val_steps=0,
        precision="bf16-mixed" if accelerator == "gpu" else "32-true")
    trainer.fit(lit, dl)

    print(f"mean reward after RL:  {mean_reward(reward_terms, lit.model, gen_device, 16, args.group_size):+.3f}")
    print("(reward toward aromatic residues; entropy/length keep it natural-looking)")


if __name__ == "__main__":
    main()
