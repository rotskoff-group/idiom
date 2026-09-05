#!/bin/bash

set -euo pipefail

###
# GRPO toward a reward you wrote, running in this interpreter. Four ways to name it below; keep one.
# The last shapes it with a rule of your own -- custom_rewards.py registers the reward and the rule,
# so the term's one `module` brings in both. If it needs its own python or torch, use
# custom_scorer.bash instead.
# Needs: 1 GPU, ~12 h.
###

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

OUT="${IDIOM_OUT:-$REPO/runs}/grpo-my-reward"

# EDIT: a file imported before every term, for shaping that lives somewhere other than the term's
# own module. null unless you need it -- custom_rewards.py registers its one_sided rule itself.
MODULE=null

# The whole objective, written out: nothing is added for you and reward.terms is empty by default.
# This one is deliberately bare -- a single term, no entropy and no length -- to show that a run
# gets exactly the objective it names. Uncomment the two below for a real run: without them the
# target is satisfiable by a low-complexity tract or a degenerate length.
# ENTROPY='{reward: entropy, weight: 1.0, shaping: {type: quadratic, target: 3.65, width: 0.2}}'
# LENGTH='{reward: length,  weight: 1.0, shaping: {type: quadratic, target: 100,  width: 1.0}}'

# EDIT: pick ONE.
MINE="{reward: fraction_charged, module: cookbook/rewards/custom_rewards.py, weight: 1.0, shaping: {type: gaussian, target: 0.25, width: 0.5}}"
# MINE="{reward: \"mypackage.scoring:score_idr\", label: mine, weight: 1.0}"                    # any importable callable
# MINE="{reward: \"mypackage.scoring:score_batch\", label: mine, batched: true, weight: 1.0}"   # f(idrs, batch), scored a step at a time
# MINE="{reward: \"mypackage.scoring:make_scorer\", label: mine, params: {cutoff: 0.3}, weight: 1.0}"  # a factory, called with params
# MINE="{reward: fraction_charged, module: cookbook/rewards/custom_rewards.py, weight: 1.0, shaping: {type: one_sided, target: 0.30, width: 0.5}}"   # shaping from the same file

export WANDB_MODE=offline

idiom_train_grpo \
    seed=0 \
    device=auto \
    init_from=jxliu2/idiom-300M \
    prompts.mode=unprompted \
    prompts.n=1000 \
    prompts.fasta=null \
    prompts.n_per=1000 \
    prompts.batch_size=4 \
    grpo.group_size=8 \
    grpo.max_new_tokens=1020 \
    grpo.lr=5.0e-6 \
    grpo.beta_kl=0.02 \
    grpo.eps_clip=0.2 \
    grpo.temperature=1.0 \
    grpo.top_k=null \
    grpo.top_p=null \
    grpo.normalize_advantage=true \
    grpo.log_samples_every=5 \
    grpo.n_log_samples=3 \
    reward.module=$MODULE \
    reward.terms="[$MINE]" \
    trainer.max_steps=3000 \
    trainer.accelerator=auto \
    trainer.devices=1 \
    trainer.gradient_clip_val=1.0 \
    trainer.accumulate_grad_batches=2 \
    trainer.log_every_n_steps=1 \
    trainer.checkpoint_every=0 \
    out_dir="$OUT" \
    hydra.run.dir="$OUT/hydra"

echo "DONE -> $OUT"
