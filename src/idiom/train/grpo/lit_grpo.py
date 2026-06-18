"""GRPO post-training LightningModule.

Each step: expand every prompt into ``group_size`` completions (generated with the KV-cached
sampler), reward each decoded IDR, compute group-normalized advantages, then the DAPO GRPO
loss against a frozen reference. Prompts in a batch are assumed equal length (the typical GRPO
setup — a prompt repeated, or one compartment's flank prompt); length-bucket if mixing.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import asdict

import lightning as L
import torch

from idiom.data.tokenizer import Tokenizer
from idiom.model.config import ModelConfig
from idiom.model.sampling import generate
from idiom.model.transformer import IDiomTransformer
from idiom.train.grpo.core import grpo_loss, group_advantages, sequence_kl, sequence_logprobs
from idiom.train.grpo.rewards import sequence_entropy


class LitGRPO(L.LightningModule):
    def __init__(
        self,
        cfg: ModelConfig,
        reward_fn: Callable[[str], float],
        *,
        group_size: int = 8,
        max_new_tokens: int = 256,
        lr: float = 5e-6,
        beta_kl: float = 0.02,
        eps_clip: float = 0.2,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        normalize_advantage: bool = True,
        log_samples_every: int = 25,
        n_log_samples: int = 3,
        reward_components: Callable[[str], dict[str, float]] | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        # Architecture travels with the checkpoint (read back by model/io.load_pretrained).
        # Only the config — reward_fn/tokenizer are not serializable hyperparameters.
        self.save_hyperparameters({"model_cfg": asdict(cfg)})
        self.model = IDiomTransformer(cfg)
        # Frozen reference = the initial policy; the KL penalty keeps the policy near it.
        self.reference = copy.deepcopy(self.model).eval()
        self.reference.requires_grad_(False)

        self.reward_fn = reward_fn
        # Optional per-term breakdown ({"raw", "length", "entropy", "total"}); when set it is the
        # source of the scalar reward (so reward_fn is not also called) and drives per-term logging.
        self.reward_components = reward_components
        self.tok = tokenizer or Tokenizer()
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.lr = lr
        self.beta_kl = beta_kl
        self.eps_clip = eps_clip
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.normalize_advantage = normalize_advantage
        self.log_samples_every = log_samples_every
        self.n_log_samples = n_log_samples

    @classmethod
    def init_from_checkpoint(cls, ckpt_path, reward_fn, **kwargs) -> "LitGRPO":
        """Warm-start GRPO from a pretrained ckpt; arch read from it (self-describing)."""
        from idiom.model.io import config_from_checkpoint  # noqa: PLC0415

        lit = cls(config_from_checkpoint(ckpt_path), reward_fn, **kwargs)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        model_state = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
        lit.model.load_state_dict(model_state)
        lit.reference.load_state_dict(model_state)
        return lit

    def _decode_idr(self, completion: torch.Tensor) -> str:
        ids: list[int] = []
        for i in completion.tolist():
            if i in (self.tok.stop_id, self.tok.pad_id):
                break  # completion ends at the first STOP/PAD
            if self.tok.is_residue(i):
                ids.append(i)  # residues only; drop stray FIM markers (1/2/3) the model may emit
        return self.tok.decode(ids)  # clean residue string for the reward fn (e.g. ProtGPS/ESM)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        prompts = batch  # [B, P], equal-length prompts
        rep = prompts.repeat_interleave(self.group_size, dim=0)  # [B*G, P]
        BG, P = rep.shape

        with torch.no_grad():  # rollouts are off-policy data; no grad through generation
            completions = generate(
                self.model, rep, max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                top_k=self.top_k, top_p=self.top_p, tokenizer=self.tok,
            )
        T = completions.size(1)

        start = torch.full((BG, 1), self.tok.start_id, dtype=torch.long, device=rep.device)
        full = torch.cat([start, rep, completions], dim=1)  # [B*G, 1+P+T]

        # completion mask over targets (= full[:, 1:]): the T completion positions, minus pad.
        mask = torch.zeros(BG, P + T, device=rep.device)
        mask[:, P:] = (completions != self.tok.pad_id).float()

        idrs = [self._decode_idr(completions[i]) for i in range(BG)]
        # Per-term breakdown when available (raw/length/entropy/total); else just the scalar reward.
        breakdown = [self.reward_components(idr) for idr in idrs] if self.reward_components else None
        rewards = torch.tensor(
            [b["total"] for b in breakdown] if breakdown else [self.reward_fn(idr) for idr in idrs],
            device=rep.device, dtype=torch.float,
        )
        advantages = group_advantages(rewards, self.group_size, normalize=self.normalize_advantage)

        if self.log_samples_every and self.global_step % self.log_samples_every == 0:
            self._print_samples(idrs, rewards)

        policy_logp = sequence_logprobs(self.model, full)
        with torch.no_grad():
            ref_logp = sequence_logprobs(self.reference, full)
        loss = grpo_loss(
            policy_logp, ref_logp, advantages, mask, beta_kl=self.beta_kl, eps_clip=self.eps_clip
        )

        seq_len = torch.tensor([float(len(idr)) for idr in idrs], device=rep.device)
        seq_ent = torch.tensor([sequence_entropy(idr) for idr in idrs], device=rep.device)
        metrics = {
            "train/loss": loss,
            "train/reward": rewards.mean(),
            "train/reward_std": rewards.std(),
            "train/kl": sequence_kl(policy_logp.detach(), ref_logp, mask),
            "train/seq_len": seq_len.mean(),
            "train/seq_entropy": seq_ent.mean(),
        }
        if breakdown:  # per-term reward means: train/reward_raw, _length, _entropy
            for key in breakdown[0]:
                if key == "total":
                    continue
                vals = torch.tensor([b[key] for b in breakdown], device=rep.device)
                metrics[f"train/reward_{key}"] = vals.mean()
        self.log_dict(metrics, prog_bar=True, on_step=True)
        return loss

    def _print_samples(self, idrs: list[str], rewards: torch.Tensor) -> None:
        """Print a few random example completions + rewards, so a tail of the log shows progress."""
        import random

        n = min(self.n_log_samples, len(idrs))
        if n == 0:
            return
        print("=" * 70, flush=True)
        print(
            f"Step {self.global_step}: example generations "
            f"(reward mean={rewards.mean():.3f} std={rewards.std():.3f})",
            flush=True,
        )
        for j, i in enumerate(random.sample(range(len(idrs)), n), 1):
            seq = idrs[i] or "[empty]"
            print(f"  [{j}] reward={rewards[i].item():.3f}  len={len(idrs[i])}  {seq}", flush=True)
        print("=" * 70, flush=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
