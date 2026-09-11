"""GRPO rollouts and optimization against a frozen reference policy."""

from __future__ import annotations

import copy
import random
from collections.abc import Callable
from dataclasses import asdict

import lightning as L
import torch

from idiom.data.tokenizer import Tokenizer
from idiom.model.config import ModelConfig
from idiom.model.io import load_model
from idiom.model.sampling import generate
from idiom.model.transformer import IDiomTransformer
from idiom.train.grpo.core import group_advantages, grpo_loss, sequence_kl, sequence_logprobs
from idiom.train.grpo.reward.builtin import composition_entropy
from idiom.utils.disorder import disorder_totals


class LitGRPO(L.LightningModule):
    """Train a policy with GRPO against a frozen reference.

    Attributes:
        cfg (ModelConfig): The architecture being trained.
        model (IDiomTransformer): The policy being optimized.
        reference (IDiomTransformer): A frozen copy of the initial policy, used by the KL penalty.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        reward_terms: Callable[[list[str], int], tuple[list[float], list[dict[str, float]]]],
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
        track_disorder: bool = True,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        """Initialize the policy, frozen reference, and GRPO settings.

        Args:
            cfg: Transformer architecture configuration.
            reward_terms: The composite reward, mapping (idrs, group_size) to per-idr totals and a
                matching per-term breakdown; see reward.build_reward.
            group_size: Number of completions generated per prompt.
            max_new_tokens: Maximum completion length to generate.
            lr: AdamW learning rate.
            beta_kl: Weight of the KL penalty to the reference policy.
            eps_clip: PPO clipping range around a ratio of 1.
            temperature: Sampling temperature for generation.
            top_k: Top-k sampling cutoff, or None.
            top_p: Nucleus sampling cutoff, or None.
            normalize_advantage: If True, divide advantages by their group's standard deviation.
            log_samples_every: Print example completions every this many steps; 0 disables.
            n_log_samples: Number of example completions to print.
            track_disorder: Log metapredict V3 disorder on CPU for each optimizer step.
            tokenizer: Tokenizer; defaults to Tokenizer().
        """
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters({"model_cfg": asdict(cfg)})
        self.model = IDiomTransformer(cfg)
        self.reference = copy.deepcopy(self.model).eval()
        self.reference.requires_grad_(False)

        self.reward_terms = reward_terms
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
        self.track_disorder = track_disorder
        self._disorder_sequences: list[str] = []

    @classmethod
    def init_from_checkpoint(cls, init_from, reward_terms, **kwargs) -> LitGRPO:
        """Initialize the policy and reference from pretrained weights.

        The architecture is read from the artifact.

        Args:
            init_from: A Lightning .ckpt, a released model directory, or a Hub repo id; any form
                idiom.model.io.load_model accepts.
            reward_terms: The composite reward; see the constructor.
            **kwargs: GRPO arguments forwarded to the constructor.

        Returns:
            A module holding the pretrained weights in both the policy and the reference.
        """
        model, cfg = load_model(init_from, eval_mode=False)
        lit = cls(cfg, reward_terms, **kwargs)
        sd = model.state_dict()
        lit.model.load_state_dict(sd)
        lit.reference.load_state_dict(sd)
        return lit

    def _decode_idr(self, completion: torch.Tensor) -> str:
        """Decode one completion to a residue string, stopping at the first STOP or PAD."""
        ids: list[int] = []
        for i in completion.tolist():
            if i in (self.tok.stop_id, self.tok.pad_id):
                break
            if self.tok.is_residue(i):
                ids.append(i)
        return self.tok.decode(ids)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """Roll out completions and return GRPO loss for equal-length prompts.

        Log reward totals and terms, KL, length, and composition entropy.

        Args:
            batch: Equal-length prompts of shape [B, P].
            batch_idx: Index of the batch within the epoch; unused.

        Returns:
            torch.Tensor: The scalar GRPO loss.
        """
        prompts = batch
        rep = prompts.repeat_interleave(self.group_size, dim=0) # [B*G, P]
        BG, P = rep.shape

        with torch.no_grad():
            completions = generate(
                self.model, rep, max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                top_k=self.top_k, top_p=self.top_p, tokenizer=self.tok,
            )
        T = completions.size(1)

        start = torch.full((BG, 1), self.tok.start_id, dtype=torch.long, device=rep.device)
        full = torch.cat([start, rep, completions], dim=1) # [B*G, 1+P+T]

        # Align the completion mask with full[:, 1:], excluding padding
        mask = torch.zeros(BG, P + T, device=rep.device)
        mask[:, P:] = (completions != self.tok.pad_id).float()

        idrs = [self._decode_idr(completions[i]) for i in range(BG)]
        if self.track_disorder:
            self._disorder_sequences.extend(idrs)
        # Score the whole step in one batch to amortize external scorer calls
        totals, breakdown = self.reward_terms(idrs, self.group_size)
        rewards = torch.tensor(totals, device=rep.device, dtype=torch.float)
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
        seq_ent = torch.tensor([composition_entropy(idr) for idr in idrs], device=rep.device)
        metrics = {
            "trainer/global_step": float(self.global_step),
            "train/loss": loss,
            "train/reward": rewards.mean(),
            "train/reward_std": rewards.std(),
            "train/kl": sequence_kl(policy_logp.detach(), ref_logp, mask),
            "train/seq_len": seq_len.mean(),
            "train/seq_entropy": seq_ent.mean(),
        }
        # Log both raw rewards and weighted contributions to the objective
        if breakdown:
            for key in breakdown[0]:
                if key == "total":
                    continue
                vals = torch.tensor([b[key] for b in breakdown], device=rep.device)
                metrics[f"train/reward_{key}"] = vals.mean()
        self.log_dict(metrics, prog_bar=True, on_step=True)
        return loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Score all accumulated rollouts once, before each optimizer update.

        Average residue scores within each nonempty sequence, then average sequences equally.
        Reduce sums/counts across ranks so unequal sequence counts remain correctly weighted.
        """
        if not self.track_disorder or not self._disorder_sequences:
            return
        totals = disorder_totals(self._disorder_sequences)
        stats = torch.tensor(totals, dtype=torch.float64, device=self.device)
        stats = self.trainer.strategy.reduce(stats, reduce_op="sum")
        score_sum, nonempty, total = stats.unbind()
        self.log_dict({
            "train/metapredict_disorder": score_sum / nonempty.clamp_min(1),
            "train/metapredict_empty_fraction": 1 - nonempty / total.clamp_min(1),
        }, on_step=True, on_epoch=False)
        self._disorder_sequences.clear()

    def _print_samples(self, idrs: list[str], rewards: torch.Tensor) -> None:
        """Print up to n_log_samples randomly chosen completions with their rewards."""
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
        """Return AdamW over policy parameters only."""
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
