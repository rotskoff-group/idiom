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

# For logging entropy
from idiom.train.grpo.reward.builtin import composition_entropy


class LitGRPO(L.LightningModule):
    """GRPO post-training module performing rollout, reward, advantages, and the GRPO step.

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
        tokenizer: Tokenizer | None = None,
    ) -> None:
        """Build the policy and its frozen reference, and record the GRPO settings.

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
            tokenizer: Tokenizer; defaults to Tokenizer().
        """
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters({"model_cfg": asdict(cfg)})
        self.model = IDiomTransformer(cfg)
        # Frozen reference = the initial policy
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

    @classmethod
    def init_from_checkpoint(cls, init_from, reward_terms, **kwargs) -> LitGRPO:
        """Build a module whose policy and reference both hold a pretrained model's weights.

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
                break  # completion ends at the first STOP/PAD
            if self.tok.is_residue(i):
                ids.append(i)  # residues only; drop stray FIM markers (1/2/3) the model may emit
        return self.tok.decode(ids)  # clean residue string for the reward fn (e.g. an external scorer)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """Roll out completions and return GRPO loss for equal-length prompts.

        Log reward totals and terms, KL, length, and composition entropy.

        Args:
            batch: Equal-length prompts of shape [B, P].
            batch_idx: Index of the batch within the epoch; unused.

        Returns:
            torch.Tensor: The scalar GRPO loss.
        """
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
        # The composite reward scores the whole step at once, so an external term makes one round
        # trip per step, and returns the per-term breakdown that is logged below.
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
            "trainer/global_step": float(self.global_step),  # real step -> W&B x-axis (see run())
            "train/loss": loss,
            "train/reward": rewards.mean(),
            "train/reward_std": rewards.std(),
            "train/kl": sequence_kl(policy_logp.detach(), ref_logp, mask),
            "train/seq_len": seq_len.mean(),
            "train/seq_entropy": seq_ent.mean(),
        }
        # Per-term means, two per term: train/reward_length is the contribution to the objective,
        # train/reward_length_raw the raw reward (98 residues, 3.6 bits) in its own units.
        if breakdown:
            for key in breakdown[0]:
                if key == "total":
                    continue
                vals = torch.tensor([b[key] for b in breakdown], device=rep.device)
                metrics[f"train/reward_{key}"] = vals.mean()
        self.log_dict(metrics, prog_bar=True, on_step=True)
        return loss

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
