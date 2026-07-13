"""SAE-feature-guided ProtGPS reward for GRPO — ``f(idr) -> ProtGPS(target) + λ·feature-match``.

Motivation: ProtGPS(target) alone has a *composition shortcut* — RL can max the classifier with a
composition unlike the real proteins, so the generated sequences do NOT encode the compartment's
natural SAE feature code (see fig_rl: pml/PSD/p-body converge at ~0 despite P≈0.99). This reward adds
an auxiliary term that pays the model for *encoding the target's specific (degree-1) features*:

    R(idr) = ProtGPS(target)  +  λ · (fraction of the target's specific features that fire)

"f fires" = f is in the SAE top-k at any IDR residue (the enrichment prevalence definition). The
feature score is computed by encoding the completion through the FROZEN 24L base + L18 SAE (the same
fixed lens used for enrichment), so a reward gain requires reproducing the real code, not just the
classifier label. Keep the grpo.yaml entropy term on — it is the naturalness guardrail.

Config (Hydra selects reward.name; hyperparameters via env, since the reward contract is f(idr)->float):
    reward.module=rewards/sae_feature_reward.py  reward.name=protgps_feat_<compartment>
    IDIOM_SAEREWARD_LAMBDA    feature-term weight λ (default 0.5)
    IDIOM_SAEREWARD_SAE       L18 SAE dir (default the 24L-recompute enrichment SAE)
    IDIOM_SAEREWARD_FEATURES  feature_sets_enrichment.json (per-comp specific/top200 feature ids)
    IDIOM_SAEREWARD_CASE      which feature set to reward: specific (default) | top200
    IDIOM_SAEREWARD_DEVICE    torch device for the base+SAE lens (default cuda if available)
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import torch

import numpy as np

from idiom.train.grpo.rewards import register_group_reward, register_reward

# reuse the ProtGPS scorer + its 12-class order (sibling module; add dir to path for path-loaded import)
sys.path.insert(0, os.path.dirname(__file__))
from protgps_reward import COMPARTMENTS, protgps_scores  # noqa: E402

_SAE_DIR = os.environ.get(
    "IDIOM_SAEREWARD_SAE",
    "/data2/scratch/group_scratch/idr_plm/2026-06-26_24L_recompute/03_sae/x16_k32_idp/L18")
_FEATURES = os.environ.get(
    "IDIOM_SAEREWARD_FEATURES",
    "/data2/scratch/group_scratch/idr_plm/2026-07-02_steer_l18/feature_sets_enrichment.json")
_CASE = os.environ.get("IDIOM_SAEREWARD_CASE", "specific")
_LAMBDA = float(os.environ.get("IDIOM_SAEREWARD_LAMBDA", "0.5"))
# feature-set compartments use the dataset/enrichment spelling; ProtGPS class uses the "pml-bdoy" typo
_PC_ALIAS = {"pml_body": "pml-bdoy"}
_FEAT_COMPS = ["stress_granule", "p-body", "nuclear_speckle", "nucleolus",
               "chromosome", "pml_body", "post_synaptic_density", "nuclear_pore_complex"]


def _saedev() -> str:
    d = os.environ.get("IDIOM_SAEREWARD_DEVICE")
    if d:
        return d
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _sae():
    """Frozen 24L base + L18 SAE lens (loaded once, shares the policy's GPU like ProtGPS)."""
    from idiom import IDiomSAE
    return IDiomSAE.from_pretrained(_SAE_DIR, device=_saedev())


@lru_cache(maxsize=1)
def _featuresets():
    return json.loads(Path(_FEATURES).read_text())[_CASE]


@lru_cache(maxsize=16)
def _comp_ids(comp: str):
    """Feature-id LongTensor for a compartment's specific set, on the SAE device."""
    return torch.tensor(_featuresets()[comp], device=_sae().device, dtype=torch.long)


@torch.no_grad()
def _feature_match(idr: str, comp: str) -> float:
    """Fraction of the compartment's specific features that FIRE (top-k at any IDR residue)."""
    from idiom.data.fim import fim_idp
    from idiom.model.activations import extract_activations
    sae = _sae()
    s = fim_idp(idr, 0, len(idr))                                   # "132" + idr (de-novo IDP)
    tokens = torch.tensor([[sae.tok.start_id, *sae.tok.encode(s)]], device=sae.device)
    acts = extract_activations(sae.model, tokens, [sae.layer], tokenizer=sae.tok,
                               drop_markers=True, region=sae.region)[sae.layer]
    feats = sae.sae.encode_dense(acts.values.to(sae.device))       # [n_idr_res, num_latents]
    ids = _comp_ids(comp)
    fired = (feats[:, ids] > 0).any(dim=0).float()                 # [n_ids]: feature in top-k anywhere
    return float(fired.mean())


def _feat_reward(comp: str):
    idx = COMPARTMENTS.index(_PC_ALIAS.get(comp, comp))

    def reward(idr: str) -> float:
        if not idr:
            return 0.0
        p = float(protgps_scores(idr)[idx])
        return p + _LAMBDA * _feature_match(idr, comp)

    return reward


def _sae_only_reward(comp: str):
    """ProtGPS-FREE reward: just the feature-match (fraction of the target's specific features that
    fire). The classifier is never in the loop -- tests whether optimizing the interpretable code
    alone yields on-code (and, we then check, actually localizing) sequences."""

    def reward(idr: str) -> float:
        if not idr:
            return 0.0
        return _feature_match(idr, comp)

    return reward


for _c in _FEAT_COMPS:
    register_reward(f"protgps_feat_{_c}")(_feat_reward(_c))     # ProtGPS + λ·feature-match
    register_reward(f"sae_only_{_c}")(_sae_only_reward(_c))     # feature-match only (no ProtGPS)


# --- GROUP coverage reward: reward POPULATION coverage of the code, not per-sequence cramming ---
@torch.no_grad()
def _fired_matrix(idrs, ids):
    """Boolean [n_idrs, n_ids]: which of the target features fire (top-k at any IDR residue) in each
    completion. One batched forward through the frozen 24L base + L18 SAE for the whole group."""
    from torch.nn.utils.rnn import pad_sequence

    from idiom.data.fim import fim_idp
    from idiom.model.activations import extract_activations
    sae = _sae()
    seqs = [fim_idp(s, 0, len(s)) for s in idrs]
    toks = [torch.tensor([sae.tok.start_id, *sae.tok.encode(s)]) for s in seqs]
    tokens = pad_sequence(toks, batch_first=True, padding_value=sae.tok.pad_id).to(sae.device)
    acts = extract_activations(sae.model, tokens, [sae.layer], tokenizer=sae.tok,
                               drop_markers=True, region=sae.region)[sae.layer]
    feats = sae.sae.encode_dense(acts.values.to(sae.device))       # [N_res, num_latents]
    fv = (feats[:, ids] > 0).cpu().numpy()                         # [N_res, n_ids]
    si = acts.seq_idx.cpu().numpy()
    out = np.zeros((len(idrs), len(ids)), dtype=bool)
    np.logical_or.at(out, si, fv)                                  # OR residues into their seq row
    return out


def _coverage_scores(idrs, group_size, comp):
    """Within-group frequency-discounted coverage: reward_j = Σ_{f fired by j} 1/(#group firing f) /
    n_ids. A feature all G seqs fire is worth 1/G each; a feature only j fires is worth 1 -> the group
    spreads to cover the whole signature and no single sequence is rewarded for cramming."""
    ids = _comp_ids(comp)
    F = _fired_matrix(idrs, ids)                                   # [n, n_ids] bool
    n_ids = F.shape[1]
    scores = np.zeros(len(idrs))
    for g in range(0, len(idrs), group_size):
        blk = F[g:g + group_size]                                 # [<=G, n_ids]
        cnt = blk.sum(0).astype(float)                            # times each feature fires in group
        inv = np.zeros_like(cnt)
        np.divide(1.0, cnt, out=inv, where=cnt > 0)               # 1/cnt where fired, else 0 (no warn)
        for j in range(blk.shape[0]):
            scores[g + j] = float((blk[j] * inv).sum() / n_ids)
    return scores.tolist()


def _coverage_group(comp: str):
    def fn(idrs, group_size):
        return _coverage_scores(idrs, group_size, comp) if idrs else []
    return fn


for _c in _FEAT_COMPS:
    register_group_reward(f"sae_coverage_{_c}")(_coverage_group(_c))
