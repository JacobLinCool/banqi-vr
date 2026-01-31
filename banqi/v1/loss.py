"""
Multi-task loss for BanqiTransformer:
  - self policy: KL(pi_target || pi_pred) implemented as cross-entropy with target probs
  - opponent policy: same
  - value: MSE or Huber
  - belief: per-square KL / cross-entropy

All losses support masking (illegal actions, masked belief squares).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import BanqiTransformer, ModelOutputs


@dataclass
class LossConfig:
    w_self_policy: float = 1.0
    w_value: float = 1.0
    w_belief: float = 0.25
    value_loss: str = "huber"  # "mse" or "huber"
    huber_delta: float = 1.0
    entropy_coef: float = 0.0  # optional regularizer on self policy entropy


def _renorm_probs(
    probs: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    probs = torch.where(mask, probs, torch.zeros_like(probs))
    z = probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return probs / z


def policy_ce_from_probs(
    logits: torch.Tensor,  # [B,A]
    target_probs: torch.Tensor,  # [B,A] (not necessarily normalized, will be renormed on legal actions)
    legal_mask: torch.Tensor,  # [B,A] bool
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Returns (loss, metrics)
    loss = - sum_a pi_target(a) * log pi_pred(a)
    where pi_pred is softmax over legal actions.
    """
    target = _renorm_probs(target_probs, legal_mask)

    logp = BanqiTransformer.masked_log_softmax(logits, legal_mask, dim=-1)  # [B,A]
    loss_per = -(target * logp).sum(dim=-1)  # [B]
    loss = loss_per.mean()

    with torch.no_grad():
        # entropy of predicted policy
        p = torch.exp(logp)
        entropy = -(p * logp).sum(dim=-1).mean()

        # KL(target||pred) = CE(target,pred) - H(target)
        h_t = -(target * torch.log(target.clamp_min(1e-8))).sum(dim=-1).mean()
        kl = loss - h_t

    return loss, {"entropy": entropy, "kl": kl}


def value_loss(
    pred: torch.Tensor,  # [B]
    target: torch.Tensor,  # [B]
    kind: str = "huber",
    delta: float = 1.0,
) -> torch.Tensor:
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "huber":
        return F.smooth_l1_loss(pred, target, beta=delta)
    raise ValueError(f"Unknown value loss kind: {kind}")


def belief_ce_from_probs(
    logits: torch.Tensor,  # [B,32,C]
    target_probs: torch.Tensor,  # [B,32,C]
    square_mask: Optional[torch.Tensor] = None,  # [B,32] bool; if None, all squares
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Cross-entropy with soft targets.
    """
    B, S, C = logits.shape
    assert target_probs.shape == (B, S, C)

    logp = F.log_softmax(logits, dim=-1)
    target = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    ce = -(target * logp).sum(dim=-1)  # [B,32]

    if square_mask is not None:
        m = square_mask.to(dtype=torch.float32)
        denom = m.sum().clamp_min(1.0)
        loss = (ce * m).sum() / denom
        with torch.no_grad():
            avg_ce = (ce * m).sum() / denom
    else:
        loss = ce.mean()
        with torch.no_grad():
            avg_ce = ce.mean()

    return loss, {"belief_ce": avg_ce}


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        out: ModelOutputs,
        target_self_policy: torch.Tensor,  # [B,A]
        target_value: torch.Tensor,  # [B]
        target_belief: torch.Tensor,  # [B,32,C]
        legal_action_mask: torch.Tensor,  # [B,A] bool
        belief_loss_mask: Optional[torch.Tensor] = None,  # [B,32] bool
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        metrics: Dict[str, float] = {}

        # self policy
        l_pi, m_pi = policy_ce_from_probs(
            out.policy_self_logits, target_self_policy, legal_action_mask
        )
        metrics["self_policy_ce"] = float(l_pi.detach().cpu())
        metrics["self_policy_entropy"] = float(m_pi["entropy"].detach().cpu())
        metrics["self_policy_kl"] = float(m_pi["kl"].detach().cpu())

        total = cfg.w_self_policy * l_pi

        # entropy regularization (encourage exploration if >0)
        if cfg.entropy_coef != 0.0:
            total = total - cfg.entropy_coef * m_pi["entropy"]

        # value
        l_v = value_loss(
            out.value, target_value, kind=cfg.value_loss, delta=cfg.huber_delta
        )
        metrics["value_loss"] = float(l_v.detach().cpu())
        total = total + cfg.w_value * l_v

        # belief
        if cfg.w_belief != 0.0:
            l_b, m_b = belief_ce_from_probs(
                out.belief_logits, target_belief, square_mask=belief_loss_mask
            )
            metrics.update({k: float(v.detach().cpu()) for k, v in m_b.items()})
            total = total + cfg.w_belief * l_b

        metrics["total_loss"] = float(total.detach().cpu())
        return total, metrics
