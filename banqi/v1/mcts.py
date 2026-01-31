"""banqi.v1.mcts

Torch-backed wrapper around evaluator-agnostic MCTS core.

For async actor/inference setups, import `banqi.v1.mcts_core` instead to avoid torch import.

The evaluator-agnostic core (mcts_core) uses an information-set hash so statistics can be
shared across determinizations (root-sampling ISMCTS), which is typically both stronger and
more sample-efficient than running one independent tree per determinization.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .env import BanqiEnv, Observation
from .mcts_core import MCTSConfig
from .mcts_core import PIMCTreeReuse as _PIMCTreeReuse
from .mcts_core import pimc_mcts_policy as _pimc_policy
from .mcts_core import run_mcts_on_env as _run_mcts
from .model import BanqiTransformer, ModelOutputs


def _obs_batch_to_tensors(
    obs_list: List[Observation], max_history: int, device: torch.device
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Convert a batch of Observation objects to torch tensors."""
    B = len(obs_list)
    board = torch.from_numpy(np.stack([o.board_tokens for o in obs_list])).to(
        device=device, dtype=torch.long
    )
    belief = torch.from_numpy(np.stack([o.belief for o in obs_list])).to(
        device=device, dtype=torch.float32
    )

    hist_pad = np.full((B, max_history), -1, dtype=np.int64)
    for i, o in enumerate(obs_list):
        hist = o.history_actions[-max_history:] if max_history > 0 else []
        H = len(hist)
        if H > 0:
            hist_pad[i, -H:] = hist

    history_actions = torch.from_numpy(hist_pad).to(device=device, dtype=torch.long)

    to_play_color = torch.from_numpy(
        np.asarray([o.to_play_color for o in obs_list], dtype=np.int64)
    ).to(device=device, dtype=torch.long)

    plies = torch.from_numpy(
        np.asarray([o.plies for o in obs_list], dtype=np.int64)
    ).to(device=device, dtype=torch.long)
    no_progress_plies = torch.from_numpy(
        np.asarray([o.no_progress_plies for o in obs_list], dtype=np.int64)
    ).to(device=device, dtype=torch.long)

    legal = torch.from_numpy(
        np.stack([o.action_mask for o in obs_list]).astype(np.bool_)
    ).to(device=device, dtype=torch.bool)

    return (
        board,
        belief,
        history_actions,
        to_play_color,
        plies,
        no_progress_plies,
        legal,
    )


@torch.inference_mode()
def evaluate_batch(
    model: BanqiTransformer,
    obs_list: List[Observation],
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Torch evaluator for MCTS core.

    Returns:
      priors: [B,A] float32 normalized over legal actions
      values: [B] float32 in [-1,1]
    """
    board, belief, hist, to_play_color, plies, no_prog, legal = _obs_batch_to_tensors(
        obs_list, max_history=model.cfg.max_history, device=device
    )

    if amp_dtype is None:
        amp_dtype = torch.bfloat16

    with torch.autocast(
        device_type="cuda" if device.type == "cuda" else "cpu",
        dtype=amp_dtype,
        enabled=device.type == "cuda",
    ):
        out: ModelOutputs = model(
            board,
            belief,
            hist,
            to_play_color,
            plies,
            no_prog,
            belief_mask=None,
        )

    logits = out.policy_self_logits.float()  # [B,A]
    masked_logits = logits.masked_fill(~legal, -1e9)
    priors = F.softmax(masked_logits, dim=-1).cpu().numpy().astype(np.float32)
    values = out.value.float().cpu().numpy().astype(np.float32)
    return priors, values


@torch.no_grad()
def evaluate(
    model: BanqiTransformer,
    obs: Observation,
    device: torch.device,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Single-observation helper.

    Returns:
      priors: [A] float32 normalized over legal actions
      value: float in [-1,1]
      legal_mask: [A] bool
    """
    priors_b, values_b = evaluate_batch(model, [obs], device=device)
    priors = priors_b[0]
    value = float(values_b[0])
    legal_mask = obs.action_mask.astype(np.bool_)
    return priors, value, legal_mask


@torch.no_grad()
def run_mcts_on_env(
    root_env: BanqiEnv,
    model: BanqiTransformer,
    cfg: MCTSConfig,
    device: torch.device,
) -> np.ndarray:
    """Backwards-compatible wrapper: run MCTS on a determinized env using the torch model."""

    def pv_fn(obs_list: List[Observation]) -> Tuple[np.ndarray, np.ndarray]:
        return evaluate_batch(model, obs_list, device=device)

    return _run_mcts(
        root_env, policy_value_fn=pv_fn, cfg=cfg, max_history=model.cfg.max_history
    )


@torch.no_grad()
def pimc_mcts_policy(
    real_env: BanqiEnv,
    model: BanqiTransformer,
    cfg: MCTSConfig,
    device: torch.device,
    determinize_seeds: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """PIMC policy using the torch model.

    Returns:
      pi: [A] float32 probability distribution (temperature applied)
      visits: [A] float32 aggregated visit counts
      v_root: float value estimate at root (visit-weighted Q) from perspective of player to play
    """

    def pv_fn(obs_list: List[Observation]) -> Tuple[np.ndarray, np.ndarray]:
        return evaluate_batch(model, obs_list, device=device)

    return _pimc_policy(
        real_env,
        policy_value_fn=pv_fn,
        cfg=cfg,
        max_history=model.cfg.max_history,
        determinize_seeds=determinize_seeds,
    )


class PIMCTreeReuse:
    """Torch-backed root-sampling information-set MCTS with root reuse across moves."""

    def __init__(
        self,
        cfg: MCTSConfig,
        max_history: int,
        determinize_seeds: Optional[List[int]] = None,
    ) -> None:
        self._core = _PIMCTreeReuse(
            cfg=cfg, max_history=max_history, determinize_seeds=determinize_seeds
        )

    def reset(self) -> None:
        self._core.reset()

    def advance(self, action_id: int, is_flip: bool = False) -> None:
        self._core.advance(action_id, is_flip=is_flip)

    def policy(
        self,
        env: BanqiEnv,
        model: BanqiTransformer,
        device: torch.device,
        amp_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        def pv_fn(obs_list: List[Observation]) -> Tuple[np.ndarray, np.ndarray]:
            return evaluate_batch(model, obs_list, device=device, amp_dtype=amp_dtype)

        return self._core.policy(env, pv_fn)
