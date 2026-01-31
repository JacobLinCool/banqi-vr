from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .actions import ActionSpace
from .model import BanqiTransformer


def save_checkpoint(
    path: Path,
    model: BanqiTransformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: Dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    obj = {
        "optimizer": optimizer.state_dict(),
        "step": step,
        "cfg": cfg,
    }
    torch.save(obj, path / "trainer_state.pt")


def load_checkpoint(
    path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[BanqiTransformer, int]:
    if path.is_file():
        raise ValueError(
            f"Checkpoint must be a directory saved with save_pretrained, got file: {path}"
        )
    model = BanqiTransformer.from_pretrained(path)
    state_path = path / "trainer_state.pt"
    if state_path.exists():
        ckpt = torch.load(state_path, map_location="cpu")
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        return model, int(ckpt.get("step", 0))
    return model, 0


def load_optimizer_state(path: Path, optimizer: torch.optim.Optimizer) -> None:
    state_path = path / "trainer_state.pt"
    if not state_path.exists():
        return
    ckpt = torch.load(state_path, map_location="cpu")
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_schedule(step: int, total: int, base_lr: float, warmup: int = 1000) -> float:
    if step < warmup:
        return base_lr * float(step) / float(max(1, warmup))
    t = float(step - warmup) / float(max(1, total - warmup))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


def scheduled_temperature(
    plies: int,
    temp_early: float,
    temp_late: float,
    switch_plies: int,
    fallback: float,
) -> float:
    if switch_plies <= 0:
        return float(fallback)
    return float(temp_early if plies < switch_plies else temp_late)


def visits_to_pi(
    visits: np.ndarray, legal_mask: np.ndarray, temperature: float
) -> np.ndarray:
    """Convert visit counts to a probability distribution over actions."""
    v = visits.astype(np.float32, copy=False)
    legal = legal_mask.astype(np.bool_, copy=False)

    if temperature <= 1e-8:
        out = np.zeros_like(v, dtype=np.float32)
        v2 = v.copy()
        v2[~legal] = -1.0
        best = int(np.argmax(v2))
        if legal[best]:
            out[best] = 1.0
            return out
        legal_ids = np.nonzero(legal)[0]
        out[int(np.random.choice(legal_ids))] = 1.0
        return out

    x = np.power(v + 1e-8, 1.0 / float(temperature))
    x[~legal] = 0.0
    z = float(x.sum())
    if z <= 0:
        out = legal.astype(np.float32)
        out /= float(out.sum() + 1e-8)
        return out
    return (x / z).astype(np.float32)


def build_mirror_maps(
    action_space: ActionSpace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    square_map = np.asarray(
        [action_space.mirror_square_index(i) for i in range(action_space.num_squares)],
        dtype=np.int64,
    )
    action_map = np.asarray(
        [action_space.mirror_action_id(a) for a in range(len(action_space))],
        dtype=np.int64,
    )
    inv_action_map = np.empty_like(action_map)
    inv_action_map[action_map] = np.arange(len(action_map), dtype=np.int64)
    return square_map, action_map, inv_action_map


def mirror_sample(
    sample: Dict[str, Any],
    square_map: np.ndarray,
    action_map: np.ndarray,
    inv_action_map: np.ndarray,
) -> Dict[str, Any]:
    board = sample["board_tokens"][square_map]
    belief = sample["belief"][square_map]
    belief_target = sample.get("belief_target", sample["belief"])[square_map]

    hist = sample["history_actions"].copy()
    if hist.size > 0:
        m = hist >= 0
        if m.any():
            hist[m] = action_map[hist[m]]

    action_mask = sample["action_mask"][inv_action_map]
    pi_self = sample["pi_self"][inv_action_map]
    pi_opp = None
    if sample.get("pi_opp", None) is not None:
        pi_opp = sample["pi_opp"][inv_action_map]

    return {
        "board_tokens": board,
        "belief": belief,
        "history_actions": hist,
        "to_play_color": int(sample["to_play_color"]),
        "plies": int(sample.get("plies", 0)),
        "no_progress_plies": int(sample.get("no_progress_plies", 0)),
        "to_play_player": int(sample["to_play_player"]),
        "action_mask": action_mask,
        "pi_self": pi_self,
        "pi_opp": pi_opp,
        "value": float(sample["value"]),
        "belief_target": belief_target,
        "v_root": float(sample.get("v_root", 0.0)),
        "captured_counts": sample.get("captured_counts", None),
    }
