"""banqi.v1.eval

Modes:
1) Dataset evaluation:
   uv run -m banqi.v1.eval --mode dataset --data_dir data/ --checkpoint runs/ckpt_final --bf16

2) Arena evaluation between two checkpoints:
   uv run -m banqi.v1.eval --mode arena --checkpoint_a a --checkpoint_b b --num_games 50

Note: Arena uses the same root-sampling ISMCTS as training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .actions import ActionSpace
from .env import BanqiEnv
from .loss import LossConfig, MultiTaskLoss
from .mcts import MCTSConfig, pimc_mcts_policy
from .model import BanqiTransformer
from .utils import ReplayDataset, collate_fn, load_checkpoint


def dataset_eval(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    action_space = ActionSpace()
    action_dim = len(action_space)

    model, _ = load_checkpoint(Path(args.checkpoint))
    model = model.to(device)
    model.eval()
    print(f"Loaded: {args.checkpoint}")
    print(f"Params: {model.parameter_count() / 1e6:.2f}M")

    loss_cfg = LossConfig(
        w_self_policy=args.w_self_policy,
        w_value=args.w_value,
        w_belief=args.w_belief,
        value_loss=args.value_loss,
        entropy_coef=0.0,
    )
    criterion = MultiTaskLoss(loss_cfg)

    ds = ReplayDataset(Path(args.data_dir))
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(
            b, action_dim=action_dim, max_history=args.max_history
        ),
    )

    autocast = torch.autocast
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float32

    sums: Dict[str, float] = {}
    n = 0

    with torch.no_grad():
        for batch in dl:
            board = batch["board"].to(device, non_blocking=True)
            belief = batch["belief"].to(device, non_blocking=True)
            history = batch["history"].to(device, non_blocking=True)
            to_play_color = batch["to_play_color"].to(device, non_blocking=True)
            plies = batch["plies"].to(device, non_blocking=True)
            no_progress_plies = batch["no_progress_plies"].to(device, non_blocking=True)
            action_mask = batch["action_mask"].to(device, non_blocking=True)
            pi_self = batch["pi_self"].to(device, non_blocking=True)
            pi_opp = (
                batch["pi_opp"].to(device, non_blocking=True)
                if batch["pi_opp"] is not None
                else None
            )
            value = batch["value"].to(device, non_blocking=True)
            belief_target = batch["belief_target"].to(device, non_blocking=True)

            with autocast(
                device_type=amp_device,
                dtype=amp_dtype,
                enabled=args.bf16 and device.type == "cuda",
            ):
                out = model(
                    board,
                    belief,
                    history,
                    to_play_color,
                    plies,
                    no_progress_plies,
                    belief_mask=None,
                )
                loss, metrics = criterion(
                    out=out,
                    target_self_policy=pi_self,
                    target_value=value,
                    target_belief=belief_target,
                    legal_action_mask=action_mask,
                    belief_loss_mask=None,
                )

            bs = board.shape[0]
            n += bs
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + float(v) * bs

    print("---- Dataset metrics (avg) ----")
    for k in sorted(sums.keys()):
        print(f"{k:20s}: {sums[k] / max(1, n):.6f}")


def play_arena_game(
    model_a: BanqiTransformer,
    model_b: BanqiTransformer,
    device: torch.device,
    action_space: ActionSpace,
    mcts_cfg: MCTSConfig,
    max_history: int,
    max_plies: int = 512,
) -> int:
    """Returns winner player id (0 or 1) or -1 for draw.

    Player 0 uses model_a, Player 1 uses model_b.
    """
    env = BanqiEnv(action_space=action_space, max_history=max_history)

    plies = 0
    while not env.done and plies < max_plies:
        obs = env.observe(max_history=max_history)

        model = model_a if env.to_play == 0 else model_b
        pi, _visits, _v_root = pimc_mcts_policy(env, model, mcts_cfg, device=device)

        legal = obs.action_mask.astype(np.bool_)
        pi[~legal] = 0.0
        if pi.sum() <= 0:
            legal_ids = np.nonzero(legal)[0]
            a_id = int(np.random.choice(legal_ids))
        else:
            a_id = int(np.argmax(pi))  # eval: greedy

        env.step(a_id)
        plies += 1

    if env.winner is None:
        return -1
    return int(env.winner)


def arena_eval(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    action_space = ActionSpace()

    def load_model(ckpt: str) -> BanqiTransformer:
        model, _ = load_checkpoint(Path(ckpt))
        model = model.to(device)
        model.eval()
        return model

    model_a = load_model(args.checkpoint_a)
    model_b = load_model(args.checkpoint_b)

    mcts_cfg = MCTSConfig(
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
        temperature=0.0,  # greedy
        num_determinize=args.num_determinize,
    )

    wins = {0: 0, 1: 0, -1: 0}
    for g in range(args.num_games):
        w = play_arena_game(
            model_a=model_a,
            model_b=model_b,
            device=device,
            action_space=action_space,
            mcts_cfg=mcts_cfg,
            max_history=args.max_history,
            max_plies=args.max_plies,
        )
        wins[w] += 1
        print(f"[arena] game {g + 1}/{args.num_games} winner: {w}")

    print("---- Arena results ----")
    print(f"A wins (P0): {wins[0]}")
    print(f"B wins (P1): {wins[1]}")
    print(f"Draws      : {wins[-1]}")
    if args.num_games > 0:
        print(
            f"A winrate: {wins[0] / args.num_games:.3f} (draws {wins[-1] / args.num_games:.3f})"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dataset", "arena"], required=True)

    # model config (must match training)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--n_layers", type=int, default=16)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--d_ff", type=int, default=3072)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_history", type=int, default=16)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--bf16", action="store_true")

    # dataset mode
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)

    # loss weights (for reporting)
    p.add_argument("--w_self_policy", type=float, default=1.0)
    p.add_argument("--w_value", type=float, default=1.0)
    p.add_argument("--w_belief", type=float, default=0.25)
    p.add_argument("--value_loss", type=str, default="huber", choices=["huber", "mse"])

    # arena mode
    p.add_argument("--checkpoint_a", type=str, default="")
    p.add_argument("--checkpoint_b", type=str, default="")
    p.add_argument("--num_games", type=int, default=20)
    p.add_argument("--max_plies", type=int, default=512)
    p.add_argument("--num_simulations", type=int, default=200)
    p.add_argument("--num_determinize", type=int, default=8)
    p.add_argument("--c_puct", type=float, default=1.5)

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "dataset":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for dataset mode")
        dataset_eval(args)
    else:
        if not args.checkpoint_a or not args.checkpoint_b:
            raise ValueError(
                "--checkpoint_a and --checkpoint_b required for arena mode"
            )
        arena_eval(args)


if __name__ == "__main__":
    main()
