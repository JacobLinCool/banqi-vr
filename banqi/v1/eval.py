"""banqi.v1.eval

Modes:
1) Arena evaluation between two checkpoints:
    uv run -m banqi.v1.eval --mode arena --checkpoint_a a --checkpoint_b b --num_games 100
2) Model vs random policy:
    uv run -m banqi.v1.eval --mode random --checkpoint a --num_games 100

Note: Arena uses the same root-sampling ISMCTS as training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .actions import ActionSpace
from .env import (
    BLACK,
    KIND_COUNTS,
    KIND_NAMES,
    PID_COLOR,
    PID_KIND,
    RED,
    BanqiEnv,
)
from .mcts import MCTSConfig, pimc_mcts_policy
from .model import BanqiTransformer
from .utils import load_checkpoint


def _color_label(color: int | None) -> str:
    if color is None:
        return "?"
    return "R" if int(color) == RED else "B"


def _summarize_game(env: BanqiEnv, max_plies: int) -> dict[str, object]:
    total_per_color = sum(KIND_COUNTS.values())
    captured_by_color = {RED: 0, BLACK: 0}
    captured_by_kind = {
        RED: {k: 0 for k in range(len(KIND_NAMES))},
        BLACK: {k: 0 for k in range(len(KIND_NAMES))},
    }

    for pid in env.captured:
        color = int(PID_COLOR[pid])
        kind = int(PID_KIND[pid])
        captured_by_color[color] += 1
        captured_by_kind[color][kind] += 1

    remaining_by_color = {
        RED: total_per_color - captured_by_color[RED],
        BLACK: total_per_color - captured_by_color[BLACK],
    }

    if env.winner is None:
        if env.no_progress_plies >= env.draw_plies:
            reason = "no-progress"
        elif env.plies >= max_plies:
            reason = "max-plies"
        else:
            reason = "draw"
    else:
        if remaining_by_color[RED] == 0 or remaining_by_color[BLACK] == 0:
            reason = "all-captured"
        else:
            reason = "stalemate"

    return {
        "plies": int(env.plies),
        "no_progress_plies": int(env.no_progress_plies),
        "winner": env.winner,
        "reason": reason,
        "player_colors": (
            _color_label(env.player_color[0]),
            _color_label(env.player_color[1]),
        ),
        "captured_by_color": captured_by_color,
        "remaining_by_color": remaining_by_color,
        "captured_by_kind": captured_by_kind,
        "final_board": env.render(),
    }


def _format_captured_by_kind(captured_by_kind: dict[int, dict[int, int]]) -> str:
    def fmt(color: int) -> str:
        parts = []
        for k, name in enumerate(KIND_NAMES):
            parts.append(f"{name}{captured_by_kind[color][k]}")
        return " ".join(parts)

    return f"R: {fmt(RED)} | B: {fmt(BLACK)}"


def play_arena_game(
    model_a: BanqiTransformer,
    model_b: BanqiTransformer,
    device: torch.device,
    action_space: ActionSpace,
    mcts_cfg: MCTSConfig,
    max_history: int,
    max_plies: int = 512,
) -> tuple[int, dict[str, object]]:
    """Returns (winner player id (0 or 1) or -1 for draw, summary dict).

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

    summary = _summarize_game(env, max_plies=max_plies)
    if env.winner is None:
        return -1, summary
    return int(env.winner), summary


def play_random_game(
    model: BanqiTransformer,
    device: torch.device,
    action_space: ActionSpace,
    mcts_cfg: MCTSConfig,
    max_history: int,
    max_plies: int = 512,
) -> tuple[int, dict[str, object]]:
    """Returns (winner player id (0 or 1) or -1 for draw, summary dict).

    Player 0 uses model, Player 1 is random.
    """
    env = BanqiEnv(action_space=action_space, max_history=max_history)

    plies = 0
    while not env.done and plies < max_plies:
        obs = env.observe(max_history=max_history)

        if env.to_play == 0:
            pi, _visits, _v_root = pimc_mcts_policy(env, model, mcts_cfg, device=device)
            legal = obs.action_mask.astype(np.bool_)
            pi[~legal] = 0.0
            if pi.sum() <= 0:
                legal_ids = np.nonzero(legal)[0]
                a_id = int(np.random.choice(legal_ids))
            else:
                a_id = int(np.argmax(pi))  # eval: greedy
        else:
            legal = obs.action_mask.astype(np.bool_)
            legal_ids = np.nonzero(legal)[0]
            a_id = int(np.random.choice(legal_ids))

        env.step(a_id)
        plies += 1

    summary = _summarize_game(env, max_plies=max_plies)
    if env.winner is None:
        return -1, summary
    return int(env.winner), summary


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

    if model_a.cfg.max_history != model_b.cfg.max_history:
        raise ValueError("Model max_history mismatch between A and B")
    max_history = model_a.cfg.max_history

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
        w, summary = play_arena_game(
            model_a=model_a,
            model_b=model_b,
            device=device,
            action_space=action_space,
            mcts_cfg=mcts_cfg,
            max_history=max_history,
            max_plies=args.max_plies,
        )
        wins[w] += 1
        p0_color, p1_color = summary["player_colors"]
        print(
            "[arena] game "
            f"{g + 1}/{args.num_games} "
            f"winner: {w} "
            f"(P0={p0_color}, P1={p1_color}) "
            f"plies={summary['plies']} "
            f"no_progress={summary['no_progress_plies']} "
            f"reason={summary['reason']} "
            f"remaining R/B={summary['remaining_by_color'][RED]}/"
            f"{summary['remaining_by_color'][BLACK]} "
            f"captured R/B={summary['captured_by_color'][RED]}/"
            f"{summary['captured_by_color'][BLACK]}"
        )
        print(
            "[arena] captured by kind "
            f"{_format_captured_by_kind(summary['captured_by_kind'])}"
        )
        if args.print_final_board:
            print("[arena] final board")
            print(summary["final_board"])

    print("---- Arena results ----")
    print(f"A wins (P0): {wins[0]}")
    print(f"B wins (P1): {wins[1]}")
    print(f"Draws      : {wins[-1]}")
    if args.num_games > 0:
        print(
            f"A winrate: {wins[0] / args.num_games:.3f} (draws {wins[-1] / args.num_games:.3f})"
        )


def random_eval(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    action_space = ActionSpace()

    model, _ = load_checkpoint(Path(args.checkpoint))
    model = model.to(device)
    model.eval()

    max_history = model.cfg.max_history

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
        w, summary = play_random_game(
            model=model,
            device=device,
            action_space=action_space,
            mcts_cfg=mcts_cfg,
            max_history=max_history,
            max_plies=args.max_plies,
        )
        wins[w] += 1
        p0_color, p1_color = summary["player_colors"]
        print(
            "[random] game "
            f"{g + 1}/{args.num_games} "
            f"winner: {w} "
            f"(P0={p0_color}, P1={p1_color}) "
            f"plies={summary['plies']} "
            f"no_progress={summary['no_progress_plies']} "
            f"reason={summary['reason']} "
            f"remaining R/B={summary['remaining_by_color'][RED]}/"
            f"{summary['remaining_by_color'][BLACK]} "
            f"captured R/B={summary['captured_by_color'][RED]}/"
            f"{summary['captured_by_color'][BLACK]}"
        )
        print(
            "[random] captured by kind "
            f"{_format_captured_by_kind(summary['captured_by_kind'])}"
        )
        if args.print_final_board:
            print("[random] final board")
            print(summary["final_board"])

    print("---- Random results ----")
    print(f"Model wins (P0): {wins[0]}")
    print(f"Random wins (P1): {wins[1]}")
    print(f"Draws          : {wins[-1]}")
    if args.num_games > 0:
        print(
            f"Model winrate: {wins[0] / args.num_games:.3f} (draws {wins[-1] / args.num_games:.3f})"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["arena", "random"], required=True)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--bf16", action="store_true")

    # arena mode
    p.add_argument("--checkpoint_a", type=str, default="")
    p.add_argument("--checkpoint_b", type=str, default="")
    p.add_argument("--num_games", type=int, default=100)
    p.add_argument("--max_plies", type=int, default=512)
    p.add_argument("--num_simulations", type=int, default=300)
    p.add_argument("--num_determinize", type=int, default=8)
    p.add_argument("--c_puct", type=float, default=1.5)

    # random mode
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--print_final_board", action="store_true")

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "arena":
        if not args.checkpoint_a or not args.checkpoint_b:
            raise ValueError(
                "--checkpoint_a and --checkpoint_b required for arena mode"
            )
        arena_eval(args)
    elif args.mode == "random":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for random mode")
        random_eval(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
