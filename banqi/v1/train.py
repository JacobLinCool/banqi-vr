"""banqi.v1.train

Modes:
1) Async AlphaZero-style pipeline (actors + GPU batched inference + learner):
   uv run -m banqi.v1.train --mode az --save_dir runs_az/ --device cuda --num_actors 32 --bf16

Notes on improvements
---------------------------
- Search: the MCTS core uses an information-set hash so statistics are shared across
  determinizations (root-sampling ISMCTS), rather than one separate tree per determinization.
- Training/pipeline:
  * Split policy target for play vs train: pi_play uses scheduled temperature for exploration,
    pi_self stored in samples is pi_train (typically visits-normalized, optionally sharpened).
  * Bootstrapped value target: optionally mix terminal z with v_root (MCTS root value estimate).
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["az", "az_split"], required=True)

    # common model args
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=16)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_history", type=int, default=64)
    p.add_argument(
        "--use_reduced_belief",
        action="store_true",
        help="use 1 global belief token instead of 32 per-square tokens",
    )

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_mode", type=str, default="reduce-overhead")

    # selfplay (single process)
    p.add_argument("--out_dir", type=str, default="data")
    p.add_argument("--num_games", type=int, default=20)
    p.add_argument("--max_plies", type=int, default=512)
    p.add_argument("--num_simulations", type=int, default=300)
    p.add_argument("--num_determinize", type=int, default=8)
    p.add_argument("--batch_leaves", type=int, default=64)
    p.add_argument("--c_puct", type=float, default=1.5)
    p.add_argument("--dirichlet_alpha", type=float, default=0.3)
    p.add_argument("--dirichlet_epsilon", type=float, default=0.25)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--temp_early",
        type=float,
        default=1.0,
        help="selfplay/az: temperature before --temp_switch_plies",
    )
    p.add_argument(
        "--temp_late",
        type=float,
        default=0.3,
        help="selfplay/az: temperature after --temp_switch_plies",
    )
    p.add_argument(
        "--temp_switch_plies",
        type=int,
        default=16,
        help="selfplay/az: ply threshold to switch temp (<=0 disables schedule)",
    )
    p.add_argument(
        "--pi_train_temperature",
        type=float,
        default=1.0,
        help="temperature used to convert MCTS visits into the stored policy target pi_self",
    )
    p.add_argument(
        "--mirror_augment",
        action="store_true",
        help="selfplay/az: add left-right mirrored samples",
    )

    # train (offline) + learner (az)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--save_dir", type=str, default="runs")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--total_steps", type=int, default=500000)
    p.add_argument("--warmup_steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=10000)

    # tensorboard
    p.add_argument("--tb_dir", type=str, default="")
    p.add_argument("--run_name", type=str, default="")

    # loss weights
    p.add_argument("--w_self_policy", type=float, default=1.0)
    p.add_argument("--w_value", type=float, default=1.0)
    p.add_argument("--w_belief", type=float, default=0.10)
    p.add_argument("--value_loss", type=str, default="huber", choices=["huber", "mse"])
    p.add_argument("--entropy_coef", type=float, default=0.0)

    # belief denoising
    p.add_argument("--belief_mask_prob", type=float, default=0.15)
    p.add_argument("--belief_loss_only_masked", action="store_true")

    # bootstrapped value target
    p.add_argument(
        "--value_target_mix",
        type=float,
        default=0.25,
        help="mix terminal z with v_root: (1-mix)*z + mix*v_root (0 disables)",
    )

    # -------- async AZ pipeline --------
    p.add_argument("--num_actors", type=int, default=32)
    p.add_argument(
        "--games_per_actor", type=int, default=0, help="0 => run until learner stops"
    )
    p.add_argument("--actor_log_every", type=int, default=0)
    p.add_argument(
        "--actor_timing_every",
        type=int,
        default=0,
        help="log per-ply timing every N plies (0 disables)",
    )

    # devices for az_split (full actor/infer/learner separation)
    p.add_argument(
        "--infer_device", type=str, default="", help="default: same as --device"
    )
    p.add_argument(
        "--learner_device", type=str, default="", help="default: same as --device"
    )
    p.add_argument(
        "--sync_every",
        type=int,
        default=200,
        help="az_split: sync weights to inference every N learner steps (0 disables)",
    )

    # replay buffer
    p.add_argument("--replay_size", type=int, default=1000000)
    p.add_argument("--train_start", type=int, default=10000)
    p.add_argument("--replay_drain_max", type=int, default=128)

    # inference batching
    p.add_argument("--infer_max_batch", type=int, default=1024)
    p.add_argument("--infer_batch_window_ms", type=float, default=3.0)
    p.add_argument("--infer_poll_timeout_ms", type=float, default=2.0)
    p.add_argument("--infer_batches_per_train", type=int, default=4)
    p.add_argument("--max_infer_batches_per_loop", type=int, default=16)

    # queue sizes (tune if you see deadlocks/backpressure)
    p.add_argument("--infer_queue_size", type=int, default=8192)
    p.add_argument("--infer_resp_queue_size", type=int, default=256)
    p.add_argument("--replay_queue_size", type=int, default=256)
    p.add_argument("--control_queue_size", type=int, default=512)

    # optional: lagged/EMA inference weights (az integrated only)
    p.add_argument(
        "--infer_sync_every",
        type=int,
        default=50,
        help="az: sync model_train -> model_infer every N learner steps",
    )
    p.add_argument(
        "--infer_ema_decay",
        type=float,
        default=0.9,
        help="az: if >0, model_infer is EMA of model_train with this decay",
    )

    # optional: reanalysis (learner recomputes pi/v for replay positions)
    p.add_argument("--reanalysis_every", type=int, default=1000)
    p.add_argument("--reanalysis_batch_size", type=int, default=128)
    p.add_argument("--reanalysis_simulations", type=int, default=128)
    p.add_argument("--reanalysis_determinize", type=int, default=4)

    # optional: prioritized replay
    p.add_argument("--prioritized_replay", action="store_true")
    p.add_argument("--per_alpha", type=float, default=0.6)
    p.add_argument("--per_eps", type=float, default=1e-3)
    p.add_argument("--per_value_coef", type=float, default=1.0)
    p.add_argument("--per_policy_coef", type=float, default=1.0)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "az":
        from .async_az import az_main

        az_main(args)
    else:
        from .async_az_split import az_split_main

        # default devices
        if not args.infer_device:
            args.infer_device = args.device
        if not args.learner_device:
            args.learner_device = args.device
        az_split_main(args)


if __name__ == "__main__":
    main()
