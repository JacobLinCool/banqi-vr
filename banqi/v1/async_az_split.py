"""banqi.v1.async_az_split

Full Actor / Inference / Learner separation into **three roles**:

  - Actors (CPU, multi-process): run self-play games + root-sampling ISMCTS.
  - Inference server (GPU): batches inference requests across all actors.
  - Learner (GPU): consumes self-play data and trains the model, periodically syncing weights
    to the inference server.

Single-GPU note
---------------
If you run both inference server and learner on the same GPU, you will have TWO model copies
resident on that GPU (one in each process). For large models this may exceed VRAM.
Recommended:
  - Use 2 GPUs: --infer_device cuda:0 --learner_device cuda:1
  - Or use the integrated single-process version in async_az.py (--mode az)

This file includes the same improvements as async_az.py:
  - Information-set hashing (in mcts_core)
  - Separate pi_play (action selection) vs pi_train (teacher distribution)
  - Optional bootstrapped value targets via stored v_root
  - Optional prioritized replay and optional reanalysis
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import time
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .actions import ActionSpace
from .async_common import DummyWriter, InferenceClient, ReplayBuffer, collate_batch
from .env import COVERED_TOK, BanqiEnv
from .loss import LossConfig, MultiTaskLoss
from .mcts_core import MCTSConfig, PIMCTreeReuse, pimc_mcts_policy
from .model import BanqiTransformer, ModelConfig
from .utils import (
    build_mirror_maps,
    cosine_schedule,
    load_checkpoint,
    load_optimizer_state,
    mirror_sample,
    save_checkpoint,
    scheduled_temperature,
    set_seed,
    visits_to_pi,
)


# ----------------- Actor process -----------------
def actor_process(
    actor_id: int,
    args: argparse.Namespace,
    infer_req_q: mp.Queue,
    infer_resp_q: mp.Queue,
    replay_q: mp.Queue,
    stop_event: MpEvent,
) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    set_seed(int(args.seed) + 1000 * int(actor_id))

    action_space = ActionSpace()
    client = InferenceClient(
        actor_id,
        infer_req_q,
        infer_resp_q,
        max_history=args.max_history,
        timing_every=args.actor_timing_every,
    )

    mcts_cfg = MCTSConfig(
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        temperature=args.temperature,
        num_determinize=args.num_determinize,
        batch_leaves=args.batch_leaves,
    )

    mirror_maps = None
    if args.mirror_augment:
        mirror_maps = build_mirror_maps(action_space)

    games = 0
    pbar = tqdm(
        desc=f"actor-{actor_id}",
        unit="game",
        dynamic_ncols=True,
        position=int(actor_id),
        leave=False,
    )
    while not stop_event.is_set():
        env = BanqiEnv(action_space=action_space, max_history=args.max_history)
        samples: List[Dict[str, Any]] = []
        plies = 0
        determinize_seeds = [
            int(np.random.randint(0, 2**31 - 1))
            for _ in range(mcts_cfg.num_determinize)
        ]
        reuse_ctx = PIMCTreeReuse(
            cfg=mcts_cfg,
            max_history=args.max_history,
            determinize_seeds=determinize_seeds,
        )

        timing_every = int(args.actor_timing_every)
        timing_enabled = timing_every > 0

        while (not env.done) and (plies < args.max_plies) and (not stop_event.is_set()):
            mcts_cfg.temperature = scheduled_temperature(
                plies,
                temp_early=args.temp_early,
                temp_late=args.temp_late,
                switch_plies=args.temp_switch_plies,
                fallback=args.temperature,
            )
            if timing_enabled:
                t0 = time.perf_counter()
            obs = env.observe(max_history=args.max_history)
            if timing_enabled:
                t1 = time.perf_counter()

            pi_play, visits, v_root = reuse_ctx.policy(
                env, policy_value_fn=client.policy_value_batch
            )
            if timing_enabled:
                t2 = time.perf_counter()

            # teacher policy
            pi_self = visits_to_pi(
                visits,
                obs.action_mask.astype(np.bool_),
                temperature=float(args.pi_train_temperature),
            )

            # choose action for play from pi_play
            legal = obs.action_mask.astype(np.bool_)
            pi_for_play = pi_play.copy()
            pi_for_play[~legal] = 0.0
            s = float(pi_for_play.sum())
            if s <= 0:
                legal_ids = np.nonzero(legal)[0]
                a_id = int(np.random.choice(legal_ids))
            else:
                pi_for_play /= s
                a_id = int(np.random.choice(np.arange(len(pi_for_play)), p=pi_for_play))

            # captured counts snapshot (for possible reanalysis)
            cap = np.zeros(15, dtype=np.int16)
            for pid in env.captured:
                cap[int(pid)] += 1

            samples.append(
                {
                    "board_tokens": obs.board_tokens.astype(np.int64),
                    "belief": obs.belief.astype(np.float32),
                    "history_actions": obs.history_actions.astype(np.int64),
                    "to_play_color": int(obs.to_play_color),
                    "plies": int(obs.plies),
                    "no_progress_plies": int(obs.no_progress_plies),
                    "to_play_player": int(env.to_play),
                    "action_mask": obs.action_mask.astype(np.bool_),
                    "pi_self": pi_self.astype(np.float32),
                    "value": 0.0,
                    "belief_target": obs.belief.astype(np.float32),
                    "v_root": float(v_root),
                    "captured_counts": cap,
                }
            )

            env.step(a_id)
            reuse_ctx.advance(a_id, is_flip=action_space.is_flip(a_id))
            plies += 1

            if timing_enabled and (plies % timing_every == 0):
                t3 = time.perf_counter()
                print(
                    f"[actor {actor_id}] observe={t1 - t0:.3f}s "
                    f"self_mcts={t2 - t1:.3f}s step={t3 - t2:.3f}s "
                    f"total={t3 - t0:.3f}s plies={plies}"
                )

        if env.winner is None:
            result = {0: 0.0, 1: 0.0}
        else:
            result = {env.winner: 1.0, 1 - env.winner: -1.0}
        for s in samples:
            s["value"] = float(result[int(s["to_play_player"])])

        if mirror_maps is not None:
            square_map, action_map, inv_action_map = mirror_maps
            samples.extend(
                [
                    mirror_sample(s, square_map, action_map, inv_action_map)
                    for s in samples
                ]
            )

        if samples:
            replay_q.put(samples)

        games += 1
        pbar.update(1)
        if args.games_per_actor > 0 and games >= args.games_per_actor:
            break

    pbar.close()


# ----------------- Inference server (GPU) -----------------
def inference_server_process(
    args: argparse.Namespace,
    infer_req_q: mp.Queue,
    infer_resp_qs: List[mp.Queue],
    weight_q: mp.Queue,
    stop_event: MpEvent,
) -> None:
    set_seed(int(args.seed) + 12345)
    torch.manual_seed(int(args.seed) + 12345)

    device = torch.device(args.infer_device)
    action_dim = len(ActionSpace())

    if args.checkpoint and Path(args.checkpoint).exists():
        model, _ = load_checkpoint(Path(args.checkpoint))
        model = model.to(device)
        model.eval()
        print(f"[infer] loaded checkpoint {args.checkpoint}")
    else:
        model_cfg = ModelConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            action_dim=action_dim,
            max_history=args.max_history,
            use_reduced_belief=args.use_reduced_belief,
            max_plies=args.max_plies,
        )
        model = BanqiTransformer(model_cfg).to(device)
        model.eval()

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)

    backlog: List[Dict[str, Any]] = []
    last_weight_step = -1

    def maybe_update_weights() -> None:
        nonlocal last_weight_step
        while True:
            try:
                msg = weight_q.get_nowait()
            except queue.Empty:
                break
            if msg is None:
                continue
            if msg.get("type") == "weights_path":
                step = int(msg.get("step", -1))
                if step <= last_weight_step:
                    continue
                path = Path(msg["path"])
                if path.exists():
                    state = torch.load(path, map_location="cpu")
                    model.load_state_dict(state, strict=True)
                    last_weight_step = step
                    model.eval()

    def gather_requests() -> Optional[List[Dict[str, Any]]]:
        if backlog:
            first = backlog.pop(0)
        else:
            try:
                first = infer_req_q.get(timeout=args.infer_poll_timeout_ms / 1000.0)
            except queue.Empty:
                return None
        if first.get("type") == "shutdown":
            return [first]
        reqs = [first]
        total = int(first["board"].shape[0])
        t_start = time.time()
        while total < args.infer_max_batch:
            if (time.time() - t_start) * 1000.0 >= args.infer_batch_window_ms:
                break
            try:
                r = infer_req_q.get_nowait()
            except queue.Empty:
                break
            if r.get("type") == "shutdown":
                backlog.append(r)
                break
            n = int(r["board"].shape[0])
            if total + n > args.infer_max_batch:
                backlog.append(r)
                break
            reqs.append(r)
            total += n
        return reqs

    autocast = torch.autocast
    while True:
        maybe_update_weights()
        reqs = gather_requests()
        if reqs is None:
            continue
        if len(reqs) == 1 and reqs[0].get("type") == "shutdown":
            break

        boards = np.concatenate([r["board"] for r in reqs], axis=0)
        beliefs = np.concatenate([r["belief"] for r in reqs], axis=0)
        hists = np.concatenate([r["history"] for r in reqs], axis=0)
        tpc = np.concatenate([r["to_play_color"] for r in reqs], axis=0)
        plies = np.concatenate([r["plies"] for r in reqs], axis=0)
        no_progress_plies = np.concatenate(
            [r["no_progress_plies"] for r in reqs], axis=0
        )
        legal = np.concatenate([r["legal"] for r in reqs], axis=0)

        board_t = torch.from_numpy(boards).to(
            device=device, dtype=torch.long, non_blocking=True
        )
        belief_t = torch.from_numpy(beliefs).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        hist_t = torch.from_numpy(hists).to(
            device=device, dtype=torch.long, non_blocking=True
        )
        tpc_t = torch.from_numpy(tpc).to(
            device=device, dtype=torch.long, non_blocking=True
        )
        plies_t = torch.from_numpy(plies).to(
            device=device, dtype=torch.long, non_blocking=True
        )
        no_prog_t = torch.from_numpy(no_progress_plies).to(
            device=device, dtype=torch.long, non_blocking=True
        )
        legal_t = torch.from_numpy(legal).to(
            device=device, dtype=torch.bool, non_blocking=True
        )

        with torch.inference_mode():
            with autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                out = model(
                    board_t,
                    belief_t,
                    hist_t,
                    tpc_t,
                    plies_t,
                    no_prog_t,
                    belief_mask=None,
                )
            logits = out.policy_self_logits.float()
            masked = logits.masked_fill(~legal_t, -1e9)
            priors = F.softmax(masked, dim=-1).cpu().numpy().astype(np.float32)
            values = out.value.float().cpu().numpy().astype(np.float32)

        off = 0
        for r in reqs:
            bsz = int(r["board"].shape[0])
            p = priors[off : off + bsz]
            v = values[off : off + bsz]
            off += bsz
            aid = int(r["actor_id"])
            infer_resp_qs[aid].put(
                {"req_id": int(r["req_id"]), "priors": p, "values": v}
            )

    print("[infer] shutdown")


def learner_process(
    args: argparse.Namespace,
    replay_q: mp.Queue,
    weight_q: mp.Queue,
    stop_event: MpEvent,
) -> None:
    set_seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device(args.learner_device)
    torch.backends.cuda.matmul.allow_tf32 = True
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    action_space = ActionSpace()
    action_dim = len(action_space)

    if args.checkpoint and Path(args.checkpoint).exists():
        model, start_step = load_checkpoint(Path(args.checkpoint))
        model = model.to(device)
        model_cfg = model.cfg
    else:
        model_cfg = ModelConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            action_dim=action_dim,
            max_history=args.max_history,
            use_reduced_belief=args.use_reduced_belief,
            max_plies=args.max_plies,
        )
        model = BanqiTransformer(model_cfg).to(device)
        start_step = 0

    loss_cfg = LossConfig(
        w_self_policy=args.w_self_policy,
        w_value=args.w_value,
        w_belief=args.w_belief,
        value_loss=args.value_loss,
        entropy_coef=args.entropy_coef,
    )
    criterion = MultiTaskLoss(loss_cfg)

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            fused=(device.type == "cuda"),
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    if args.checkpoint and Path(args.checkpoint).exists():
        load_optimizer_state(Path(args.checkpoint), optimizer)
        print(f"[learner] loaded checkpoint {args.checkpoint} at step {start_step}")

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    writer = (
        SummaryWriter(log_dir=str(save_dir / "tb_split"))
        if SummaryWriter is not None
        else DummyWriter()
    )

    replay = ReplayBuffer(
        args.replay_size,
        prioritized=bool(getattr(args, "prioritized_replay", False)),
        alpha=float(getattr(args, "per_alpha", 0.6)),
        eps=float(getattr(args, "per_eps", 1e-3)),
    )

    autocast = torch.autocast
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float32

    pbar = tqdm(
        total=args.total_steps,
        initial=int(start_step),
        desc="learner",
        unit="step",
        dynamic_ncols=True,
    )

    def write_weights(step: int) -> None:
        tmp = save_dir / "_infer_weights.tmp"
        final = save_dir / "infer_weights.pt"
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, final)
        weight_q.put({"type": "weights_path", "step": step, "path": str(final)})

    write_weights(start_step)

    step = start_step
    last_sync = step
    t0 = time.time()

    last_reanalysis = step

    def sample_to_env(sample: Dict[str, Any]) -> BanqiEnv:
        env = BanqiEnv(action_space=action_space, max_history=args.max_history)

        bt = sample["board_tokens"].astype(np.int64, copy=False)
        env.piece = [0] * action_space.num_squares
        env.revealed = [False] * action_space.num_squares
        for i, tok in enumerate(bt.tolist()):
            if tok == 1:  # EMPTY_TOK
                env.piece[i] = 0
                env.revealed[i] = False
            elif tok == 2:  # COVERED_TOK
                env.piece[i] = 1
                env.revealed[i] = False
            else:
                pid = int(tok - 2)
                env.piece[i] = pid
                env.revealed[i] = True

        tpc = int(sample.get("to_play_color", 2))
        tpp = int(sample.get("to_play_player", 0))
        if tpc in (0, 1):
            env.player_color[tpp] = tpc
            env.player_color[1 - tpp] = 1 - tpc
        else:
            env.player_color = [None, None]

        env.to_play = tpp
        env.plies = int(sample.get("plies", 0))
        env.no_progress_plies = int(sample.get("no_progress_plies", 0))

        cap = sample.get("captured_counts", None)
        if cap is not None:
            captured: List[int] = []
            for pid in range(1, 15):
                captured.extend([pid] * int(cap[pid]))
            env.captured = captured
        else:
            env.captured = []

        h = sample.get("history_actions", None)
        if h is not None:
            env.history = [int(a) for a in np.asarray(h).tolist() if int(a) >= 0]
        else:
            env.history = []
        if env.max_history > 0 and len(env.history) > env.max_history:
            env.history = env.history[-env.max_history :]

        env.done = False
        env.winner = None
        env._update_terminal_fast()
        return env

    def reanalysis_step() -> None:
        nonlocal last_reanalysis
        if args.reanalysis_every <= 0:
            return
        if (step - last_reanalysis) < args.reanalysis_every:
            return
        if len(replay) < max(args.train_start, args.reanalysis_batch_size):
            return

        idx, batch_samples = replay.sample(args.reanalysis_batch_size)

        re_cfg = MCTSConfig(
            num_simulations=args.reanalysis_simulations,
            c_puct=args.c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            temperature=1.0,
            num_determinize=args.reanalysis_determinize,
            batch_leaves=args.batch_leaves,
        )

        def pv_fn(obs_list):
            B = len(obs_list)
            boards = np.stack([o.board_tokens for o in obs_list]).astype(np.int64)
            beliefs = np.stack([o.belief for o in obs_list]).astype(np.float32)
            tpc = np.asarray([o.to_play_color for o in obs_list], dtype=np.int64)
            pl = np.asarray([o.plies for o in obs_list], dtype=np.int64)
            npg = np.asarray([o.no_progress_plies for o in obs_list], dtype=np.int64)
            legal = np.stack([o.action_mask for o in obs_list]).astype(np.bool_)

            H = args.max_history
            hist = np.full((B, H), -1, dtype=np.int64)
            for i, o in enumerate(obs_list):
                h = o.history_actions[-H:]
                if len(h) > 0:
                    hist[i, -len(h) :] = h

            board_t = torch.from_numpy(boards).to(device=device, dtype=torch.long)
            belief_t = torch.from_numpy(beliefs).to(device=device, dtype=torch.float32)
            hist_t = torch.from_numpy(hist).to(device=device, dtype=torch.long)
            tpc_t = torch.from_numpy(tpc).to(device=device, dtype=torch.long)
            pl_t = torch.from_numpy(pl).to(device=device, dtype=torch.long)
            npg_t = torch.from_numpy(npg).to(device=device, dtype=torch.long)
            legal_t = torch.from_numpy(legal).to(device=device, dtype=torch.bool)

            was_training = model.training
            model.eval()
            with torch.inference_mode():
                with autocast(
                    device_type=amp_device,
                    dtype=torch.bfloat16,
                    enabled=(device.type == "cuda"),
                ):
                    out = model(
                        board_t,
                        belief_t,
                        hist_t,
                        tpc_t,
                        pl_t,
                        npg_t,
                        belief_mask=None,
                    )
                logits = out.policy_self_logits.float()
                masked = logits.masked_fill(~legal_t, -1e9)
                pri = F.softmax(masked, dim=-1).cpu().numpy().astype(np.float32)
                val = out.value.float().cpu().numpy().astype(np.float32)
            if was_training:
                model.train()
            return pri, val

        for s in batch_samples:
            try:
                env = sample_to_env(s)
            except Exception:
                continue
            obs = env.observe(max_history=args.max_history)
            _pi_play, visits, v_root = pimc_mcts_policy(
                env,
                policy_value_fn=pv_fn,
                cfg=re_cfg,
                max_history=args.max_history,
            )
            s["pi_self"] = visits_to_pi(
                visits,
                obs.action_mask.astype(np.bool_),
                temperature=float(args.pi_train_temperature),
            )
            s["v_root"] = float(v_root)

        last_reanalysis = step

    while step < args.total_steps:
        for _ in range(args.replay_drain_max):
            try:
                samples = replay_q.get_nowait()
            except queue.Empty:
                break
            replay.add_many(samples)

        if len(replay) < args.train_start:
            time.sleep(0.001)
            continue

        # occasional reanalysis
        reanalysis_step()

        step += 1
        lr = cosine_schedule(step, args.total_steps, args.lr, warmup=args.warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        pbar.update(1)

        idx, batch_samples = replay.sample(args.batch_size)
        batch = collate_batch(batch_samples, max_history=args.max_history)

        board = batch["board"].to(device, non_blocking=True)
        belief = batch["belief"].to(device, non_blocking=True)
        history = batch["history"].to(device, non_blocking=True)
        to_play_color = batch["to_play_color"].to(device, non_blocking=True)
        plies = batch["plies"].to(device, non_blocking=True)
        no_progress_plies = batch["no_progress_plies"].to(device, non_blocking=True)
        action_mask = batch["action_mask"].to(device, non_blocking=True)
        pi_self = batch["pi_self"].to(device, non_blocking=True)
        value_z = batch["value"].to(device, non_blocking=True)
        v_root = batch["v_root"].to(device, non_blocking=True)
        belief_target = batch["belief_target"].to(device, non_blocking=True)

        # mixed value target
        mix = float(getattr(args, "value_target_mix", 0.0))
        if mix > 0:
            target_value = (1.0 - mix) * value_z + mix * v_root
            target_value = target_value.clamp(-1.0, 1.0)
        else:
            target_value = value_z

        if args.belief_mask_prob > 0:
            covered = board == COVERED_TOK
            rand = torch.rand_like(board.float())
            bmask = covered & (rand < args.belief_mask_prob)
        else:
            bmask = None

        optimizer.zero_grad(set_to_none=True)
        with autocast(
            device_type=amp_device,
            dtype=amp_dtype,
            enabled=(args.bf16 and device.type == "cuda"),
        ):
            out = model(
                board,
                belief,
                history,
                to_play_color,
                plies,
                no_progress_plies,
                belief_mask=bmask,
            )
            loss, metrics = criterion(
                out=out,
                target_self_policy=pi_self,
                target_value=target_value,
                target_belief=belief_target,
                legal_action_mask=action_mask,
                belief_loss_mask=(
                    bmask
                    if (bmask is not None and args.belief_loss_only_masked)
                    else None
                ),
            )

        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        writer.add_scalar("train/loss_total", metrics["total_loss"], step)
        writer.add_scalar("replay/size", len(replay), step)

        # update PER priorities
        if replay.prioritized:
            with torch.no_grad():
                v_err = (out.value.detach() - target_value).abs()
                logp = BanqiTransformer.masked_log_softmax(
                    out.policy_self_logits.detach(), action_mask, dim=-1
                )
                t = pi_self * action_mask.to(dtype=pi_self.dtype)
                t = t / t.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                ce_per = -(t * logp).sum(dim=-1)
                pr = (
                    (
                        float(args.per_value_coef) * v_err
                        + float(args.per_policy_coef) * ce_per
                    )
                    .cpu()
                    .numpy()
                )
                replay.update_priorities(idx, pr + float(args.per_eps))

        if step % args.log_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            pbar.set_postfix(
                {
                    "loss": f"{metrics['total_loss']:.4f}",
                    "lr": f"{lr:.2e}",
                    "replay": len(replay),
                }
            )
            print(
                f"[learner step {step}] replay={len(replay)} loss={metrics['total_loss']:.4f} lr={lr:.2e} ({args.log_every / dt:.1f} steps/s)"
            )

        if args.sync_every > 0 and (step - last_sync) >= args.sync_every:
            write_weights(step)
            last_sync = step

        if step % args.save_every == 0:
            save_checkpoint(
                save_dir / f"ckpt_step{step}",
                model,
                optimizer,
                step,
                cfg={"model_cfg": model_cfg.__dict__, "loss_cfg": loss_cfg.__dict__},
            )

    pbar.close()
    write_weights(step)
    stop_event.set()
    writer.flush()
    writer.close()

    save_checkpoint(
        save_dir / "ckpt_final",
        model,
        optimizer,
        step,
        cfg={"model_cfg": model_cfg.__dict__, "loss_cfg": loss_cfg.__dict__},
    )
    print("[learner] done")


# ----------------- Main entry -----------------
def az_split_main(args: argparse.Namespace) -> None:
    num_actors = int(args.num_actors)
    if num_actors <= 0:
        raise ValueError("--num_actors must be > 0")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    infer_req_q: mp.Queue = mp.Queue(maxsize=args.infer_queue_size)
    infer_resp_qs: List[mp.Queue] = [
        mp.Queue(maxsize=args.infer_resp_queue_size) for _ in range(num_actors)
    ]
    replay_q: mp.Queue = mp.Queue(maxsize=args.replay_queue_size)
    weight_q: mp.Queue = mp.Queue(maxsize=64)
    stop_event: MpEvent = mp.Event()

    infer = mp.Process(
        target=inference_server_process,
        args=(args, infer_req_q, infer_resp_qs, weight_q, stop_event),
        daemon=True,
    )
    learner = mp.Process(
        target=learner_process,
        args=(args, replay_q, weight_q, stop_event),
        daemon=True,
    )
    infer.start()
    learner.start()

    actors: List[mp.Process] = []
    for i in range(num_actors):
        p = mp.Process(
            target=actor_process,
            args=(i, args, infer_req_q, infer_resp_qs[i], replay_q, stop_event),
            daemon=True,
        )
        p.start()
        actors.append(p)

    try:
        learner.join()
    except KeyboardInterrupt:
        print("\n[main] KeyboardInterrupt: stopping...")
        stop_event.set()

    for p in actors:
        p.join(timeout=2.0)
    for p in actors:
        if p.is_alive():
            p.terminate()

    try:
        infer_req_q.put({"type": "shutdown"})
    except Exception:
        pass
    infer.join(timeout=5.0)
    if infer.is_alive():
        infer.terminate()
