"""banqi.v1.async_az

Actor / Inference / Learner separation for Banqi:
  - Actors (CPU, multi-process): run self-play games + root-sampling ISMCTS, requesting
    batched NN inference.
  - Inference server (GPU): batches inference requests across all actors.
  - Learner (GPU): consumes self-play data, trains the model.

This "integrated" variant keeps inference+learning in a single GPU process (the learner
server), time-slicing between:
  1) Serving inference requests (high priority, batched)
  2) Training steps (fairness-controlled so training still progresses)

Compared to the original implementation, this version applies two major upgrades:

Imperfect-information search upgrade
    - MCTS core uses an *information-set* hash so statistics share across
      determinizations (root-sampling ISMCTS), rather than building one independent tree
      per determinization.

Training/pipeline upgrades
    - Separate play-temperature from training-temperature (pi_play vs pi_train)
    - Bootstrapped value target mixing (z with v_root)
    - Optional lagged/EMA inference weights for stability
    - Optional reanalysis (recompute pi/v on replay with newest net)
    - Optional prioritized replay

Usage (via train.py):
  uv run -m banqi.v1.train --mode az --device cuda --num_actors 16 --total_steps 20000 --bf16
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
from .env import COVERED_TOK, EMPTY_TOK, BanqiEnv
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

# ----------------- IPC messages -----------------
# Request:
#   {"type":"infer", "actor_id":int, "req_id":int,
#    "board": np.int64 [B,32],
#    "belief": np.float32 [B,32,15],
#    "history": np.int64 [B,H],
#    "to_play_color": np.int64 [B],
#    "legal": np.bool_ [B,A]}
#
# Response:
#   {"req_id":int, "priors": np.float32 [B,A], "values": np.float32 [B]}
#
# Replay:
#   {"type":"samples", "actor_id":int, "samples": List[dict]}
#
# Control:
#   {"type":"actor_done", "actor_id":int, "games":int, "samples":int}


class InferenceClient:
    """Synchronous RPC client used by an actor to request batched inference."""

    def __init__(
        self,
        actor_id: int,
        req_q: mp.Queue,
        resp_q: mp.Queue,
        max_history: int,
        timing_every: int = 0,
    ) -> None:
        self.actor_id = int(actor_id)
        self.req_q = req_q
        self.resp_q = resp_q
        self.max_history = int(max_history)
        self._timing_every = int(timing_every)
        self._timing_count = 0
        self._req_id = 0
        # in case out-of-order responses happen (shouldn't for single outstanding request)
        self._stash: Dict[int, Dict[str, Any]] = {}

    def policy_value_batch(self, obs_list: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Callback for MCTS core: obs_list -> (priors [B,A], values [B])."""
        B = len(obs_list)
        if B == 0:
            raise ValueError("policy_value_batch called with empty obs_list")

        board = np.stack([o.board_tokens for o in obs_list]).astype(
            np.int64, copy=False
        )
        belief = np.stack([o.belief for o in obs_list]).astype(np.float32, copy=False)
        to_play_color = np.asarray([o.to_play_color for o in obs_list], dtype=np.int64)
        plies = np.asarray([o.plies for o in obs_list], dtype=np.int64)
        no_progress_plies = np.asarray(
            [o.no_progress_plies for o in obs_list], dtype=np.int64
        )
        legal = np.stack([o.action_mask for o in obs_list]).astype(np.bool_, copy=False)

        # history pad/right-align to max_history
        H = self.max_history
        hist = np.full((B, H), -1, dtype=np.int64)
        if H > 0:
            for i, o in enumerate(obs_list):
                h = o.history_actions
                if h is None:
                    continue
                h = h[-H:]
                if len(h) > 0:
                    hist[i, -len(h) :] = h

        self._req_id += 1
        rid = self._req_id

        msg = {
            "type": "infer",
            "actor_id": self.actor_id,
            "req_id": rid,
            "board": board,
            "belief": belief,
            "history": hist,
            "to_play_color": to_play_color,
            "plies": plies,
            "no_progress_plies": no_progress_plies,
            "legal": legal,
        }
        self._timing_count += 1
        t_send = time.perf_counter() if self._timing_every > 0 else 0.0

        self.req_q.put(msg)

        # wait response
        if rid in self._stash:
            resp = self._stash.pop(rid)
        else:
            while True:
                resp = self.resp_q.get()
                if resp.get("req_id") == rid:
                    break
                self._stash[int(resp["req_id"])] = resp

        if self._timing_every > 0 and (self._timing_count % self._timing_every == 0):
            t_recv = time.perf_counter()
            print(f"[actor {self.actor_id}] infer_wait={t_recv - t_send:.3f}s B={B}")

        priors = resp["priors"].astype(np.float32, copy=False)
        values = resp["values"].astype(np.float32, copy=False)
        return priors, values


# ----------------- Replay buffer -----------------
class ReplayBuffer:
    """Simple ring-buffer of python dict samples.

    Optionally supports prioritized replay (PER) with proportional sampling.
    """

    def __init__(
        self,
        capacity: int,
        prioritized: bool = False,
        alpha: float = 0.6,
        eps: float = 1e-3,
    ) -> None:
        self.capacity = int(capacity)
        self.data: List[Dict[str, Any]] = []
        self.pos = 0

        self.prioritized = bool(prioritized)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.priorities: List[float] = []
        self.max_priority = 1.0

    def __len__(self) -> int:
        return len(self.data)

    def add_many(self, samples: List[Dict[str, Any]]) -> None:
        for s in samples:
            if len(self.data) < self.capacity:
                self.data.append(s)
                if self.prioritized:
                    self.priorities.append(self.max_priority)
            else:
                self.data[self.pos] = s
                if self.prioritized:
                    self.priorities[self.pos] = self.max_priority
                self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if len(self.data) < batch_size:
            raise ValueError(
                f"Not enough samples: have {len(self.data)}, need {batch_size}"
            )

        n = len(self.data)
        if not self.prioritized:
            idx = np.random.randint(0, n, size=(batch_size,), dtype=np.int64)
        else:
            p = np.asarray(self.priorities, dtype=np.float64)
            # numerical stability
            p = np.maximum(p, self.eps) ** self.alpha
            s = float(p.sum())
            if s <= 0:
                idx = np.random.randint(0, n, size=(batch_size,), dtype=np.int64)
            else:
                p = p / s
                idx = np.random.choice(n, size=(batch_size,), replace=True, p=p).astype(
                    np.int64
                )

        return idx, [self.data[int(i)] for i in idx]

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        if not self.prioritized:
            return
        if len(indices) != len(priorities):
            raise ValueError("indices/priorities length mismatch")
        for i, pr in zip(indices, priorities):
            ii = int(i)
            p = float(max(self.eps, pr))
            self.priorities[ii] = p
            if p > self.max_priority:
                self.max_priority = p


def collate_batch(
    batch: List[Dict[str, Any]],
    action_dim: int,
    max_history: int,
):
    """Collate to torch tensors."""
    B = len(batch)
    board = torch.tensor(np.stack([b["board_tokens"] for b in batch]), dtype=torch.long)
    belief = torch.tensor(np.stack([b["belief"] for b in batch]), dtype=torch.float32)
    to_play_color = torch.tensor([b["to_play_color"] for b in batch], dtype=torch.long)
    plies = torch.tensor([b.get("plies", 0) for b in batch], dtype=torch.long)
    no_progress_plies = torch.tensor(
        [b.get("no_progress_plies", 0) for b in batch], dtype=torch.long
    )
    action_mask = torch.tensor(
        np.stack([b["action_mask"] for b in batch]), dtype=torch.bool
    )
    pi_self = torch.tensor(np.stack([b["pi_self"] for b in batch]), dtype=torch.float32)
    value = torch.tensor([b["value"] for b in batch], dtype=torch.float32)
    belief_t = torch.tensor(
        np.stack([b.get("belief_target", b["belief"]) for b in batch]),
        dtype=torch.float32,
    )
    v_root = torch.tensor([b.get("v_root", 0.0) for b in batch], dtype=torch.float32)

    hist = np.full((B, max_history), -1, dtype=np.int64)
    for i, b in enumerate(batch):
        h = b["history_actions"]
        h = h[-max_history:] if max_history > 0 else []
        if len(h) > 0:
            hist[i, -len(h) :] = h
    history = torch.tensor(hist, dtype=torch.long)

    return {
        "board": board,
        "belief": belief,
        "history": history,
        "to_play_color": to_play_color,
        "plies": plies,
        "no_progress_plies": no_progress_plies,
        "action_mask": action_mask,
        "pi_self": pi_self,
        "value": value,
        "belief_target": belief_t,
        "v_root": v_root,
    }


# ----------------- Actor process -----------------
def actor_process(
    actor_id: int,
    args: argparse.Namespace,
    infer_req_q: mp.Queue,
    infer_resp_q: mp.Queue,
    replay_q: mp.Queue,
    control_q: mp.Queue,
    stop_event: MpEvent,
) -> None:
    """CPU-only actor: runs self-play games and sends samples to replay_q."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    seed = int(args.seed) + 1000 * int(actor_id)
    set_seed(seed)

    action_space = ActionSpace()
    client = InferenceClient(
        actor_id=actor_id,
        req_q=infer_req_q,
        resp_q=infer_resp_q,
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
    total_samples = 0

    pbar = tqdm(
        desc=f"actor-{actor_id}",
        unit="game",
        dynamic_ncols=True,
        position=int(actor_id),
        leave=False,
    )

    while not stop_event.is_set():
        if args.games_per_actor > 0 and games >= args.games_per_actor:
            break

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
            # play temperature schedule (used for action selection)
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

            # current player's policy via root-sampling ISMCTS
            pi_play, visits, v_root = reuse_ctx.policy(
                env, policy_value_fn=client.policy_value_batch
            )
            if timing_enabled:
                t2 = time.perf_counter()

            # training target (pi_train) uses a separate temperature
            pi_self = visits_to_pi(
                visits, obs.action_mask.astype(np.bool_), args.pi_train_temperature
            )

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

            # captured piece counts (public)
            captured_counts = np.zeros(15, dtype=np.int16)
            if getattr(env, "captured", None) is not None:
                for pid in env.captured:
                    if 0 <= int(pid) < 15:
                        captured_counts[int(pid)] += 1

            sample = {
                "board_tokens": obs.board_tokens.astype(np.int64),
                "belief": obs.belief.astype(np.float32),
                "history_actions": obs.history_actions.astype(np.int64),
                "to_play_color": int(obs.to_play_color),
                "plies": int(obs.plies),
                "no_progress_plies": int(obs.no_progress_plies),
                "to_play_player": int(env.to_play),
                "action_mask": obs.action_mask.astype(np.bool_),
                "pi_self": pi_self.astype(np.float32),
                "value": 0.0,  # filled at end
                "belief_target": obs.belief.astype(np.float32),
                "v_root": float(v_root),
                "captured_counts": captured_counts,
            }
            samples.append(sample)

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

        # game result -> value targets
        if env.winner is None:
            result = {0: 0.0, 1: 0.0}
        else:
            result = {env.winner: 1.0, 1 - env.winner: -1.0}

        for s in samples:
            pid = int(s["to_play_player"])
            s["value"] = float(result[pid])

        if mirror_maps is not None:
            square_map, action_map, inv_action_map = mirror_maps
            samples.extend(
                [
                    mirror_sample(s, square_map, action_map, inv_action_map)
                    for s in samples
                ]
            )

        if samples:
            replay_q.put({"type": "samples", "actor_id": actor_id, "samples": samples})
            total_samples += len(samples)

        games += 1
        pbar.update(1)
        if args.actor_log_every > 0 and (games % args.actor_log_every == 0):
            control_q.put(
                {
                    "type": "actor_stats",
                    "actor_id": actor_id,
                    "games": games,
                    "samples": total_samples,
                }
            )

    control_q.put(
        {
            "type": "actor_done",
            "actor_id": actor_id,
            "games": games,
            "samples": total_samples,
        }
    )
    pbar.close()


# ----------------- Learner / inference server process -----------------
def learner_server_process(
    args: argparse.Namespace,
    infer_req_q: mp.Queue,
    infer_resp_qs: List[mp.Queue],
    replay_q: mp.Queue,
    control_q: mp.Queue,
    stop_event: MpEvent,
) -> None:
    """Owns the GPU model and serves both:
    - batched inference for actors
    - training updates from replay buffer
    """

    class _DummyWriter:
        def add_scalar(self, *args, **kwargs):
            return None

        def add_text(self, *args, **kwargs):
            return None

        def add_hparams(self, *args, **kwargs):
            return None

        def flush(self):
            return None

        def close(self):
            return None

    set_seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    action_space = ActionSpace()
    action_dim = len(action_space)

    # ----------------- Models -----------------
    if args.checkpoint and Path(args.checkpoint).exists():
        model_raw, start_step = load_checkpoint(Path(args.checkpoint))
        model_raw = model_raw.to(device)
        model_cfg = model_raw.cfg
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
        model_raw = BanqiTransformer(model_cfg).to(device)
        start_step = 0

    print(f"[learner] model params: {model_raw.parameter_count() / 1e6:.2f}M")

    # training model (may be compiled)
    model_train = model_raw

    # inference model (lagged/EMA for stability)
    model_infer = BanqiTransformer(model_cfg).to(device)
    model_infer.load_state_dict(model_raw.state_dict(), strict=True)
    model_infer.eval()

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
            model_raw.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            fused=(device.type == "cuda"),
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            model_raw.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    # TensorBoard
    tb_root = Path(args.tb_dir) if args.tb_dir else (Path(args.save_dir) / "tb")
    tb_root.mkdir(parents=True, exist_ok=True)
    base_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = tb_root / f"{base_name}_az"
    suffix = 1
    while run_dir.exists():
        run_dir = tb_root / f"{base_name}_az_{suffix}"
        suffix += 1
    writer = (
        SummaryWriter(log_dir=str(run_dir))
        if SummaryWriter is not None
        else _DummyWriter()
    )
    writer.add_text("run/dir", str(run_dir))
    writer.add_text("run/save_dir", str(args.save_dir))

    if args.checkpoint and Path(args.checkpoint).exists():
        load_optimizer_state(Path(args.checkpoint), optimizer)
        print(f"[learner] loaded checkpoint {args.checkpoint} at step {start_step}")

    # Optional compile (train model only); keep model_raw for checkpoint IO.
    if args.compile:
        try:
            model_train = torch.compile(model_train, mode=args.compile_mode)
        except Exception as e:
            print(f"[learner] torch.compile failed, continuing without compile: {e}")
            model_train = model_raw

    # Replay
    replay = ReplayBuffer(
        capacity=args.replay_size,
        prioritized=bool(getattr(args, "prioritized_replay", False)),
        alpha=float(getattr(args, "per_alpha", 0.6)),
        eps=float(getattr(args, "per_eps", 1e-3)),
    )

    # Inference batching bookkeeping
    infer_backlog: List[Dict[str, Any]] = []
    infer_batches_since_train = 0

    done_actors: set[int] = set()
    t0 = time.time()

    model_raw.train()

    autocast = torch.autocast
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float32

    # (⑤) inference weight sync
    def sync_infer_weights() -> None:
        decay = float(getattr(args, "infer_ema_decay", 0.0))
        if decay > 0.0:
            with torch.no_grad():
                for p_inf, p_tr in zip(
                    model_infer.parameters(), model_raw.parameters()
                ):
                    p_inf.data.mul_(decay).add_(p_tr.data, alpha=(1.0 - decay))
        else:
            model_infer.load_state_dict(model_raw.state_dict(), strict=True)
        model_infer.eval()

    # ----------------- Inference serving -----------------
    def serve_one_infer_batch() -> bool:
        nonlocal infer_batches_since_train

        reqs: List[Dict[str, Any]] = []
        total_obs = 0

        def get_first_req() -> Optional[Dict[str, Any]]:
            if infer_backlog:
                return infer_backlog.pop(0)
            try:
                return infer_req_q.get(timeout=args.infer_poll_timeout_ms / 1000.0)
            except queue.Empty:
                return None

        first = get_first_req()
        if first is None:
            return False

        reqs.append(first)
        total_obs += int(first["board"].shape[0])

        t_start = time.time()
        while total_obs < args.infer_max_batch:
            if (time.time() - t_start) * 1000.0 >= args.infer_batch_window_ms:
                break
            try:
                req = infer_req_q.get_nowait()
            except queue.Empty:
                break
            n = int(req["board"].shape[0])
            if total_obs + n > args.infer_max_batch and len(reqs) > 0:
                infer_backlog.append(req)
                break
            reqs.append(req)
            total_obs += n

        boards = np.concatenate([r["board"] for r in reqs], axis=0)
        beliefs = np.concatenate([r["belief"] for r in reqs], axis=0)
        hists = np.concatenate([r["history"] for r in reqs], axis=0)
        to_play_color = np.concatenate([r["to_play_color"] for r in reqs], axis=0)
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
        tpc_t = torch.from_numpy(to_play_color).to(
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
                device_type=amp_device,
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                out = model_infer(
                    board_t,
                    belief_t,
                    hist_t,
                    tpc_t,
                    plies_t,
                    no_prog_t,
                    belief_mask=None,
                )
            logits = out.policy_self_logits.float()
            masked_logits = logits.masked_fill(~legal_t, -1e9)
            priors = F.softmax(masked_logits, dim=-1).cpu().numpy().astype(np.float32)
            values = out.value.float().cpu().numpy().astype(np.float32)

        offset = 0
        for r in reqs:
            bsz = int(r["board"].shape[0])
            p = priors[offset : offset + bsz]
            v = values[offset : offset + bsz]
            offset += bsz
            actor_id = int(r["actor_id"])
            infer_resp_qs[actor_id].put(
                {"req_id": int(r["req_id"]), "priors": p, "values": v}
            )

        infer_batches_since_train += 1
        return True

    # ----------------- Queues ingress -----------------
    def drain_replay(max_items: int = 32) -> int:
        n = 0
        for _ in range(max_items):
            try:
                msg = replay_q.get_nowait()
            except queue.Empty:
                break
            if msg is None:
                continue
            if msg.get("type") == "samples":
                replay.add_many(msg["samples"])
                n += 1
        return n

    def drain_control(max_items: int = 128) -> None:
        nonlocal done_actors
        for _ in range(max_items):
            try:
                msg = control_q.get_nowait()
            except queue.Empty:
                break
            t = msg.get("type")
            if t == "actor_done":
                done_actors.add(int(msg["actor_id"]))
            elif t == "actor_stats":
                pass

    # ----------------- Reanalysis (⑤) -----------------
    def _env_from_sample(sample: Dict[str, Any]) -> BanqiEnv:
        """Reconstruct a BanqiEnv that matches the sample's *public* state.

        We set covered squares to an arbitrary non-zero pid (occupied+unrevealed) and
        reconstruct captured counts from sample['captured_counts'] if available.
        """
        env = BanqiEnv(action_space=action_space, max_history=args.max_history)

        bt = sample["board_tokens"]
        piece = [0] * action_space.num_squares
        revealed = [False] * action_space.num_squares
        for i in range(action_space.num_squares):
            tok = int(bt[i])
            if tok == EMPTY_TOK:
                piece[i] = 0
                revealed[i] = False
            elif tok == COVERED_TOK:
                piece[i] = 1  # placeholder pid
                revealed[i] = False
            else:
                pid = tok - 2
                piece[i] = int(pid)
                revealed[i] = True

        env.piece = piece
        env.revealed = revealed

        # to_play + color assignment
        to_play_player = int(sample.get("to_play_player", 0))
        env.to_play = to_play_player
        tpc = int(sample.get("to_play_color", 2))
        if tpc in (0, 1):
            env.player_color[to_play_player] = tpc
            env.player_color[1 - to_play_player] = 1 - tpc
        else:
            env.player_color = [None, None]

        env.plies = int(sample.get("plies", 0))
        env.no_progress_plies = int(sample.get("no_progress_plies", 0))

        # history
        h = sample.get("history_actions", None)
        if h is not None:
            env.history = [int(a) for a in np.asarray(h).tolist() if int(a) >= 0]
        else:
            env.history = []
        if env.max_history > 0 and len(env.history) > env.max_history:
            env.history = env.history[-env.max_history :]

        # captured list
        captured_counts = sample.get("captured_counts", None)
        if captured_counts is not None:
            cc = np.asarray(captured_counts).astype(np.int64)
            cap: List[int] = []
            for pid in range(1, 15):
                n = int(cc[pid]) if pid < len(cc) else 0
                if n > 0:
                    cap.extend([pid] * n)
            env.captured = cap
        else:
            env.captured = []

        # terminal flags
        env.winner = None
        env.done = False
        env._update_terminal_fast()
        return env

    def reanalysis_step(step: int) -> None:
        if getattr(args, "reanalysis_every", 0) <= 0:
            return
        if step <= 0 or (step % int(args.reanalysis_every) != 0):
            return
        if len(replay) < max(int(args.train_start), int(args.reanalysis_batch_size)):
            return

        bs = int(args.reanalysis_batch_size)
        idxs, batch_samples = replay.sample(bs)

        # Build evaluator fn using model_infer
        def pv_fn(obs_list: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
            B = len(obs_list)
            board = np.stack([o.board_tokens for o in obs_list]).astype(
                np.int64, copy=False
            )
            belief = np.stack([o.belief for o in obs_list]).astype(
                np.float32, copy=False
            )
            to_play_color = np.asarray(
                [o.to_play_color for o in obs_list], dtype=np.int64
            )
            plies = np.asarray([o.plies for o in obs_list], dtype=np.int64)
            no_prog = np.asarray(
                [o.no_progress_plies for o in obs_list], dtype=np.int64
            )
            legal = np.stack([o.action_mask for o in obs_list]).astype(
                np.bool_, copy=False
            )

            H = int(args.max_history)
            hist = np.full((B, H), -1, dtype=np.int64)
            for i, o in enumerate(obs_list):
                h = o.history_actions
                h = h[-H:] if H > 0 else []
                if len(h) > 0:
                    hist[i, -len(h) :] = h

            board_t = torch.from_numpy(board).to(device=device, dtype=torch.long)
            belief_t = torch.from_numpy(belief).to(device=device, dtype=torch.float32)
            hist_t = torch.from_numpy(hist).to(device=device, dtype=torch.long)
            tpc_t = torch.from_numpy(to_play_color).to(device=device, dtype=torch.long)
            plies_t = torch.from_numpy(plies).to(device=device, dtype=torch.long)
            no_prog_t = torch.from_numpy(no_prog).to(device=device, dtype=torch.long)
            legal_t = torch.from_numpy(legal).to(device=device, dtype=torch.bool)

            with torch.inference_mode():
                with autocast(
                    device_type=amp_device,
                    dtype=torch.bfloat16,
                    enabled=(device.type == "cuda"),
                ):
                    out = model_infer(
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
            return priors, values

        # Reanalysis MCTS config (no root noise; teacher-like)
        cfg = MCTSConfig(
            num_simulations=int(args.reanalysis_simulations),
            c_puct=float(args.c_puct),
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            temperature=1.0,  # only used for pi_play; we build pi_train from visits
            num_determinize=int(args.reanalysis_determinize),
            batch_leaves=int(args.batch_leaves),
        )

        for s in batch_samples:
            env_s = _env_from_sample(s)
            obs_s = env_s.observe(max_history=int(args.max_history))
            # reuse core helper
            _pi_play, visits, v_root = pimc_mcts_policy(
                env_s,
                policy_value_fn=pv_fn,
                cfg=cfg,
                max_history=int(args.max_history),
            )
            pi_train = visits_to_pi(
                visits,
                obs_s.action_mask.astype(np.bool_),
                float(args.pi_train_temperature),
            )
            s["pi_self"] = pi_train.astype(np.float32)
            s["v_root"] = float(v_root)

        # After changing targets, bump priorities to encourage learning.
        if replay.prioritized:
            replay.update_priorities(
                idxs, np.full_like(idxs, replay.max_priority, dtype=np.float32)
            )

    # ----------------- Training -----------------
    def train_one_step(step: int) -> Tuple[float, Dict[str, float], float]:
        """One SGD step. Returns (loss, metrics, grad_norm)."""
        idxs, batch_samples = replay.sample(args.batch_size)
        batch = collate_batch(
            batch_samples, action_dim=action_dim, max_history=args.max_history
        )

        board = batch["board"].to(device, non_blocking=True)
        belief = batch["belief"].to(device, non_blocking=True)
        history = batch["history"].to(device, non_blocking=True)
        to_play_color = batch["to_play_color"].to(device, non_blocking=True)
        plies = batch["plies"].to(device, non_blocking=True)
        no_progress_plies = batch["no_progress_plies"].to(device, non_blocking=True)
        action_mask = batch["action_mask"].to(device, non_blocking=True)
        pi_self = batch["pi_self"].to(device, non_blocking=True)
        value_z = batch["value"].to(device, non_blocking=True)
        belief_target = batch["belief_target"].to(device, non_blocking=True)
        v_root = batch["v_root"].to(device, non_blocking=True)

        # (⑤) bootstrapped value target mixing
        mix = float(getattr(args, "value_target_mix", 0.0))
        if mix > 0.0:
            target_value = (1.0 - mix) * value_z + mix * v_root
            target_value = torch.clamp(target_value, -1.0, 1.0)
        else:
            target_value = value_z

        # Belief denoising: mask some belief tokens (only on covered squares)
        if args.belief_mask_prob > 0:
            covered = board == COVERED_TOK
            rand = torch.rand_like(board.float())
            bmask = covered & (rand < args.belief_mask_prob)
        else:
            bmask = None

        lr = cosine_schedule(step, args.total_steps, args.lr, warmup=args.warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            device_type=amp_device,
            dtype=amp_dtype,
            enabled=(args.bf16 and device.type == "cuda"),
        ):
            out = model_train(
                board_tokens=board,
                belief=belief,
                history_actions=history,
                to_play_color=to_play_color,
                plies=plies,
                no_progress_plies=no_progress_plies,
                belief_mask=bmask,
            )

            belief_loss_mask = (
                bmask if (bmask is not None and args.belief_loss_only_masked) else None
            )

            loss, metrics = criterion(
                out=out,
                target_self_policy=pi_self,
                target_value=target_value,
                target_belief=belief_target,
                legal_action_mask=action_mask,
                belief_loss_mask=belief_loss_mask,
            )

        loss.backward()
        if args.clip_grad > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model_raw.parameters(), args.clip_grad
            )
        else:
            total_norm = 0.0
            for p in model_raw.parameters():
                if p.grad is None:
                    continue
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            grad_norm = total_norm**0.5

        optimizer.step()

        # (⑤) PER priority update
        if replay.prioritized:
            with torch.no_grad():
                # value error
                v_err = (
                    out.value.detach().float() - target_value.detach().float()
                ).abs()

                # policy cross-entropy per sample
                logits = out.policy_self_logits.detach().float()
                logp = BanqiTransformer.masked_log_softmax(logits, action_mask, dim=-1)
                tgt = pi_self.detach().float()
                tgt = tgt * action_mask.float()
                tgt = tgt / tgt.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                ce_per = -(tgt * logp).sum(dim=-1)

                coef_v = float(getattr(args, "per_value_coef", 1.0))
                coef_pi = float(getattr(args, "per_policy_coef", 1.0))
                pr = coef_v * v_err + coef_pi * ce_per
                pr_np = pr.cpu().numpy().astype(np.float32) + float(replay.eps)
            replay.update_priorities(idxs, pr_np)

        return float(loss.detach().cpu()), metrics, float(grad_norm)

    # ----------------- Main loop -----------------
    pbar = tqdm(
        total=args.total_steps,
        initial=int(start_step),
        desc="learner",
        unit="step",
        dynamic_ncols=True,
    )

    step = int(start_step)
    last_sync_step = step
    last_queue_log = time.time()

    def _safe_qsize(q: mp.Queue) -> int:
        try:
            return int(q.qsize())
        except (NotImplementedError, AttributeError):
            return -1

    while True:
        now = time.time()
        if now - last_queue_log >= 60.0:
            last_queue_log = now
            infer_req = _safe_qsize(infer_req_q)
            infer_resp = [_safe_qsize(q) for q in infer_resp_qs]
            replay_size = _safe_qsize(replay_q)
            control_size = _safe_qsize(control_q)
            infer_resp_max = max(infer_resp) if infer_resp else -1
            infer_resp_sum = sum(x for x in infer_resp if x >= 0)
            infer_resp_unknown = any(x < 0 for x in infer_resp)
            print(
                "[queues] infer_req={} infer_resp_total={} infer_resp_max={} replay={} control={}".format(
                    infer_req,
                    -1 if infer_resp_unknown else infer_resp_sum,
                    infer_resp_max,
                    replay_size,
                    control_size,
                )
            )
        drain_control()
        drain_replay(max_items=args.replay_drain_max)

        served = 0
        for _ in range(args.max_infer_batches_per_loop):
            if serve_one_infer_batch():
                served += 1
            else:
                break

        # training fairness
        can_train = (len(replay) >= args.train_start) and (step < args.total_steps)
        should_train = can_train and (
            infer_batches_since_train >= args.infer_batches_per_train or served == 0
        )

        if should_train:
            step += 1
            loss_val, metrics, grad_norm = train_one_step(step)
            infer_batches_since_train = 0
            pbar.update(1)

            # (⑤) optionally sync inference weights
            sync_every = int(getattr(args, "infer_sync_every", 0))
            if sync_every > 0 and (step - last_sync_step) >= sync_every:
                sync_infer_weights()
                last_sync_step = step

            # (⑤) optional reanalysis
            reanalysis_step(step)

            # tensorboard
            writer.add_scalar("train/loss_total", metrics["total_loss"], step)
            writer.add_scalar("train/loss_policy_self", metrics["self_policy_ce"], step)
            writer.add_scalar("train/loss_value", metrics["value_loss"], step)
            writer.add_scalar(
                "train/entropy_self", metrics["self_policy_entropy"], step
            )
            writer.add_scalar("train/kl_self", metrics["self_policy_kl"], step)
            if "belief_ce" in metrics:
                writer.add_scalar("train/loss_belief", metrics["belief_ce"], step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
            writer.add_scalar("train/grad_norm", grad_norm, step)
            writer.add_scalar("replay/size", len(replay), step)

            if step % args.log_every == 0:
                dt = time.time() - t0
                t0 = time.time()
                steps_per_sec = args.log_every / max(1e-8, dt)
                pbar.set_postfix(
                    {
                        "loss": f"{metrics['total_loss']:.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        "replay": len(replay),
                    }
                )
                print(
                    f"[learner step {step}] replay={len(replay)} "
                    f"loss={metrics['total_loss']:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"{steps_per_sec:.1f} steps/s"
                )

            if step % args.save_every == 0:
                save_dir = Path(args.save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = save_dir / f"ckpt_step{step}"

                save_checkpoint(
                    ckpt_path,
                    model=model_raw,
                    optimizer=optimizer,
                    step=step,
                    cfg={
                        "model_cfg": model_cfg.__dict__,
                        "loss_cfg": loss_cfg.__dict__,
                    },
                )
                print(f"[learner] saved checkpoint: {ckpt_path}")

        # stop condition
        if step >= args.total_steps and not stop_event.is_set():
            stop_event.set()
            print("[learner] reached total_steps, stopping actors...")

        if stop_event.is_set():
            if len(done_actors) >= int(args.num_actors):
                drained_any = False
                for _ in range(8):
                    if serve_one_infer_batch():
                        drained_any = True
                    else:
                        break
                if not drained_any:
                    break

        if served == 0 and not should_train:
            time.sleep(0.001)

    pbar.close()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    final = save_dir / "ckpt_final"

    save_checkpoint(
        final,
        model=model_raw,
        optimizer=optimizer,
        step=step,
        cfg={"model_cfg": model_cfg.__dict__, "loss_cfg": loss_cfg.__dict__},
    )
    writer.flush()
    writer.close()
    print(f"[learner] done. saved: {final}")


# ----------------- Main entry -----------------
def az_main(args: argparse.Namespace) -> None:
    """Spawns:
    - 1 learner/inference server process (GPU)
    - N actor processes (CPU)
    """
    num_actors = int(args.num_actors)
    if num_actors <= 0:
        raise ValueError("--num_actors must be > 0")

    # Ensure spawn (safe for CUDA)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    infer_req_q: mp.Queue = mp.Queue(maxsize=args.infer_queue_size)
    infer_resp_qs: List[mp.Queue] = [
        mp.Queue(maxsize=args.infer_resp_queue_size) for _ in range(num_actors)
    ]
    replay_q: mp.Queue = mp.Queue(maxsize=args.replay_queue_size)
    control_q: mp.Queue = mp.Queue(maxsize=args.control_queue_size)
    stop_event: MpEvent = mp.Event()

    learner = mp.Process(
        target=learner_server_process,
        args=(args, infer_req_q, infer_resp_qs, replay_q, control_q, stop_event),
        daemon=True,
    )
    learner.start()

    actors: List[mp.Process] = []
    for i in range(num_actors):
        p = mp.Process(
            target=actor_process,
            args=(
                i,
                args,
                infer_req_q,
                infer_resp_qs[i],
                replay_q,
                control_q,
                stop_event,
            ),
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
        p.join(timeout=1.0)

    for p in actors:
        if p.is_alive():
            p.terminate()

    if learner.is_alive():
        learner.terminate()
