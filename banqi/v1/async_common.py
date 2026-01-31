"""Shared async components for Banqi.

This module consolidates logic used by both async_az and async_az_split:
- InferenceClient: synchronous RPC client for batched inference
- ReplayBuffer: ring buffer with optional prioritized replay
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any, Dict, List, Tuple

import numpy as np


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
            if s <= 0 or not np.isfinite(s):
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
