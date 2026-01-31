"""
Action space definition for Chinese Dark Chess (Banqi) on a 4x8 board (Taiwan rules).

We use a fixed, enumerated action space so the policy head can be a simple linear layer.

Actions:
  - FLIP: choose a square to flip (reveal a face-down piece): (kind="flip", src=i, dst=i)
  - MOVE/CAPTURE: move from src->dst if src and dst are in same row or same column (directed pair)
    * For most pieces, only adjacent dst will be legal.
    * For cannon, long-range captures (with exactly one intervening piece) are legal along row/col.
The legality is enforced by the environment via an action mask.

This file contains only enumeration + helpers; game rules live in env.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Action:
    kind: str  # "flip" or "move"
    src: int
    dst: int


def rc_to_i(r: int, c: int, rows: int = 4, cols: int = 8) -> int:
    return r * cols + c


def i_to_rc(i: int, rows: int = 4, cols: int = 8) -> Tuple[int, int]:
    return divmod(i, cols)


class ActionSpace:
    """
    Fixed action space for 4x8 Banqi.

    Total actions:
      flips: 32
      moves: for each square, to any other square in same row (7) or same col (3) = 10, so 32*10 = 320
      total = 352
    """

    def __init__(self, rows: int = 4, cols: int = 8) -> None:
        self.rows = rows
        self.cols = cols
        self.num_squares = rows * cols
        self.actions: List[Action] = []
        self._action_to_id: Dict[Tuple[str, int, int], int] = {}

        # 1) flips
        for i in range(self.num_squares):
            self._add(Action("flip", i, i))

        # 2) row/col directed pairs (i != j)
        for src in range(self.num_squares):
            r, c = i_to_rc(src, rows, cols)
            # same row
            for cc in range(cols):
                if cc == c:
                    continue
                dst = rc_to_i(r, cc, rows, cols)
                self._add(Action("move", src, dst))
            # same col
            for rr in range(rows):
                if rr == r:
                    continue
                dst = rc_to_i(rr, c, rows, cols)
                self._add(Action("move", src, dst))

        self.n_actions = len(self.actions)

        # Precompute fast lookup for move action ids: move_id[src, dst] -> action_id
        self.move_id = np.full((self.num_squares, self.num_squares), -1, dtype=np.int32)
        for aid, a in enumerate(self.actions):
            if a.kind == "move":
                self.move_id[a.src, a.dst] = aid

    def _add(self, a: Action) -> None:
        key = (a.kind, a.src, a.dst)
        if key in self._action_to_id:
            return
        self._action_to_id[key] = len(self.actions)
        self.actions.append(a)

    def encode(self, kind: str, src: int, dst: int) -> int:
        return self._action_to_id[(kind, src, dst)]

    def decode(self, action_id: int) -> Action:
        return self.actions[action_id]

    def is_flip(self, action_id: int) -> bool:
        return self.actions[action_id].kind == "flip"

    def mirror_square_index(self, i: int) -> int:
        r, c = i_to_rc(i, self.rows, self.cols)
        mc = self.cols - 1 - c
        return rc_to_i(r, mc, self.rows, self.cols)

    def mirror_action_id(self, action_id: int) -> int:
        a = self.actions[action_id]
        src_m = self.mirror_square_index(a.src)
        dst_m = self.mirror_square_index(a.dst)
        return self.encode(a.kind, src_m, dst_m)

    def __len__(self) -> int:
        return self.n_actions
