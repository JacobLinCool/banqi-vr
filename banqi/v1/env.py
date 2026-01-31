"""
Minimal Chinese Dark Chess (Banqi) environment compatible with AlphaZero-style self-play.

Rules source (Taiwan / ICGA CDC):
- 4x8 board, 32 pieces, all face-down at start.
- A move is either FLIP (reveal a face-down piece) or MOVE/CAPTURE with a revealed piece.
- Pieces move 1 step orthogonally (no diagonals). Cannons capture with exactly one intervening piece along row/col.
- Capture requires target is a revealed opponent piece (Taiwan common rule).
- Rank order: King(1) > Guard(2) > Minister(3) > Rook(4) > Knight(5) > Cannon(6) > Pawn(7).
  A piece can capture equal or lower rank (numerically >=), with exception:
    * King cannot capture Pawns
    * Pawns can capture King
- First move must be a flip. After the first flip, the flipper's color is assigned to that revealed piece.

This env is designed for research / self-play data generation. It is NOT optimized.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .actions import ActionSpace, i_to_rc, rc_to_i

# Colors
RED = 0
BLACK = 1
UNKNOWN_COLOR = 2

# Piece kinds (7)
KING = 0
GUARD = 1
MINISTER = 2
ROOK = 3
KNIGHT = 4
CANNON = 5
PAWN = 6

KIND_NAMES = ["K", "G", "M", "R", "N", "C", "P"]

# Rank table: smaller is stronger
RANK = {
    KING: 1,
    GUARD: 2,
    MINISTER: 3,
    ROOK: 4,
    KNIGHT: 5,
    CANNON: 6,
    PAWN: 7,
}

# Initial counts per kind
KIND_COUNTS = {
    KING: 1,
    GUARD: 2,
    MINISTER: 2,
    ROOK: 2,
    KNIGHT: 2,
    CANNON: 2,
    PAWN: 5,
}

_TOTAL_PER_COLOR = sum(KIND_COUNTS.values())

# Fast piece lookup tables (index by pid: 0..14; 0 is empty)
PID_COLOR = np.zeros(15, dtype=np.int8)
PID_KIND = np.zeros(15, dtype=np.int8)
PID_RANK = np.zeros(15, dtype=np.int8)
PID_IS_CANNON = np.zeros(15, dtype=np.bool_)
PID_IS_PAWN = np.zeros(15, dtype=np.bool_)
PID_IS_KING = np.zeros(15, dtype=np.bool_)
for _pid in range(1, 15):
    _pid0 = _pid - 1
    _color = _pid0 // 7
    _kind = _pid0 % 7
    PID_COLOR[_pid] = _color
    PID_KIND[_pid] = _kind
    PID_RANK[_pid] = RANK[_kind]
    PID_IS_CANNON[_pid] = _kind == CANNON
    PID_IS_PAWN[_pid] = _kind == PAWN
    PID_IS_KING[_pid] = _kind == KING


# Piece identity ids: 1..14
def piece_id(color: int, kind: int) -> int:
    return color * 7 + kind + 1


def decode_piece_id(pid: int) -> Tuple[int, int]:
    assert 1 <= pid <= 14
    pid0 = pid - 1
    return pid0 // 7, pid0 % 7


def is_cannon(pid: int) -> bool:
    return bool(PID_IS_CANNON[pid])


def is_pawn(pid: int) -> bool:
    return bool(PID_IS_PAWN[pid])


def is_king(pid: int) -> bool:
    return bool(PID_IS_KING[pid])


def piece_rank(pid: int) -> int:
    return int(PID_RANK[pid])


# Board tokenization for model input
PAD_TOK = 0
EMPTY_TOK = 1
COVERED_TOK = 2
# revealed piece token = 2 + pid (pid in 1..14) => 3..16


@dataclass
class Observation:
    board_tokens: np.ndarray  # [32] int64
    belief: np.ndarray  # [32, 15] float32 (empty + 14 pieces)
    history_actions: (
        np.ndarray
    )  # [H] int64 (action ids), padded with -1 if needed by caller
    to_play_color: int  # 0/1/2
    plies: int  # total plies elapsed
    no_progress_plies: int  # plies since last capture/flip
    action_mask: np.ndarray  # [A] bool (legal actions)


@dataclass
class _SearchUndo:
    kind: str
    src: int
    dst: int
    prev_piece_src: int
    prev_piece_dst: int
    prev_revealed_src: bool
    prev_revealed_dst: bool
    prev_player_color: Tuple[Optional[int], Optional[int]]
    prev_to_play: int
    prev_plies: int
    prev_no_progress_plies: int
    prev_done: bool
    prev_winner: Optional[int]
    prev_captured_len: int
    history_appended: bool
    history_dropped: List[int]


class BanqiEnv:
    def __init__(
        self,
        action_space: ActionSpace,
        draw_plies: int = 100,
        max_history: int = 16,
        seed: Optional[int] = None,
    ) -> None:
        self.action_space = action_space
        self.draw_plies = draw_plies
        self.max_history = int(max_history)
        self.rng = random.Random(seed)

        self.reset(seed=seed)

    # ---------- Game lifecycle ----------
    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng.seed(seed)

        # Create multiset of 32 piece ids
        pieces: List[int] = []
        for color in (RED, BLACK):
            for kind, n in KIND_COUNTS.items():
                pieces.extend([piece_id(color, kind)] * n)
        assert len(pieces) == self.action_space.num_squares

        self.rng.shuffle(pieces)

        self.piece = pieces[:]  # 0 means empty; at start all occupied
        self.revealed = [False] * self.action_space.num_squares

        self.player_color: List[Optional[int]] = [None, None]
        self.to_play: int = 0  # player id 0 starts
        self.plies: int = 0
        self.no_progress_plies: int = 0
        self.captured: List[
            int
        ] = []  # list of captured piece ids (identities are public because captures require revealed)
        self.winner: Optional[int] = None  # 0/1 or None
        self.done: bool = False

        # history for observation
        self.history: List[int] = []

    def clone(self) -> "BanqiEnv":
        env = object.__new__(BanqiEnv)

        # shared / immutable
        env.action_space = self.action_space
        env.draw_plies = self.draw_plies
        env.max_history = int(self.max_history)
        env.rng = self.rng

        # mutable state (copy)
        env.piece = self.piece[:]
        env.revealed = self.revealed[:]
        env.player_color = self.player_color[:]
        env.to_play = int(self.to_play)
        env.plies = int(self.plies)
        env.no_progress_plies = int(self.no_progress_plies)
        env.captured = self.captured[:]
        env.winner = self.winner
        env.done = bool(self.done)
        env.history = self.history[:]
        return env

    def _append_history(self, action_id: int) -> None:
        if self.max_history <= 0:
            return
        self.history.append(int(action_id))
        if len(self.history) > self.max_history:
            del self.history[: -self.max_history]

    def _append_history_with_undo(self, action_id: int) -> Tuple[bool, List[int]]:
        if self.max_history <= 0:
            return False, []
        self.history.append(int(action_id))
        dropped: List[int] = []
        if len(self.history) > self.max_history:
            excess = len(self.history) - self.max_history
            if excess > 0:
                dropped = self.history[:excess]
                del self.history[:excess]
        return True, dropped

    # ---------- Public information / belief ----------
    def _initial_counts(self) -> np.ndarray:
        # [15] counts (0=empty, 1..14 piece ids)
        counts = np.zeros(15, dtype=np.int32)
        for color in (RED, BLACK):
            for kind, n in KIND_COUNTS.items():
                counts[piece_id(color, kind)] += n
        return counts

    def public_remaining_hidden_counts(self) -> np.ndarray:
        """
        Returns counts of piece identities that are still face-down (unrevealed) on the board,
        computed only from public info: initial counts - revealed - captured.
        Shape: [15] int32 (0=empty, 1..14 piece ids)
        """
        counts = self._initial_counts()

        # subtract revealed on board
        for i in range(self.action_space.num_squares):
            if self.piece[i] != 0 and self.revealed[i]:
                counts[self.piece[i]] -= 1

        # subtract captured
        for pid in self.captured:
            counts[pid] -= 1

        # remaining hidden are those not revealed & not captured -> should equal number of covered squares
        return counts

    def belief_marginals(self) -> np.ndarray:
        """
        Returns per-square belief distribution over piece identities given public info.
        Output: [32, 15] float32 (0=empty, 1..14 piece ids)
        """
        n = self.action_space.num_squares
        belief = np.zeros((n, 15), dtype=np.float32)

        # Compute distribution for covered squares
        hidden_counts = self.public_remaining_hidden_counts().astype(np.float32)
        covered_squares = [
            i for i in range(n) if self.piece[i] != 0 and not self.revealed[i]
        ]
        n_cov = len(covered_squares)

        if n_cov > 0:
            probs = hidden_counts / float(n_cov)
            probs[0] = 0.0  # empty not possible for covered squares
        else:
            probs = np.zeros(15, dtype=np.float32)

        for i in range(n):
            if self.piece[i] == 0:
                belief[i, 0] = 1.0
            elif self.revealed[i]:
                belief[i, self.piece[i]] = 1.0
            else:
                belief[i, :] = probs

        return belief

    def board_tokens(self) -> np.ndarray:
        n = self.action_space.num_squares
        out = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if self.piece[i] == 0:
                out[i] = EMPTY_TOK
            elif not self.revealed[i]:
                out[i] = COVERED_TOK
            else:
                out[i] = 2 + int(self.piece[i])  # 3..16
        return out

    def to_play_color_token(self) -> int:
        c = self.player_color[self.to_play]
        return UNKNOWN_COLOR if c is None else int(c)

    # ---------- Determinization ----------
    def determinize(self, seed: Optional[int] = None) -> "BanqiEnv":
        """
        Returns a determinized clone by randomly assigning the remaining hidden pieces
        to the currently covered squares (uniform among permutations of the multiset).

        NOTE: This is a standard PIMC determinization baseline.
        """
        env = self.clone()
        rng = random.Random(seed) if seed is not None else self.rng

        hidden_counts = env.public_remaining_hidden_counts()
        covered = [
            i
            for i in range(env.action_space.num_squares)
            if env.piece[i] != 0 and not env.revealed[i]
        ]
        # Build the multiset
        pool: List[int] = []
        for pid in range(1, 15):
            pool.extend([pid] * int(hidden_counts[pid]))
        assert len(pool) == len(covered), (len(pool), len(covered))

        rng.shuffle(pool)
        for sq, pid in zip(covered, pool):
            env.piece[sq] = int(pid)

        return env

    # ---------- Move generation ----------
    def _adjacent(self, src: int, dst: int) -> bool:
        sr, sc = i_to_rc(src, self.action_space.rows, self.action_space.cols)
        dr, dc = i_to_rc(dst, self.action_space.rows, self.action_space.cols)
        return abs(sr - dr) + abs(sc - dc) == 1

    def _between_squares_rowcol(self, src: int, dst: int) -> List[int]:
        """
        Squares strictly between src and dst along row or column. Assumes same row or same col.
        """
        sr, sc = i_to_rc(src, self.action_space.rows, self.action_space.cols)
        dr, dc = i_to_rc(dst, self.action_space.rows, self.action_space.cols)
        between: List[int] = []
        if sr == dr:
            step = 1 if dc > sc else -1
            for c in range(sc + step, dc, step):
                between.append(
                    rc_to_i(sr, c, self.action_space.rows, self.action_space.cols)
                )
        elif sc == dc:
            step = 1 if dr > sr else -1
            for r in range(sr + step, dr, step):
                between.append(
                    rc_to_i(r, sc, self.action_space.rows, self.action_space.cols)
                )
        else:
            raise ValueError("src and dst not aligned")
        return between

    def _capture_allowed(self, attacker: int, defender: int) -> bool:
        """
        attacker/defender are piece ids 1..14 (both revealed).
        """
        # Cannon handled separately (requires jump)
        if is_cannon(attacker):
            return False
        a_rank = piece_rank(attacker)
        d_rank = piece_rank(defender)

        # Exceptions
        if is_king(attacker) and is_pawn(defender):
            return False
        if is_pawn(attacker) and is_king(defender):
            return True

        # General rule: can capture equal or lower rank => attacker is stronger => smaller rank number
        return a_rank <= d_rank

    def legal_action_mask(self) -> np.ndarray:
        """
        Boolean mask [A] for currently legal actions.
        """
        A = len(self.action_space)
        mask = np.zeros(A, dtype=np.bool_)

        if self.done:
            return mask

        move_id = self.action_space.move_id

        # flips: always allowed on covered squares
        for i in range(self.action_space.num_squares):
            if self.piece[i] != 0 and (not self.revealed[i]):
                # flip action ids are enumerated first (flip(i) == i)
                mask[i] = True

        # If colors not assigned yet, only flips are legal (first move must be flip).
        if self.player_color[0] is None and self.player_color[1] is None:
            return mask

        my_color = self.player_color[self.to_play]
        assert my_color is not None

        rows = self.action_space.rows
        cols = self.action_space.cols

        for src in range(self.action_space.num_squares):
            pid = self.piece[src]
            if pid == 0 or (not self.revealed[src]):
                continue

            if int(PID_COLOR[pid]) != my_color:
                continue

            sr, sc = i_to_rc(src, rows, cols)
            attacker_is_cannon = is_cannon(pid)

            # adjacent moves/captures
            if sr > 0:
                dst = rc_to_i(sr - 1, sc, rows, cols)
                self._add_adjacent_move_or_capture(
                    mask, move_id, src, dst, pid, my_color, attacker_is_cannon
                )
            if sr + 1 < rows:
                dst = rc_to_i(sr + 1, sc, rows, cols)
                self._add_adjacent_move_or_capture(
                    mask, move_id, src, dst, pid, my_color, attacker_is_cannon
                )
            if sc > 0:
                dst = rc_to_i(sr, sc - 1, rows, cols)
                self._add_adjacent_move_or_capture(
                    mask, move_id, src, dst, pid, my_color, attacker_is_cannon
                )
            if sc + 1 < cols:
                dst = rc_to_i(sr, sc + 1, rows, cols)
                self._add_adjacent_move_or_capture(
                    mask, move_id, src, dst, pid, my_color, attacker_is_cannon
                )

            # cannon captures (long-range, exactly one screen)
            if attacker_is_cannon:
                # up
                self._add_cannon_captures_dir(mask, move_id, src, -1, 0, my_color)
                # down
                self._add_cannon_captures_dir(mask, move_id, src, 1, 0, my_color)
                # left
                self._add_cannon_captures_dir(mask, move_id, src, 0, -1, my_color)
                # right
                self._add_cannon_captures_dir(mask, move_id, src, 0, 1, my_color)

        return mask

    def _add_adjacent_move_or_capture(
        self,
        mask: np.ndarray,
        move_id: np.ndarray,
        src: int,
        dst: int,
        attacker: int,
        my_color: int,
        attacker_is_cannon: bool,
    ) -> None:
        if self.piece[dst] == 0:
            aid = int(move_id[src, dst])
            if aid >= 0:
                mask[aid] = True
            return

        # capture: target must be revealed opponent
        if not self.revealed[dst]:
            return
        if int(PID_COLOR[self.piece[dst]]) == my_color:
            return

        if attacker_is_cannon:
            return

        if self._capture_allowed(attacker, self.piece[dst]):
            aid = int(move_id[src, dst])
            if aid >= 0:
                mask[aid] = True

    def _add_cannon_captures_dir(
        self,
        mask: np.ndarray,
        move_id: np.ndarray,
        src: int,
        dr: int,
        dc: int,
        my_color: int,
    ) -> None:
        rows = self.action_space.rows
        cols = self.action_space.cols
        r, c = i_to_rc(src, rows, cols)
        r += dr
        c += dc
        seen_screen = False

        while 0 <= r < rows and 0 <= c < cols:
            dst = rc_to_i(r, c, rows, cols)
            if self.piece[dst] != 0:
                if not seen_screen:
                    seen_screen = True
                else:
                    if self.revealed[dst]:
                        if int(PID_COLOR[self.piece[dst]]) != my_color:
                            aid = int(move_id[src, dst])
                            if aid >= 0:
                                mask[aid] = True
                    break
            r += dr
            c += dc

    def _has_any_legal_move(self) -> bool:
        return bool(self.legal_action_mask().any())

    def _remaining_by_color_from_captured(self) -> Tuple[int, int]:
        cap = [0, 0]
        for pid in self.captured:
            cap[int(PID_COLOR[pid])] += 1
        return (_TOTAL_PER_COLOR - cap[RED], _TOTAL_PER_COLOR - cap[BLACK])

    # ---------- Step ----------
    def step(self, action_id: int) -> Tuple[Observation, float, bool, Dict]:
        """
        Apply action for current player. Returns (obs, reward, done, info)
        Reward is from the perspective of the player who just acted (standard episodic RL convention):
          +1 win, -1 loss, 0 draw/nonterminal
        """
        if self.done:
            raise RuntimeError("step() called on terminal state")

        a = self.action_space.decode(action_id)
        legal = self.legal_action_mask()
        if not legal[action_id]:
            raise ValueError(f"Illegal action: {a}")

        progress = False
        captured_pid: Optional[int] = None

        if a.kind == "flip":
            sq = a.src
            self.revealed[sq] = True
            progress = True

            # assign colors if first flip
            if self.player_color[0] is None and self.player_color[1] is None:
                color = int(PID_COLOR[self.piece[sq]])
                self.player_color[self.to_play] = color
                self.player_color[1 - self.to_play] = 1 - color

        else:  # move/capture
            src, dst = a.src, a.dst
            # move piece
            if self.piece[dst] != 0:
                # capture
                captured_pid = self.piece[dst]
                self.captured.append(int(captured_pid))
                progress = True
            self.piece[dst] = self.piece[src]
            self.revealed[dst] = True  # moved piece stays revealed
            self.piece[src] = 0
            self.revealed[src] = False

        # Update counters
        self.plies += 1
        self._append_history(action_id)

        if progress:
            self.no_progress_plies = 0
        else:
            self.no_progress_plies += 1

        # switch player
        self.to_play = 1 - self.to_play

        # check terminal
        self._update_terminal()

        obs = self.observe(max_history=len(self.history))  # caller can truncate
        reward = 0.0
        if self.done:
            if self.winner is None:
                reward = 0.0
            else:
                # winner is the player who made the last move; because we already toggled to_play,
                # the "previous player" is 1 - self.to_play.
                last_player = 1 - self.to_play
                reward = 1.0 if self.winner == last_player else -1.0

        info = {"captured": captured_pid}
        return obs, reward, self.done, info

    def step_search(self, action_id: int) -> None:
        """
        Apply an action for search:
          - no legality check
          - no observe/reward return
          - terminal update WITHOUT stalemate scan
        """
        if self.done:
            return

        a = self.action_space.decode(int(action_id))
        progress = False

        if a.kind == "flip":
            sq = a.src
            self.revealed[sq] = True
            progress = True

            # assign colors if first flip
            if self.player_color[0] is None and self.player_color[1] is None:
                color = int(PID_COLOR[self.piece[sq]])
                self.player_color[self.to_play] = color
                self.player_color[1 - self.to_play] = 1 - color

        else:  # move/capture
            src, dst = a.src, a.dst
            if self.piece[dst] != 0:
                self.captured.append(int(self.piece[dst]))
                progress = True
            self.piece[dst] = self.piece[src]
            self.revealed[dst] = True
            self.piece[src] = 0
            self.revealed[src] = False

        self.plies += 1
        self._append_history(action_id)
        self.no_progress_plies = 0 if progress else (self.no_progress_plies + 1)
        self.to_play = 1 - self.to_play

        self._update_terminal_fast()

    def step_search_with_undo(
        self, action_id: int, chance_flip: bool = False
    ) -> _SearchUndo:
        """Apply an action for search and return an undo record for fast rollback."""
        if self.done:
            return _SearchUndo(
                kind="noop",
                src=-1,
                dst=-1,
                prev_piece_src=0,
                prev_piece_dst=0,
                prev_revealed_src=False,
                prev_revealed_dst=False,
                prev_player_color=(self.player_color[0], self.player_color[1]),
                prev_to_play=int(self.to_play),
                prev_plies=int(self.plies),
                prev_no_progress_plies=int(self.no_progress_plies),
                prev_done=bool(self.done),
                prev_winner=self.winner,
                prev_captured_len=len(self.captured),
                history_appended=False,
                history_dropped=[],
            )

        a = self.action_space.decode(int(action_id))
        prev_player_color = (self.player_color[0], self.player_color[1])
        prev_to_play = int(self.to_play)
        prev_plies = int(self.plies)
        prev_no_progress = int(self.no_progress_plies)
        prev_done = bool(self.done)
        prev_winner = self.winner
        prev_captured_len = len(self.captured)

        progress = False

        if a.kind == "flip":
            sq = a.src
            prev_revealed = bool(self.revealed[sq])
            prev_piece = int(self.piece[sq])

            if chance_flip and (not self.revealed[sq]) and prev_piece != 0:
                counts = self.public_remaining_hidden_counts()
                weights = counts[1:].astype(np.float64)
                total = float(weights.sum())
                if total > 0:
                    r = self.rng.random() * total
                    acc = 0.0
                    sampled_pid = prev_piece
                    for idx, w in enumerate(weights, start=1):
                        if w <= 0:
                            continue
                        acc += float(w)
                        if r <= acc:
                            sampled_pid = int(idx)
                            break
                    self.piece[sq] = sampled_pid
            self.revealed[sq] = True
            progress = True

            if self.player_color[0] is None and self.player_color[1] is None:
                color = int(PID_COLOR[self.piece[sq]])
                self.player_color[self.to_play] = color
                self.player_color[1 - self.to_play] = 1 - color

            history_appended, history_dropped = self._append_history_with_undo(
                action_id
            )

            self.plies += 1
            self.no_progress_plies = 0
            self.to_play = 1 - self.to_play

            self._update_terminal_fast()
            return _SearchUndo(
                kind="flip",
                src=sq,
                dst=sq,
                prev_piece_src=prev_piece,
                prev_piece_dst=prev_piece,
                prev_revealed_src=prev_revealed,
                prev_revealed_dst=prev_revealed,
                prev_player_color=prev_player_color,
                prev_to_play=prev_to_play,
                prev_plies=prev_plies,
                prev_no_progress_plies=prev_no_progress,
                prev_done=prev_done,
                prev_winner=prev_winner,
                prev_captured_len=prev_captured_len,
                history_appended=history_appended,
                history_dropped=history_dropped,
            )

        # move/capture
        src, dst = a.src, a.dst
        prev_piece_src = int(self.piece[src])
        prev_piece_dst = int(self.piece[dst])
        prev_revealed_src = bool(self.revealed[src])
        prev_revealed_dst = bool(self.revealed[dst])

        if self.piece[dst] != 0:
            self.captured.append(int(self.piece[dst]))
            progress = True

        self.piece[dst] = self.piece[src]
        self.revealed[dst] = True
        self.piece[src] = 0
        self.revealed[src] = False

        history_appended, history_dropped = self._append_history_with_undo(action_id)

        self.plies += 1
        self.no_progress_plies = 0 if progress else (self.no_progress_plies + 1)
        self.to_play = 1 - self.to_play

        self._update_terminal_fast()

        return _SearchUndo(
            kind="move",
            src=src,
            dst=dst,
            prev_piece_src=prev_piece_src,
            prev_piece_dst=prev_piece_dst,
            prev_revealed_src=prev_revealed_src,
            prev_revealed_dst=prev_revealed_dst,
            prev_player_color=prev_player_color,
            prev_to_play=prev_to_play,
            prev_plies=prev_plies,
            prev_no_progress_plies=prev_no_progress,
            prev_done=prev_done,
            prev_winner=prev_winner,
            prev_captured_len=prev_captured_len,
            history_appended=history_appended,
            history_dropped=history_dropped,
        )

    def undo_search(self, undo: _SearchUndo) -> None:
        if undo.kind == "noop":
            return

        # restore core state
        self.player_color = [undo.prev_player_color[0], undo.prev_player_color[1]]
        self.to_play = int(undo.prev_to_play)
        self.plies = int(undo.prev_plies)
        self.no_progress_plies = int(undo.prev_no_progress_plies)
        self.done = bool(undo.prev_done)
        self.winner = undo.prev_winner

        # restore pieces / revealed
        if undo.kind == "flip":
            self.piece[undo.src] = int(undo.prev_piece_src)
            self.revealed[undo.src] = bool(undo.prev_revealed_src)
        else:
            self.piece[undo.src] = int(undo.prev_piece_src)
            self.piece[undo.dst] = int(undo.prev_piece_dst)
            self.revealed[undo.src] = bool(undo.prev_revealed_src)
            self.revealed[undo.dst] = bool(undo.prev_revealed_dst)

        # restore captured list
        if len(self.captured) > int(undo.prev_captured_len):
            del self.captured[int(undo.prev_captured_len) :]

        # restore history
        if undo.history_appended and self.history:
            self.history.pop()
        if undo.history_dropped:
            self.history = undo.history_dropped + self.history

    def _update_terminal(self) -> None:
        if self.no_progress_plies >= self.draw_plies:
            self.done = True
            self.winner = None
            return

        # If colors not assigned yet, cannot be terminal
        if self.player_color[0] is None and self.player_color[1] is None:
            return

        # Win by capturing all opponent pieces (of opponent color)
        rem_red, rem_black = self._remaining_by_color_from_captured()

        # If any color has 0 pieces, the player of that color loses.
        if rem_red == 0 or rem_black == 0:
            self.done = True
            # map color winner to player winner
            winner_color = BLACK if rem_red == 0 else RED
            # player_color is assigned; find which player has winner_color
            self.winner = 0 if self.player_color[0] == winner_color else 1
            return

        # Win by stalemate: current player has no legal actions
        if not self._has_any_legal_move():
            self.done = True
            self.winner = 1 - self.to_play
            return

    def _update_terminal_fast(self) -> None:
        if self.no_progress_plies >= self.draw_plies:
            self.done = True
            self.winner = None
            return

        # If colors not assigned yet, cannot be terminal
        if self.player_color[0] is None and self.player_color[1] is None:
            return

        # Win by capturing all opponent pieces (of opponent color)
        rem_red, rem_black = self._remaining_by_color_from_captured()

        if rem_red == 0 or rem_black == 0:
            self.done = True
            winner_color = BLACK if rem_red == 0 else RED
            self.winner = 0 if self.player_color[0] == winner_color else 1
            return

    # ---------- Observation ----------
    def observe(self, max_history: int = 16) -> Observation:
        mask = self.legal_action_mask()
        hist = self.history[-max_history:] if max_history > 0 else []
        return Observation(
            board_tokens=self.board_tokens(),
            belief=self.belief_marginals(),
            history_actions=np.asarray(hist, dtype=np.int64),
            to_play_color=self.to_play_color_token(),
            plies=int(self.plies),
            no_progress_plies=int(self.no_progress_plies),
            action_mask=mask,
        )

    # ---------- Debug / pretty print ----------
    def _tok_to_str(self, tok: int) -> str:
        if tok == EMPTY_TOK:
            return "."
        if tok == COVERED_TOK:
            return "?"
        pid = tok - 2
        color = int(PID_COLOR[pid])
        kind = int(PID_KIND[pid])
        ch = KIND_NAMES[kind]
        return ch.lower() if color == BLACK else ch

    def render(self) -> str:
        bt = self.board_tokens()
        rows, cols = self.action_space.rows, self.action_space.cols
        lines = []
        for r in range(rows):
            row = []
            for c in range(cols):
                i = rc_to_i(r, c, rows, cols)
                row.append(self._tok_to_str(int(bt[i])))
            lines.append(" ".join(row))
        col = self.to_play_color_token()
        col_s = "?" if col == UNKNOWN_COLOR else ("R" if col == RED else "B")
        lines.append(
            f"to_play: P{self.to_play} (color {col_s}), plies={self.plies}, no_progress={self.no_progress_plies}"
        )
        return "\n".join(lines)
