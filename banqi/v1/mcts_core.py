"""banqi.v1.mcts_core

Information-set MCTS (PUCT) + root-sampling (determinization) for Banqi.

This module is evaluator-agnostic and intentionally does NOT import torch.

Why this design?
----------------
A common imperfect-information baseline is PIMC (Perfect-Information Monte Carlo):
  1) Sample a determinization (assign identities to hidden pieces).
  2) Run a perfect-information MCTS on that determinization.
  3) Repeat and aggregate.

The classic implementation ("one tree per determinization") wastes work: statistics do not
share across determinizations, and the search can suffer from strategy-fusion artifacts.

This implementation instead builds a *single* tree keyed by an **information-set hash**:
- Revealed pieces are hashed by identity.
- Face-down pieces are hashed only as "covered" (identity ignored).
- Empty squares are ignored (as in typical Zobrist hashing).

We still use root sampling (determinization) to resolve chance outcomes (notably flips),
but all simulations update the same information-set tree.

Interface
---------
The search core accepts a callback:
    policy_value_fn(obs_list) -> (priors, values)

Where:
  - obs_list: List[Observation]
  - priors: np.ndarray [B, A] float32, *already normalized over legal actions*
  - values: np.ndarray [B] float32 in [-1, 1], from the perspective of the player to play

The env MUST provide:
  - env.clone()
  - env.step_search(action_id)   (fast step for MCTS)
  - env.observe(max_history)
  - env.done, env.winner, env.to_play

This file provides:
  - MCTSConfig
  - PIMCTreeReuse (root reuse across moves; resets on flips)
  - pimc_mcts_policy (stateless convenience wrapper)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .actions import ActionSpace
from .env import BanqiEnv, Observation

PolicyValueFn = Callable[[List[Observation]], Tuple[np.ndarray, np.ndarray]]


@dataclass
class MCTSConfig:
    num_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    num_determinize: int = 8
    batch_leaves: int = 64
    virtual_loss: float = 1.0
    # progressive widening for flip actions (action ids < num_squares)
    # set flip_pw_k <= 0 to disable and expand all flips
    flip_pw_k: int = 8
    flip_pw_alpha: float = 0.5
    flip_pw_max: int = 32
    flip_as_chance: bool = False  # Current implementation has some problems when True


# ----------------- Zobrist hashing (information-set) -----------------
@dataclass
class _ZobristTables:
    # NOTE: piece table is indexed by:
    #   - square i
    #   - pid_idx in [0..14]
    #   - rev_idx in {0,1}
    #
    # We interpret:
    #   (pid_idx=0, rev_idx=0)  : empty square (we typically skip empties)
    #   (pid_idx=0, rev_idx=1)  : covered/unknown piece (identity ignored)
    #   (pid_idx=1..14, rev_idx=1) : revealed piece identity
    piece: np.ndarray  # [S, 15, 2] uint64
    to_play: np.ndarray  # [2] uint64
    player_color: np.ndarray  # [2,3] uint64
    no_progress: np.ndarray  # [draw_plies+1] uint64
    captured: np.ndarray  # [max_captures+1, 15] uint64 (capture_index, pid)


_ZOBRIST_CACHE: Dict[Tuple[int, int, int], _ZobristTables] = {}


def _get_zobrist(action_space: ActionSpace, draw_plies: int) -> _ZobristTables:
    key = (action_space.rows, action_space.cols, int(draw_plies))
    if key in _ZOBRIST_CACHE:
        return _ZOBRIST_CACHE[key]

    rng = np.random.RandomState(0xBADC0FFE)
    S = action_space.num_squares
    piece = rng.randint(0, 2**63 - 1, size=(S, 15, 2), dtype=np.uint64)
    to_play = rng.randint(0, 2**63 - 1, size=(2,), dtype=np.uint64)
    player_color = rng.randint(0, 2**63 - 1, size=(2, 3), dtype=np.uint64)
    no_progress = rng.randint(
        0, 2**63 - 1, size=(int(draw_plies) + 1,), dtype=np.uint64
    )
    max_captures = int(action_space.num_squares)
    captured = rng.randint(0, 2**63 - 1, size=(max_captures + 1, 15), dtype=np.uint64)

    z = _ZobristTables(
        piece=piece,
        to_play=to_play,
        player_color=player_color,
        no_progress=no_progress,
        captured=captured,
    )
    _ZOBRIST_CACHE[key] = z
    return z


def _state_hash(env: BanqiEnv, z: _ZobristTables) -> np.uint64:
    """Information-set hash.

    - Empty squares do not contribute.
    - Revealed squares hash by true identity.
    - Covered squares hash as a generic "covered" marker (pid_idx=0, rev_idx=1).

    This allows statistics to share across different determinizations that correspond to
    the same public state.
    """

    h = np.uint64(0)
    S = env.action_space.num_squares

    for i in range(S):
        pid = int(env.piece[i])
        if pid <= 0:
            continue
        # revealed -> hash identity; covered -> hash generic covered marker
        pid_idx = pid if env.revealed[i] else 0
        h ^= z.piece[i, pid_idx, 1]

    h ^= z.to_play[int(env.to_play)]

    c0 = 2 if env.player_color[0] is None else int(env.player_color[0])
    c1 = 2 if env.player_color[1] is None else int(env.player_color[1])
    h ^= z.player_color[0, c0]
    h ^= z.player_color[1, c1]

    np_idx = int(min(max(0, env.no_progress_plies), len(z.no_progress) - 1))
    h ^= z.no_progress[np_idx]

    # captured pieces are public info and affect belief; include in hash
    for k, pid in enumerate(env.captured):
        if k >= z.captured.shape[0]:
            break
        p = int(pid)
        if 0 <= p < 15:
            h ^= z.captured[k, p]

    return h


def _square_hash(i: int, pid: int, revealed: bool, z: _ZobristTables) -> np.uint64:
    if pid <= 0:
        return np.uint64(0)
    pid_idx = int(pid) if revealed else 0
    return z.piece[int(i), pid_idx, 1]


def _no_progress_idx(plies: int, z: _ZobristTables) -> int:
    return int(min(max(0, int(plies)), len(z.no_progress) - 1))


def _update_hash_after_action(
    h: np.uint64, env: BanqiEnv, undo: Any, z: _ZobristTables
) -> np.uint64:
    if undo.kind == "noop":
        return h

    # squares
    if undo.kind == "flip":
        i = int(undo.src)
        h ^= _square_hash(i, int(undo.prev_piece_src), bool(undo.prev_revealed_src), z)
        h ^= _square_hash(i, int(env.piece[i]), bool(env.revealed[i]), z)
    else:
        src = int(undo.src)
        dst = int(undo.dst)
        h ^= _square_hash(
            src, int(undo.prev_piece_src), bool(undo.prev_revealed_src), z
        )
        h ^= _square_hash(
            dst, int(undo.prev_piece_dst), bool(undo.prev_revealed_dst), z
        )
        h ^= _square_hash(src, int(env.piece[src]), bool(env.revealed[src]), z)
        h ^= _square_hash(dst, int(env.piece[dst]), bool(env.revealed[dst]), z)

    # to_play
    h ^= z.to_play[int(undo.prev_to_play)]
    h ^= z.to_play[int(env.to_play)]

    # player colors
    c0_prev = 2 if undo.prev_player_color[0] is None else int(undo.prev_player_color[0])
    c1_prev = 2 if undo.prev_player_color[1] is None else int(undo.prev_player_color[1])
    c0_new = 2 if env.player_color[0] is None else int(env.player_color[0])
    c1_new = 2 if env.player_color[1] is None else int(env.player_color[1])
    h ^= z.player_color[0, c0_prev]
    h ^= z.player_color[1, c1_prev]
    h ^= z.player_color[0, c0_new]
    h ^= z.player_color[1, c1_new]

    # no_progress_plies
    h ^= z.no_progress[_no_progress_idx(int(undo.prev_no_progress_plies), z)]
    h ^= z.no_progress[_no_progress_idx(int(env.no_progress_plies), z)]

    # captured pieces (order-sensitive hash)
    if undo.kind == "move":
        cap_pid = int(undo.prev_piece_dst)
        if cap_pid != 0:
            cap_idx = int(undo.prev_captured_len)
            if 0 <= cap_idx < z.captured.shape[0] and 0 <= cap_pid < 15:
                h ^= z.captured[cap_idx, cap_pid]

    return h


# ----------------- Tree data structures -----------------
@dataclass
class Edge:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0

    @property
    def q(self) -> float:
        if self.visit_count <= 0:
            return 0.0
        return self.value_sum / float(self.visit_count)


class Node:
    __slots__ = (
        "visit_count",
        "edges",
        "is_expanded",
        "in_flight",
        "flip_order",
        "flip_priors",
        "flip_added",
    )

    def __init__(self) -> None:
        self.visit_count: int = 0
        self.edges: Dict[int, Edge] = {}
        self.is_expanded: bool = False
        self.in_flight: bool = False
        self.flip_order: List[int] = []
        self.flip_priors: List[float] = []
        self.flip_added: int = 0

    def expand(
        self,
        priors: np.ndarray,
        legal_mask: np.ndarray,
        num_squares: int,
        cfg: MCTSConfig,
    ) -> None:
        if self.is_expanded:
            return
        self.is_expanded = True

        legal_ids = np.nonzero(legal_mask)[0]
        if cfg.flip_pw_k <= 0:
            for a_id in legal_ids:
                a = int(a_id)
                self.edges[a] = Edge(prior=float(priors[a]))
            return

        flip_ids: List[int] = []
        for a_id in legal_ids:
            a = int(a_id)
            if a < num_squares:
                flip_ids.append(a)
            else:
                self.edges[a] = Edge(prior=float(priors[a]))

        if not flip_ids:
            return

        flip_ids.sort(key=lambda x: float(priors[x]), reverse=True)
        self.flip_order = flip_ids
        self.flip_priors = [float(priors[a]) for a in flip_ids]
        self.flip_added = 0
        self.maybe_expand_flips(cfg)

    def maybe_expand_flips(self, cfg: MCTSConfig) -> None:
        if not self.flip_order:
            return
        if cfg.flip_pw_k <= 0:
            return
        target = int(max(cfg.flip_pw_k, (self.visit_count + 1) ** cfg.flip_pw_alpha))
        if cfg.flip_pw_max > 0:
            target = min(target, int(cfg.flip_pw_max))
        target = min(target, len(self.flip_order))
        if target <= self.flip_added:
            return
        for i in range(self.flip_added, target):
            a = int(self.flip_order[i])
            self.edges[a] = Edge(prior=float(self.flip_priors[i]))
        self.flip_added = target


def terminal_value(env: BanqiEnv) -> float:
    """Value from perspective of env.to_play (player to move at terminal state)."""
    if not env.done:
        raise ValueError("terminal_value called on non-terminal env")
    if env.winner is None:
        return 0.0
    return 1.0 if env.winner == env.to_play else -1.0


def _visits_to_pi(
    visits: np.ndarray, legal: np.ndarray, temperature: float
) -> np.ndarray:
    """Convert visit counts to a policy distribution."""
    v = visits.astype(np.float32, copy=True)
    v[~legal] = 0.0

    if temperature <= 1e-8:
        pi = np.zeros_like(v, dtype=np.float32)
        if v.sum() <= 0:
            legal_ids = np.nonzero(legal)[0]
            if len(legal_ids) > 0:
                pi[int(np.random.choice(legal_ids))] = 1.0
            return pi
        pi[int(np.argmax(v))] = 1.0
        return pi

    x = np.power(v + 1e-8, 1.0 / float(temperature)).astype(np.float32)
    x[~legal] = 0.0
    z = float(x.sum())
    if z <= 0:
        pi = legal.astype(np.float32)
        pi /= float(pi.sum() + 1e-8)
        return pi
    return (x / z).astype(np.float32)


# ----------------- Information-set tree -----------------
class InfoSetMCTree:
    """A single information-set MCTS tree (PUCT) with a transposition table keyed by infoset hash."""

    def __init__(self, cfg: MCTSConfig, max_history: int) -> None:
        self.cfg = cfg
        self.max_history = int(max_history)
        self.num_squares: int = 0

        self.root: Optional[Node] = None
        self.root_hash: Optional[np.uint64] = None
        self.tt: Dict[np.uint64, Node] = {}
        self.z: Optional[_ZobristTables] = None

        # representative env for root (publicly equivalent); used for root eval on reset
        self.root_env: Optional[BanqiEnv] = None
        self.root_net_value: float = 0.0

    def clear(self) -> None:
        self.root = None
        self.root_hash = None
        self.tt = {}
        self.z = None
        self.root_env = None
        self.root_net_value = 0.0

    def reset(self, root_env: BanqiEnv, policy_value_fn: PolicyValueFn) -> None:
        # Representative env (may include hidden identities; hash ignores them)
        self.root_env = root_env.clone()
        self.z = _get_zobrist(self.root_env.action_space, self.root_env.draw_plies)
        self.num_squares = int(self.root_env.action_space.num_squares)

        self.root_hash = _state_hash(self.root_env, self.z)
        self.root = Node()
        self.tt = {self.root_hash: self.root}

        # Expand root once
        obs0 = self.root_env.observe(max_history=self.max_history)
        priors0, values0 = policy_value_fn([obs0])
        priors = priors0[0].astype(np.float32, copy=False)
        self.root_net_value = float(values0[0])
        self.root.expand(
            priors,
            obs0.action_mask.astype(np.bool_, copy=False),
            self.num_squares,
            self.cfg,
        )

    def advance(self, action_id: int) -> None:
        """Advance the root after applying a real move (not a flip reset).

        This keeps only the new root node in the TT to bound memory.
        """
        if self.root is None or self.z is None:
            return
        if self.root_env is None:
            # no representative env; just clear
            self.clear()
            return

        self.root_env.step_search(int(action_id))
        self.root_hash = _state_hash(self.root_env, self.z)

        new_root = self.tt.get(self.root_hash)
        if new_root is None:
            new_root = Node()

        self.root = new_root
        self.root.in_flight = False
        self.tt = {self.root_hash: self.root}
        self.root_net_value = 0.0

    def root_visits(self, action_dim: int) -> np.ndarray:
        v = np.zeros(int(action_dim), dtype=np.float32)
        if self.root is None:
            return v
        for a_id, e in self.root.edges.items():
            v[int(a_id)] = float(e.visit_count)
        return v

    def root_value(self) -> float:
        """Visit-weighted Q at the root; falls back to the network value if unvisited."""
        if self.root is None:
            return 0.0
        total = 0
        s = 0.0
        for e in self.root.edges.values():
            if e.visit_count <= 0:
                continue
            total += e.visit_count
            s += float(e.q) * float(e.visit_count)
        if total <= 0:
            return float(self.root_net_value)
        return float(s / float(total))

    def make_root_noise(self, action_dim: int) -> Optional[np.ndarray]:
        """Compute a noisy prior vector for root selection (Dirichlet), or None if disabled."""
        if self.root is None:
            return None
        if self.cfg.dirichlet_epsilon <= 0 or self.cfg.dirichlet_alpha <= 0:
            return None

        legal_actions = list(self.root.edges.keys())
        if len(legal_actions) == 0:
            return None

        noise = np.random.dirichlet(
            [self.cfg.dirichlet_alpha] * len(legal_actions)
        ).astype(np.float32)

        root_noise = np.zeros(int(action_dim), dtype=np.float32)
        eps = float(self.cfg.dirichlet_epsilon)
        for a_id, n in zip(legal_actions, noise):
            base = float(self.root.edges[int(a_id)].prior)
            root_noise[int(a_id)] = (1.0 - eps) * base + eps * float(n)
        return root_noise

    def _ensure_root_expanded(
        self,
        root_env: BanqiEnv,
        policy_value_fn: PolicyValueFn,
    ) -> None:
        if self.root is None:
            raise RuntimeError("Tree has no root; call reset() first")
        if self.root.is_expanded:
            return
        obs0 = root_env.observe(max_history=self.max_history)
        priors0, values0 = policy_value_fn([obs0])
        priors = priors0[0].astype(np.float32, copy=False)
        self.root_net_value = float(values0[0])
        self.root.expand(
            priors,
            obs0.action_mask.astype(np.bool_, copy=False),
            self.num_squares,
            self.cfg,
        )

    def search(
        self,
        root_env: BanqiEnv,
        policy_value_fn: PolicyValueFn,
        root_noise: Optional[np.ndarray] = None,
    ) -> None:
        """Run cfg.num_simulations simulations starting from the given determinized root env.

        All simulations update the shared information-set transposition table.
        """
        if self.root is None:
            raise RuntimeError("search() called before reset()")
        if self.z is None:
            self.z = _get_zobrist(root_env.action_space, root_env.draw_plies)

        self._ensure_root_expanded(root_env, policy_value_fn)

        cfg = self.cfg
        z = self.z
        tt = self.tt

        # Pending NN evaluations (batched)
        pending: List[Tuple[Node, List[Node], List[int], Observation]] = []

        def backup(nodes: List[Node], actions: List[int], value: float) -> None:
            # node visit counts: include leaf
            nodes[-1].visit_count += 1
            # value is from perspective of the player to play at the leaf;
            # edges at the parent are from the parent's perspective => flip sign once first.
            v = -float(value)
            # backup along edges
            for n, a in zip(reversed(nodes[:-1]), reversed(actions)):
                n.visit_count += 1
                e = n.edges[int(a)]
                e.visit_count += 1
                e.value_sum += v
                v = -v

        def flush_pending() -> None:
            if not pending:
                return
            obs_list = [p[3] for p in pending]
            priors_b, values_b = policy_value_fn(obs_list)
            for (node, nodes, actions, obs), priors_i, value_i in zip(
                pending, priors_b, values_b
            ):
                node.expand(priors_i, obs.action_mask, self.num_squares, cfg)
                node.in_flight = False
                backup(nodes, actions, float(value_i))
            pending.clear()

        batch_leaves = max(1, int(cfg.batch_leaves))

        env = root_env.clone()
        h = _state_hash(env, z)

        def rollback(undo_stack: List[Any], hash_stack: List[np.uint64]) -> None:
            nonlocal h
            if not undo_stack:
                return
            for u in reversed(undo_stack):
                env.undo_search(u)
                if hash_stack:
                    h = hash_stack.pop()

        for _ in range(int(cfg.num_simulations)):
            node = self.root
            nodes: List[Node] = [node]
            actions: List[int] = []
            undo_stack: List[Any] = []
            hash_stack: List[np.uint64] = []

            rn = root_noise  # only used on the first selection from root

            # Traverse until an unexpanded node or terminal
            while node.is_expanded and (not env.done):
                node.maybe_expand_flips(cfg)
                if not node.edges:
                    break

                # select action
                sqrt_N = np.sqrt(float(node.visit_count) + 1.0)
                best_score = -1e18
                best_a = -1

                for a_id, e in node.edges.items():
                    prior = float(e.prior)
                    if rn is not None:
                        prior = float(rn[int(a_id)])
                    q = float(e.q)
                    u = (
                        float(cfg.c_puct)
                        * prior
                        * float(sqrt_N)
                        / (1.0 + float(e.visit_count))
                    )
                    score = q + u
                    if score > best_score:
                        best_score = score
                        best_a = int(a_id)

                if best_a < 0:
                    break

                actions.append(best_a)
                hash_stack.append(h)
                undo = env.step_search_with_undo(
                    best_a, chance_flip=bool(cfg.flip_as_chance)
                )
                undo_stack.append(undo)
                h = _update_hash_after_action(h, env, undo, z)

                # after the first step, disable root noise
                rn = None

                # transposition lookup
                # use incremental hash
                nxt = tt.get(h)
                if nxt is None:
                    nxt = Node()
                    tt[h] = nxt
                node = nxt
                nodes.append(node)

            # Terminal leaf
            if env.done:
                backup(nodes, actions, terminal_value(env))
                rollback(undo_stack, hash_stack)
                continue

            # If node already expanded but has no edges, treat as no-move => lose
            obs = env.observe(max_history=self.max_history)
            if not obs.action_mask.any():
                backup(nodes, actions, -1.0)
                rollback(undo_stack, hash_stack)
                continue

            if node.is_expanded:
                # Expanded but we stopped due to missing edges; be conservative
                backup(nodes, actions, 0.0)
                rollback(undo_stack, hash_stack)
                continue

            # Avoid duplicate eval of same leaf within batch window
            if node.in_flight:
                vloss = float(cfg.virtual_loss)
                backup(nodes, actions, -vloss if vloss > 0 else 0.0)
                rollback(undo_stack, hash_stack)
                continue

            node.in_flight = True
            pending.append((node, nodes, actions, obs))
            rollback(undo_stack, hash_stack)

            if len(pending) >= batch_leaves:
                flush_pending()

        flush_pending()


# ----------------- Root-sampling PIMC wrapper -----------------
class PIMCTreeReuse:
    """Root-sampling information-set MCTS with reuse across moves.

    - Keeps a set of determinized "particles" of the hidden pieces.
    - Runs PUCT simulations on each particle, updating a shared information-set tree.
    - On a FLIP action, hidden information is revealed -> the particle set becomes invalid.
      Call advance(..., is_flip=True) to reset.

    This mirrors the old API used elsewhere in the repo.
    """

    def __init__(
        self,
        cfg: MCTSConfig,
        max_history: int,
        determinize_seeds: Optional[List[int]] = None,
    ) -> None:
        self.cfg = cfg
        self.max_history = int(max_history)

        if determinize_seeds is not None:
            seeds = list(determinize_seeds)
            if len(seeds) < cfg.num_determinize:
                extra = [
                    int(np.random.randint(0, 2**31 - 1))
                    for _ in range(cfg.num_determinize - len(seeds))
                ]
                seeds.extend(extra)
            self.determinize_seeds = seeds[: int(cfg.num_determinize)]
        else:
            self.determinize_seeds = None

        self.tree = InfoSetMCTree(cfg=cfg, max_history=max_history)
        self.particles: Optional[List[BanqiEnv]] = None
        self.needs_reset: bool = True

    def reset(self) -> None:
        self.needs_reset = True
        self.particles = None
        self.tree.clear()

    def advance(self, action_id: int, is_flip: bool = False) -> None:
        if is_flip:
            self.reset()
            return

        if self.particles is not None:
            for p in self.particles:
                p.step_search(int(action_id))

        self.tree.advance(int(action_id))

    def policy(
        self, real_env: BanqiEnv, policy_value_fn: PolicyValueFn
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        A = len(real_env.action_space)

        # seeds
        if self.determinize_seeds is None:
            self.determinize_seeds = [
                int(np.random.randint(0, 2**31 - 1))
                for _ in range(int(self.cfg.num_determinize))
            ]

        # reset particle set + tree at start or after flips
        if self.needs_reset or self.particles is None or self.tree.root is None:
            self.particles = [
                real_env.determinize(seed=int(s)) for s in self.determinize_seeds
            ]
            self.tree.reset(real_env, policy_value_fn)
            self.needs_reset = False

        # root noise sampled once per decision
        root_noise = self.tree.make_root_noise(action_dim=A)

        # run sims on each particle
        assert self.particles is not None
        for p in self.particles:
            self.tree.search(p, policy_value_fn=policy_value_fn, root_noise=root_noise)

        visits = self.tree.root_visits(action_dim=A)
        legal = real_env.legal_action_mask().astype(np.bool_, copy=False)
        pi_play = _visits_to_pi(visits, legal, temperature=float(self.cfg.temperature))
        v_root = self.tree.root_value()
        return pi_play, visits, float(v_root)


def pimc_mcts_policy(
    real_env: BanqiEnv,
    policy_value_fn: PolicyValueFn,
    cfg: MCTSConfig,
    max_history: int,
    determinize_seeds: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Root-sampling PIMC ensemble that updates a single information-set tree.

    Returns:
      pi_play: [A] float32 distribution (temperature applied)
      visits : [A] float32 aggregated visit counts
      v_root : float32 visit-weighted root value (perspective of player to play)
    """
    A = len(real_env.action_space)

    if determinize_seeds is None:
        determinize_seeds = [
            int(np.random.randint(0, 2**31 - 1))
            for _ in range(int(cfg.num_determinize))
        ]

    tree = InfoSetMCTree(cfg=cfg, max_history=max_history)
    tree.reset(real_env, policy_value_fn)
    root_noise = tree.make_root_noise(action_dim=A)

    for s in determinize_seeds:
        det = real_env.determinize(seed=int(s))
        tree.search(det, policy_value_fn=policy_value_fn, root_noise=root_noise)

    visits = tree.root_visits(action_dim=A)
    legal = real_env.legal_action_mask().astype(np.bool_, copy=False)
    pi_play = _visits_to_pi(visits, legal, temperature=float(cfg.temperature))
    v_root = tree.root_value()
    return pi_play, visits, float(v_root)
