from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import numpy as np
import torch

from banqi.v1.actions import ActionSpace, i_to_rc
from banqi.v1.env import BanqiEnv, BLACK, RED, UNKNOWN_COLOR
from banqi.v1.mcts import MCTSConfig, pimc_mcts_policy
from banqi.v1.model import BanqiTransformer
from banqi.v1.utils import load_checkpoint


@dataclass
class AppState:
    action_space: ActionSpace
    env: Optional[BanqiEnv]
    model: Optional[BanqiTransformer]
    device: torch.device
    human_player: int
    selected_src: Optional[int]
    last_action: Optional[int]


def _default_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def _init_state() -> AppState:
    return AppState(
        action_space=ActionSpace(),
        env=None,
        model=None,
        device=torch.device("cpu"),
        human_player=0,
        selected_src=None,
        last_action=None,
    )


def _piece_label(env: BanqiEnv, idx: int) -> str:
    tok = int(env.board_tokens()[idx])
    return env._tok_to_str(tok)


def _coord_label(idx: int, rows: int, cols: int) -> str:
    r, c = i_to_rc(idx, rows, cols)
    col = chr(ord("A") + c)
    row = str(r + 1)
    return f"{col}{row}"


def _action_to_str(action_space: ActionSpace, action_id: Optional[int]) -> str:
    if action_id is None:
        return "-"
    a = action_space.decode(int(action_id))
    if a.kind == "flip":
        return f"flip {_coord_label(a.src, action_space.rows, action_space.cols)}"
    return (
        f"move {_coord_label(a.src, action_space.rows, action_space.cols)}"
        f" -> {_coord_label(a.dst, action_space.rows, action_space.cols)}"
    )


def _status_text(state: AppState) -> str:
    if state.env is None:
        return "No game running. Load a model and start a new game."
    env = state.env
    p0_color = env.player_color[0]
    p1_color = env.player_color[1]

    def color_label(color: Optional[int]) -> str:
        if color is None or color == UNKNOWN_COLOR:
            return "?"
        return "R" if int(color) == RED else "B"

    who = "Human" if env.to_play == state.human_player else "AI"
    if env.done:
        if env.winner is None:
            result = "Draw"
        else:
            result = f"Winner: P{env.winner} ({'Human' if env.winner == state.human_player else 'AI'})"
        return (
            f"Game over. {result}. "
            f"P0={color_label(p0_color)} P1={color_label(p1_color)}. "
            f"Last action: {_action_to_str(state.action_space, state.last_action)}"
        )

    return (
        f"Turn: P{env.to_play} ({who}). "
        f"P0={color_label(p0_color)} P1={color_label(p1_color)}. "
        f"Last action: {_action_to_str(state.action_space, state.last_action)}"
    )


def _board_labels(state: AppState) -> list[str]:
    action_space = state.action_space
    if state.env is None:
        return ["" for _ in range(action_space.num_squares)]

    env = state.env
    labels: list[str] = []
    last_src = None
    last_dst = None
    if state.last_action is not None:
        a = action_space.decode(int(state.last_action))
        last_src = int(a.src)
        last_dst = int(a.dst)

    for idx in range(action_space.num_squares):
        coord = _coord_label(idx, action_space.rows, action_space.cols)
        piece = _piece_label(env, idx)
        marker = ""
        if state.selected_src == idx:
            marker = "[S]"
        elif idx == last_src or idx == last_dst:
            marker = "*"
        labels.append(f"{coord}\n{piece}{marker}")
    return labels


def _select_ai_action(
    state: AppState,
    num_simulations: int,
    c_puct: float,
    num_determinize: int,
    temperature: float,
) -> Optional[int]:
    if state.env is None or state.model is None:
        return None

    mcts_cfg = MCTSConfig(
        num_simulations=int(num_simulations),
        c_puct=float(c_puct),
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
        temperature=float(temperature),
        num_determinize=int(num_determinize),
    )
    pi, _visits, _v_root = pimc_mcts_policy(
        state.env, state.model, mcts_cfg, device=state.device
    )
    legal = state.env.legal_action_mask().astype(np.bool_)
    pi = pi.astype(np.float64, copy=False)
    pi[~legal] = 0.0

    if pi.sum() <= 0:
        legal_ids = np.nonzero(legal)[0]
        if legal_ids.size == 0:
            return None
        return int(np.random.choice(legal_ids))

    if temperature <= 1e-6:
        return int(np.argmax(pi))
    pi = pi / float(pi.sum())
    return int(np.random.choice(len(pi), p=pi))


def _apply_human_action(
    state: AppState,
    action_id: int,
) -> Optional[str]:
    if state.env is None:
        return "No game running."
    legal = state.env.legal_action_mask()
    if not legal[int(action_id)]:
        return "Illegal action."
    state.env.step(int(action_id))
    state.last_action = int(action_id)
    state.selected_src = None
    return None


def _maybe_ai_turn(
    state: AppState,
    num_simulations: int,
    c_puct: float,
    num_determinize: int,
    temperature: float,
) -> None:
    if state.env is None or state.model is None:
        return
    if state.env.done:
        return
    if state.env.to_play == state.human_player:
        return
    action_id = _select_ai_action(
        state,
        num_simulations=num_simulations,
        c_puct=c_puct,
        num_determinize=num_determinize,
        temperature=temperature,
    )
    if action_id is None:
        return
    state.env.step(int(action_id))
    state.last_action = int(action_id)
    state.selected_src = None


def load_model_clicked(
    state: AppState,
    checkpoint_path: str,
    device_choice: str,
) -> tuple[AppState, str, str, list[str]]:
    if not checkpoint_path:
        return state, "", "Checkpoint path is required.", _board_labels(state)

    path = Path(checkpoint_path)
    if not path.exists():
        return (
            state,
            "",
            f"Checkpoint not found: {checkpoint_path}",
            _board_labels(state),
        )

    try:
        device = _default_device(device_choice)
        model, _ = load_checkpoint(path)
        model = model.to(device)
        model.eval()
    except Exception as exc:
        return state, "", f"Failed to load model: {exc}", _board_labels(state)

    state.model = model
    state.device = device
    model_info = (
        f"Loaded: {checkpoint_path}\n"
        f"Device: {device}\n"
        f"Max history: {model.cfg.max_history}"
    )
    return state, model_info, "Model loaded. Start a new game.", _board_labels(state)


def new_game_clicked(
    state: AppState,
    human_player: str,
    seed: int,
    num_simulations: int,
    c_puct: float,
    num_determinize: int,
    temperature: float,
) -> tuple[AppState, str, list[str]]:
    if state.model is None:
        return state, "Load a model first.", _board_labels(state)

    state.human_player = 0 if human_player == "Player 0 (first)" else 1
    state.env = BanqiEnv(
        action_space=state.action_space,
        max_history=int(state.model.cfg.max_history),
        seed=int(seed) if seed >= 0 else None,
    )
    state.selected_src = None
    state.last_action = None

    _maybe_ai_turn(
        state,
        num_simulations=num_simulations,
        c_puct=c_puct,
        num_determinize=num_determinize,
        temperature=temperature,
    )

    return state, _status_text(state), _board_labels(state)


def board_click(
    idx: int,
    state: AppState,
    num_simulations: int,
    c_puct: float,
    num_determinize: int,
    temperature: float,
) -> tuple[AppState, str, list[str]]:
    if state.env is None:
        return state, "Start a new game first.", _board_labels(state)
    if state.model is None:
        return state, "Load a model first.", _board_labels(state)
    if state.env.done:
        return state, _status_text(state), _board_labels(state)
    if state.env.to_play != state.human_player:
        return state, "Wait for AI to move.", _board_labels(state)

    action_space = state.action_space

    if state.selected_src is None:
        # If clicking a covered square, attempt flip immediately.
        if state.env.piece[idx] != 0 and not state.env.revealed[idx]:
            action_id = action_space.encode("flip", idx, idx)
            err = _apply_human_action(state, action_id)
            if err:
                return state, err, _board_labels(state)
        else:
            state.selected_src = int(idx)
            return state, _status_text(state), _board_labels(state)
    else:
        src = int(state.selected_src)
        dst = int(idx)
        action_id = action_space.encode("move", src, dst)
        err = _apply_human_action(state, action_id)
        if err:
            state.selected_src = None
            return state, err, _board_labels(state)

    _maybe_ai_turn(
        state,
        num_simulations=num_simulations,
        c_puct=c_puct,
        num_determinize=num_determinize,
        temperature=temperature,
    )

    return state, _status_text(state), _board_labels(state)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Banqi VR - Play") as demo:
        state = gr.State(_init_state())

        with gr.Row():
            with gr.Column(scale=1):
                checkpoint_path = gr.Textbox(
                    label="Checkpoint path",
                    placeholder="/path/to/checkpoint_dir",
                )
                device_choice = gr.Dropdown(
                    choices=["auto", "cpu", "cuda"],
                    value="auto",
                    label="Device",
                )
                load_btn = gr.Button("Load model", variant="primary")
                model_info = gr.Textbox(
                    label="Model info",
                    interactive=False,
                    lines=3,
                )

                human_player = gr.Radio(
                    choices=["Player 0 (first)", "Player 1 (second)"],
                    value="Player 0 (first)",
                    label="Human plays as",
                )
                seed = gr.Number(value=-1, label="Seed (-1 for random)")

                gr.Markdown("### MCTS settings")
                num_simulations = gr.Slider(
                    16, 800, value=200, step=1, label="Simulations"
                )
                c_puct = gr.Slider(0.5, 4.0, value=1.5, step=0.1, label="c_puct")
                num_determinize = gr.Slider(1, 32, value=8, step=1, label="Determinize")
                temperature = gr.Slider(
                    0.0, 2.0, value=0.0, step=0.05, label="Temperature"
                )

                new_game_btn = gr.Button("New game", variant="secondary")

            with gr.Column(scale=2):
                status = gr.Markdown("Load a model to start.")
                board_buttons: list[gr.Button] = []
                rows = 4
                cols = 8
                for r in range(rows):
                    with gr.Row():
                        for c in range(cols):
                            idx = r * cols + c
                            btn = gr.Button("", min_width=70)
                            board_buttons.append(btn)

        load_btn.click(
            load_model_clicked,
            inputs=[state, checkpoint_path, device_choice],
            outputs=[state, model_info, status] + board_buttons,
        )

        new_game_btn.click(
            new_game_clicked,
            inputs=[
                state,
                human_player,
                seed,
                num_simulations,
                c_puct,
                num_determinize,
                temperature,
            ],
            outputs=[state, status] + board_buttons,
        )

        for idx, btn in enumerate(board_buttons):
            btn.click(
                board_click,
                inputs=[
                    gr.State(idx),
                    state,
                    num_simulations,
                    c_puct,
                    num_determinize,
                    temperature,
                ],
                outputs=[state, status] + board_buttons,
            )

    return demo


if __name__ == "__main__":
    build_app().launch()
