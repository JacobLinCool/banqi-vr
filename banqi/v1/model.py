"""
Banqi SOTA-ish PyTorch model:
Transformer trunk + belief/state encoder + multi-head outputs:
  - self policy logits
  - opponent policy logits
  - value
  - belief logits (per-square distribution)

Input is token sequence: [CLS] + [TURN] + 32 board tokens + 32 belief tokens + H history tokens.

This file only defines the neural network. Self-play/MCTS lives elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


@dataclass
class ModelConfig:
    # model size
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1

    # game sizes
    board_size: int = 32
    belief_dim: int = 15
    action_dim: int = 352
    max_history: int = 16

    # belief tokenization
    belief_tokens: int = 32  # 32 per-square belief tokens by default
    use_reduced_belief: bool = False  # if True, use 1 global belief token

    # vocab sizes
    board_vocab: int = 17  # 0..16 inclusive (PAD, empty, covered, 14 pieces)
    turn_vocab: int = 3  # 0=red,1=black,2=unknown

    # misc
    use_tanh_value: bool = True
    draw_plies: int = 40
    max_plies: int = 512


@dataclass
class ModelOutputs:
    policy_self_logits: torch.Tensor  # [B, A]
    policy_opp_logits: torch.Tensor  # [B, A]
    value: torch.Tensor  # [B]
    belief_logits: torch.Tensor  # [B, 32, belief_dim]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BanqiTransformer(PyTorchModelHubMixin, nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if isinstance(cfg, dict):
            cfg = ModelConfig(**cfg)
        self.cfg = cfg
        # Hugging Face convention
        self.config = cfg

        if self.cfg.use_reduced_belief:
            self.cfg.belief_tokens = 1

        # embeddings
        self.board_embed = nn.Embedding(cfg.board_vocab, cfg.d_model)
        self.turn_embed = nn.Embedding(cfg.turn_vocab, cfg.d_model)
        self.no_progress_embed = nn.Embedding(cfg.draw_plies + 1, cfg.d_model)
        self.plies_embed = nn.Embedding(cfg.max_plies + 1, cfg.d_model)
        self.action_embed = nn.Embedding(
            cfg.action_dim + 1, cfg.d_model
        )  # 0=PAD, 1..A = action_id+1

        self.belief_proj = nn.Linear(cfg.belief_dim, cfg.d_model)
        self.belief_global_proj = nn.Linear(cfg.belief_dim * 2, cfg.d_model)

        # learned special tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.belief_mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

        # position + type embeddings
        self.max_seq_len = 1 + 1 + cfg.board_size + cfg.belief_tokens + cfg.max_history
        self.pos_embed = nn.Embedding(self.max_seq_len, cfg.d_model)
        # token type ids: 0=CLS,1=TURN,2=BOARD,3=BELIEF,4=HIST
        self.type_embed = nn.Embedding(5, cfg.d_model)

        self.drop = nn.Dropout(cfg.dropout)

        # transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

        # heads
        self.policy_self = nn.Linear(cfg.d_model, cfg.action_dim)
        self.policy_opp = nn.Linear(cfg.d_model, cfg.action_dim)

        self.value_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

        self.belief_head = nn.Linear(cfg.d_model, cfg.belief_dim)

        # init
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.belief_mask_token, mean=0.0, std=0.02)

    @torch.no_grad()
    def parameter_count(self) -> int:
        return count_parameters(self)

    def forward(
        self,
        board_tokens: torch.Tensor,  # [B, 32] long
        belief: torch.Tensor,  # [B, 32, belief_dim] float
        history_actions: torch.Tensor,  # [B, H] long, -1 for PAD
        to_play_color: torch.Tensor,  # [B] long in {0,1,2}
        plies: Optional[torch.Tensor] = None,  # [B] long, total plies
        no_progress_plies: Optional[torch.Tensor] = None,  # [B] long
        belief_mask: Optional[
            torch.Tensor
        ] = None,  # [B, 32] bool, True means masked/unknown
    ) -> ModelOutputs:
        """
        Returns logits/values. Masking of illegal actions is done in loss/search code.
        """
        B, S = board_tokens.shape
        assert S == self.cfg.board_size, (S, self.cfg.board_size)
        assert belief.shape[:2] == (B, self.cfg.board_size), belief.shape
        H = history_actions.shape[1]

        if H > self.cfg.max_history:
            raise ValueError(
                f"history length {H} exceeds cfg.max_history={self.cfg.max_history}"
            )

        # Build token embeddings
        cls = self.cls_token.expand(B, 1, -1)  # [B,1,D]

        if plies is None:
            plies = torch.zeros(B, device=board_tokens.device, dtype=torch.long)
        if no_progress_plies is None:
            no_progress_plies = torch.zeros(
                B, device=board_tokens.device, dtype=torch.long
            )
        plies_idx = torch.clamp(plies, min=0, max=self.cfg.max_plies)
        no_prog_idx = torch.clamp(no_progress_plies, min=0, max=self.cfg.draw_plies)

        turn = (
            self.turn_embed(to_play_color)
            + self.plies_embed(plies_idx)
            + self.no_progress_embed(no_prog_idx)
        ).unsqueeze(1)  # [B,1,D]
        board = self.board_embed(board_tokens)  # [B,32,D]

        if self.cfg.use_reduced_belief:
            if belief_mask is not None:
                keep = (~belief_mask).to(dtype=belief.dtype).unsqueeze(-1)
                denom = keep.sum(dim=1).clamp_min(1.0)
                mean_belief = (belief * keep).sum(dim=1) / denom
                count_belief = (belief * keep).sum(dim=1)
            else:
                mean_belief = belief.mean(dim=1)
                count_belief = belief.sum(dim=1)
            global_feat = torch.cat([mean_belief, count_belief], dim=-1)
            belief_emb = self.belief_global_proj(global_feat).unsqueeze(1)  # [B,1,D]
        else:
            belief_emb = self.belief_proj(belief)  # [B,32,D]
            if belief_mask is not None:
                # replace masked squares by a learned token (denoising objective can use this)
                m = belief_mask.unsqueeze(-1).to(dtype=torch.bool)
                belief_emb = torch.where(
                    m,
                    self.belief_mask_token.expand(B, self.cfg.board_size, -1),
                    belief_emb,
                )

        # history: map -1 -> 0 PAD, else action_id+1
        if H > 0:
            hist_idx = history_actions + 1
            hist_idx = torch.clamp(hist_idx, min=0, max=self.cfg.action_dim)
            hist = self.action_embed(hist_idx)  # [B,H,D]
        else:
            hist = board.new_zeros((B, 0, self.cfg.d_model), dtype=board.dtype)

        x = torch.cat([cls, turn, board, belief_emb, hist], dim=1)  # [B, L, D]
        L = x.shape[1]

        if L > self.max_seq_len:
            raise ValueError(f"seq_len={L} exceeds max_seq_len={self.max_seq_len}")

        # position ids
        pos_ids = torch.arange(L, device=x.device)
        pos = self.pos_embed(pos_ids).unsqueeze(0)  # [1,L,D]

        # type ids
        # layout: 0 CLS, 1 TURN, 2..2+31 BOARD, next belief tokens, rest HIST
        type_ids = torch.empty(L, device=x.device, dtype=torch.long)
        type_ids[0] = 0
        type_ids[1] = 1
        board_start = 2
        belief_start = 2 + self.cfg.board_size
        hist_start = belief_start + self.cfg.belief_tokens
        type_ids[board_start:belief_start] = 2
        type_ids[belief_start:hist_start] = 3
        if hist_start < L:
            type_ids[hist_start:] = 4
        type = self.type_embed(type_ids).unsqueeze(0)  # [1,L,D]

        x = x + pos + type
        x = self.drop(x)

        # key padding mask (True for PAD tokens)
        if H > 0:
            hist_pad = history_actions < 0  # [B,H]
        else:
            hist_pad = torch.zeros((B, 0), device=x.device, dtype=torch.bool)
        pad_mask = torch.cat(
            [
                torch.zeros(
                    (B, 1 + 1 + self.cfg.board_size + self.cfg.belief_tokens),
                    device=x.device,
                    dtype=torch.bool,
                ),
                hist_pad,
            ],
            dim=1,
        )  # [B,L]
        assert pad_mask.shape[1] == L

        y = self.encoder(x, src_key_padding_mask=pad_mask)  # [B,L,D]

        cls_out = y[:, 0, :]  # [B,D]

        policy_self_logits = self.policy_self(cls_out)
        policy_opp_logits = self.policy_opp(cls_out)

        value = self.value_head(cls_out).squeeze(-1)
        if self.cfg.use_tanh_value:
            value = torch.tanh(value)

        if self.cfg.use_reduced_belief:
            belief_out = y[:, board_start:belief_start, :]  # [B,32,D]
        else:
            belief_out = y[:, belief_start:hist_start, :]  # [B,32,D]
        belief_logits = self.belief_head(belief_out)  # [B,32,belief_dim]

        return ModelOutputs(
            policy_self_logits=policy_self_logits,
            policy_opp_logits=policy_opp_logits,
            value=value,
            belief_logits=belief_logits,
        )

    @staticmethod
    def masked_log_softmax(
        logits: torch.Tensor, mask: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        """
        logits: [..., A]
        mask:  same shape, bool, True for legal, False for illegal
        returns log_softmax over legal actions with illegal entries = -inf
        """
        neg_inf = torch.finfo(logits.dtype).min
        masked = torch.where(
            mask,
            logits,
            torch.tensor(neg_inf, device=logits.device, dtype=logits.dtype),
        )
        return F.log_softmax(masked, dim=dim)
