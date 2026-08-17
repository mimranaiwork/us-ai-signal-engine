"""
Time-series Transformer multi-horizon model (Phase 2 item 28). Same curated
feature window and multi-head-per-horizon design as the LSTM
(models/lstm/model.py), swapping the encoder for a small Transformer
encoder with learned positional embeddings. `seq_len` is configurable
(the spec calls for testing 30/60/120 — default 60, matching the LSTM, so
the two are directly comparable; pass seq_len= to try others).
"""

from __future__ import annotations

import os

import torch
from torch import nn

from models.sequence_common import BaseSequenceModel, MultiHeadOutput

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "saved", "transformer")


class TransformerMultiHorizon(nn.Module):
    def __init__(self, n_features: int, seq_len: int, hidden_size: int = 64,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=hidden_size * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = MultiHeadOutput(hidden_size)

    def forward(self, x):
        h = self.input_proj(x) + self.pos_embedding[:, : x.size(1), :]
        h = self.encoder(h)
        encoded = h[:, -1, :]  # last-timestep representation, causal enough since
        # nn.TransformerEncoder without a mask attends to all timesteps in the
        # window — the window itself only ever contains data up to T, so this
        # stays leakage-free at the sequence-construction level (see
        # models/sequence_common.py's windowing).
        return self.output(encoded)


class TransformerMultiHorizonModel(BaseSequenceModel):
    ARCH_NAME = "Transformer"

    def _build_net(self, n_features: int) -> nn.Module:
        return TransformerMultiHorizon(
            n_features, seq_len=self.seq_len, hidden_size=self.hidden_size, **self.arch_kwargs
        )

    def save(self, save_dir: str = SAVE_DIR) -> None:
        super().save(save_dir)

    @classmethod
    def load(cls, save_dir: str = SAVE_DIR) -> "TransformerMultiHorizonModel":
        return super().load(save_dir)
