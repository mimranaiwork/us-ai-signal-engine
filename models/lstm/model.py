"""
LSTM multi-horizon model (Phase 2 item 27). Shared LSTM encoder over the
trailing `seq_len` trading sessions (default 60) of curated features, with
one small linear head per horizon — see models/sequence_common.py for the
shared training loop and for why this is a joint multi-task design rather
than seven separate LSTMs.
"""

from __future__ import annotations

import os

from torch import nn

from models.sequence_common import BaseSequenceModel, MultiHeadOutput

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "saved", "lstm")


class LSTMMultiHorizon(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.output = MultiHeadOutput(hidden_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        encoded = h_n[-1]  # (batch, hidden_size) — last layer's final hidden state
        return self.output(encoded)


class LSTMMultiHorizonModel(BaseSequenceModel):
    ARCH_NAME = "LSTM"

    def _build_net(self, n_features: int) -> nn.Module:
        return LSTMMultiHorizon(n_features, hidden_size=self.hidden_size, **self.arch_kwargs)

    def save(self, save_dir: str = SAVE_DIR) -> None:
        super().save(save_dir)

    @classmethod
    def load(cls, save_dir: str = SAVE_DIR) -> "LSTMMultiHorizonModel":
        return super().load(save_dir)
