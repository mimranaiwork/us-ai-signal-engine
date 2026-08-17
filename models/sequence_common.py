"""
Shared sequence-dataset construction for the LSTM and Transformer models
(Phase 2 items 27/28). Both are sequence encoders over the same curated
feature window, differing only in the encoder architecture — so the dataset
plumbing (windowing, multi-horizon label alignment, scaling) lives here once.

Multi-task design: one shared encoder body, seven small linear heads (one per
horizon), trained jointly — this matches the spec's own "Features -> Multi-
Horizon Engine -> 7 heads" diagram (section 30) more directly than training
seven fully separate LSTMs would, and is far cheaper to train.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from calendar_us.trading_calendar import HORIZON_SESSIONS
from models.common import LABEL_TO_IDX, PROCESSED_DIR

# A curated, moderate-size feature set for the sequence models — the full
# ~100-column tree-model feature set would make the input dimension (seq_len
# x n_features) unnecessarily large for a small CPU-trained demo model.
SEQUENCE_FEATURE_COLUMNS = [
    "daily_return", "log_return", "gap_return", "high_low_range",
    "rolling_return_5d", "rolling_return_20d", "rolling_volatility_20d",
    "atr_percent", "volume_change", "dollar_volume_change",
    "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
    "sma20_slope", "sma50_slope", "ema_crossover_10_20",
    "rsi_7", "rsi_14", "rsi_21", "macd", "macd_signal", "macd_histogram",
    "roc_10", "stochastic_k", "williams_r", "cci_20",
    "bollinger_band_width", "volatility_20d", "volatility_60d",
    "volume_ratio", "mfi_14",
    "relative_strength_vs_SPY_5d", "relative_strength_vs_SPY_21d",
    "relative_strength_vs_QQQ_5d",
]

MISSING_LABEL = -1


def _load_symbol(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(os.path.join(PROCESSED_DIR, f"{symbol}_features.csv"), parse_dates=["date"])
    labels = pd.read_csv(os.path.join(PROCESSED_DIR, f"{symbol}_labels.csv"), parse_dates=["date"])
    return features, labels


def _labels_wide(labels: pd.DataFrame) -> pd.DataFrame:
    """long (date, horizon, label) -> wide (date, label_1D, label_1W, ...)."""
    wide = labels.pivot(index="date", columns="horizon", values="label")
    return wide.reindex(columns=list(HORIZON_SESSIONS.keys()))


def build_sequences(
    symbols: list[str], seq_len: int = 60, feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    """
    Returns:
      X: float32 array (N, seq_len, n_features)
      Y: {horizon: int array (N,)} — MISSING_LABEL where that horizon's label
         is unavailable for that sample (still returned so callers can mask it
         out of the loss rather than dropping the sample entirely, since other
         horizons may still be valid for the same window)
      meta: DataFrame (N rows) with symbol/date for traceability (e.g. which
         row backs which prediction, for backtesting/debugging)
    """
    feature_cols = feature_cols or SEQUENCE_FEATURE_COLUMNS
    X_parts, meta_parts = [], []
    Y_parts: dict[str, list[np.ndarray]] = {h: [] for h in HORIZON_SESSIONS}

    for symbol in symbols:
        try:
            features, labels = _load_symbol(symbol)
        except FileNotFoundError:
            continue

        features = features.dropna(subset=feature_cols).reset_index(drop=True)
        if len(features) < seq_len + 1:
            continue

        wide_labels = _labels_wide(labels)
        aligned = features[["date"] + feature_cols].merge(
            wide_labels, on="date", how="left"
        )

        feat_matrix = aligned[feature_cols].to_numpy(dtype=np.float32)
        windows = sliding_window_view(feat_matrix, window_shape=seq_len, axis=0)
        # sliding_window_view puts the window axis last; move it back to axis 1
        windows = np.moveaxis(windows, -1, 1)  # (n_windows, seq_len, n_features)

        # window i (0-indexed) ends at row i + seq_len - 1 of `aligned`
        end_idx = np.arange(seq_len - 1, len(aligned))
        X_parts.append(windows)
        meta_parts.append(pd.DataFrame({
            "symbol": symbol,
            "date": aligned["date"].to_numpy()[end_idx],
        }))
        for horizon in HORIZON_SESSIONS:
            raw = aligned[horizon].to_numpy()[end_idx]
            encoded = np.array(
                [LABEL_TO_IDX.get(v, MISSING_LABEL) for v in raw], dtype=np.int64
            )
            Y_parts[horizon].append(encoded)

    if not X_parts:
        raise RuntimeError("No sequences built — check that Phase 1 processed data exists.")

    X = np.concatenate(X_parts, axis=0)
    Y = {h: np.concatenate(v, axis=0) for h, v in Y_parts.items()}
    meta = pd.concat(meta_parts, ignore_index=True)
    return X, Y, meta


def fit_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std computed over the TRAIN split only (no leakage
    into val/test), across both the sample and timestep axes."""
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def masked_multi_head_loss(logits: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
    loss_fn = nn.CrossEntropyLoss(ignore_index=MISSING_LABEL)
    total = torch.tensor(0.0)
    for horizon in logits:
        total = total + loss_fn(logits[horizon], targets[horizon])
    return total


class MultiHeadOutput(nn.Module):
    """One linear classification head per horizon, sitting on top of any
    shared sequence encoder that reduces (batch, seq_len, n_features) to a
    single (batch, hidden_size) embedding."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.heads = nn.ModuleDict({h: nn.Linear(hidden_size, 3) for h in HORIZON_SESSIONS})

    def forward(self, encoded: torch.Tensor) -> dict[str, torch.Tensor]:
        return {h: head(encoded) for h, head in self.heads.items()}


class BaseSequenceModel:
    """
    Shared fit/predict/save/load for sequence encoders (LSTM, Transformer).
    Subclasses implement `_build_net(n_features) -> nn.Module`, where the
    returned module's forward(x) -> dict[horizon, logits] (encoder + heads
    combined — see LSTMMultiHorizon / TransformerMultiHorizon).
    """

    ARCH_NAME = "base"

    def __init__(self, seq_len: int = 60, hidden_size: int = 64, **arch_kwargs):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.arch_kwargs = arch_kwargs
        self.feature_cols = SEQUENCE_FEATURE_COLUMNS
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.net: nn.Module | None = None

    def _build_net(self, n_features: int) -> nn.Module:
        raise NotImplementedError

    def fit(
        self, X_train: np.ndarray, Y_train: dict[str, np.ndarray],
        X_val: np.ndarray, Y_val: dict[str, np.ndarray],
        mean: np.ndarray, std: np.ndarray,
        epochs: int = 5, batch_size: int = 256, lr: float = 1e-3,
    ) -> list[dict]:
        self.mean, self.std = mean, std
        self.net = self._build_net(n_features=X_train.shape[-1])
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        horizons = list(HORIZON_SESSIONS.keys())
        train_tensors = [torch.from_numpy(X_train)] + [torch.from_numpy(Y_train[h]) for h in horizons]
        loader = DataLoader(TensorDataset(*train_tensors), batch_size=batch_size, shuffle=True)

        X_val_t = torch.from_numpy(X_val)
        Y_val_t = {h: torch.from_numpy(Y_val[h]) for h in horizons}

        history = []
        for epoch in range(epochs):
            self.net.train()
            epoch_loss = 0.0
            for batch in loader:
                xb, *ybs = batch
                yb = dict(zip(horizons, ybs))
                optimizer.zero_grad()
                logits = self.net(xb)
                loss = masked_multi_head_loss(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(loader.dataset)

            self.net.eval()
            with torch.no_grad():
                val_logits = self.net(X_val_t)
                val_loss = masked_multi_head_loss(val_logits, Y_val_t).item()
                accs = {}
                for h in horizons:
                    mask = Y_val_t[h] != MISSING_LABEL
                    if mask.sum() == 0:
                        continue
                    pred = val_logits[h].argmax(dim=1)
                    accs[h] = float((pred[mask] == Y_val_t[h][mask]).float().mean())

            record = {"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss, "val_acc": accs}
            history.append(record)
            print(f"  [{self.ARCH_NAME}] epoch {epoch+1}/{epochs}  train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}")

        return history

    def predict_proba_batch(self, X: np.ndarray) -> dict[str, np.ndarray]:
        X_scaled = (X - self.mean) / self.std
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.from_numpy(X_scaled.astype(np.float32)))
            return {h: torch.softmax(logits[h], dim=1).numpy() for h in logits}

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(save_dir, "weights.pt"))
        meta = {
            "seq_len": self.seq_len, "hidden_size": self.hidden_size, "arch_kwargs": self.arch_kwargs,
            "feature_cols": self.feature_cols,
            "mean": self.mean.tolist(), "std": self.std.tolist(),
        }
        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, save_dir: str) -> "BaseSequenceModel":
        with open(os.path.join(save_dir, "meta.json")) as f:
            meta = json.load(f)
        obj = cls(seq_len=meta["seq_len"], hidden_size=meta["hidden_size"], **meta.get("arch_kwargs", {}))
        obj.feature_cols = meta["feature_cols"]
        obj.mean = np.array(meta["mean"], dtype=np.float32)
        obj.std = np.array(meta["std"], dtype=np.float32)
        obj.net = obj._build_net(n_features=len(obj.feature_cols))
        obj.net.load_state_dict(torch.load(os.path.join(save_dir, "weights.pt")))
        obj.net.eval()
        return obj
