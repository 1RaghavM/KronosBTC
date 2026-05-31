from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from model import KronosPredictor

logger = logging.getLogger(__name__)

_PRICE_COLS = ["open", "high", "low", "close"]


class _BatchPredictor(Protocol):
    def predict_batch(
        self,
        df_list: list[pd.DataFrame],
        x_timestamp_list: list[pd.Series],
        y_timestamp_list: list[pd.Series],
        pred_len: int,
        T: float = ...,
        top_k: int = ...,
        top_p: float = ...,
        sample_count: int = ...,
        verbose: bool = ...,
    ) -> list[pd.DataFrame]: ...


class KronosPathSampler:
    """Adapter exposing Kronos forecast paths as independent close samples.

    Wraps a :class:`model.KronosPredictor`. The upstream ``predict`` /
    ``predict_batch`` average across ``sample_count`` internally, which
    discards the path distribution we need for a binary probability. To
    recover independent draws *without modifying the fork* (NFR-009), we
    issue ``predict_batch`` with ``sample_count=1`` over ``sample_count``
    copies of the lookback, so each batch element is one independent Monte
    Carlo path. Batches are chunked to bound peak memory.
    """

    def __init__(self, predictor: KronosPredictor | _BatchPredictor, max_batch: int = 256) -> None:
        if max_batch <= 0:
            raise ValueError(f"max_batch must be positive, got {max_batch}")
        self._predictor = predictor
        self._max_batch = max_batch

    def sample_closes(
        self,
        lookback_df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        sample_count: int,
        temperature: float,
        top_p: float,
    ) -> np.ndarray:
        """Draw ``sample_count`` independent terminal-close samples (USD).

        Returns:
            1-D float array of the final-window close from each sampled path.
        """
        pred_len = len(y_timestamp)
        df = lookback_df.reset_index(drop=True)

        closes = np.empty(sample_count, dtype=float)
        filled = 0
        while filled < sample_count:
            batch = min(self._max_batch, sample_count - filled)
            pred_dfs = self._predictor.predict_batch(
                [df] * batch,
                [x_timestamp] * batch,
                [y_timestamp] * batch,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=1,
                verbose=False,
            )
            for j, pdf in enumerate(pred_dfs):
                closes[filled + j] = float(pdf["close"].to_numpy()[-1])
            filled += batch

        return closes


def build_kronos_path_sampler(
    checkpoint: str,
    tokenizer_name: str,
    device: str = "auto",
    max_context: int = 512,
    max_batch: int = 256,
) -> KronosPathSampler:
    """Load Kronos weights from HuggingFace and return a path sampler.

    Args:
        checkpoint: HF id of the Kronos predictor model (e.g. ``NeoQuasar/Kronos-small``).
        tokenizer_name: HF id of the matching tokenizer.
        device: ``auto`` lets ``KronosPredictor`` pick CUDA/MPS/CPU. On Apple
            Silicon this resolves to ``mps``.
        max_context: Lookback clamp (512 for small/base, 2048 for mini).
        max_batch: Max parallel paths per ``predict_batch`` call. Lower this
            (e.g. 64) if MPS runs out of memory on large lookbacks.
    """
    import os

    # Let unsupported ops fall back to CPU instead of erroring on Apple Silicon (MPS).
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
    model = Kronos.from_pretrained(checkpoint)
    predictor = KronosPredictor(
        model,
        tokenizer,
        device=None if device == "auto" else device,
        max_context=max_context,
    )
    logger.info(
        "Loaded Kronos predictor: checkpoint=%s tokenizer=%s device=%s",
        checkpoint,
        tokenizer_name,
        predictor.device,
    )
    return KronosPathSampler(predictor, max_batch=max_batch)
