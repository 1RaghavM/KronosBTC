from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strikecast.constants import WINDOW_SECONDS


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with subdirectories."""
    for subdir in ["candles", "pm_markets", "resolution_labels", "reports"]:
        (tmp_path / subdir).mkdir()
    return tmp_path


@pytest.fixture
def sample_candles() -> pd.DataFrame:
    """20 consecutive 5-min BTC candles starting at a grid-aligned timestamp."""
    base_ts = 1_700_000_000
    base_ts = base_ts - (base_ts % WINDOW_SECONDS)
    n = 20
    rng = np.random.RandomState(42)
    prices = 35000.0 + rng.randn(n).cumsum() * 10
    return pd.DataFrame(
        {
            "symbol": "BTC/USD",
            "granularity": WINDOW_SECONDS,
            "window_open_ts": [base_ts + i * WINDOW_SECONDS for i in range(n)],
            "open": prices,
            "high": prices + rng.uniform(5, 20, n),
            "low": prices - rng.uniform(5, 20, n),
            "close": prices + rng.randn(n) * 5,
            "volume": rng.uniform(0.1, 10.0, n),
            "amount": 0.0,
            "source": "coinbase",
        }
    )


@pytest.fixture
def sample_markets() -> pd.DataFrame:
    """5 sample Polymarket market records."""
    base_ts = 1_700_000_000
    base_ts = base_ts - (base_ts % WINDOW_SECONDS)
    return pd.DataFrame(
        {
            "window_open_ts": [base_ts + i * WINDOW_SECONDS for i in range(5)],
            "condition_id": [f"cond_{i}" for i in range(5)],
            "token_id_up": [f"tok_up_{i}" for i in range(5)],
            "token_id_down": [f"tok_down_{i}" for i in range(5)],
            "price_to_beat": [35000.0 + i * 10 for i in range(5)],
            "price_up": [0.52, 0.48, 0.51, 0.49, 0.53],
            "price_down": [0.48, 0.52, 0.49, 0.51, 0.47],
            "captured_ts": [base_ts + i * WINDOW_SECONDS + 60 for i in range(5)],
        }
    )


@pytest.fixture
def sample_labels() -> pd.DataFrame:
    """5 sample resolution labels."""
    base_ts = 1_700_000_000
    base_ts = base_ts - (base_ts % WINDOW_SECONDS)
    return pd.DataFrame(
        {
            "window_open_ts": [base_ts + i * WINDOW_SECONDS for i in range(5)],
            "oracle_close": [35005.0, 34995.0, 35015.0, 34990.0, 35020.0],
            "coinbase_close": [35004.5, 34994.8, 35014.2, 34989.5, 35019.8],
            "outcome_up": [True, False, True, False, True],
        }
    )


@pytest.fixture
def large_candle_series() -> pd.DataFrame:
    """500 consecutive 5-min BTC candles with geometric random-walk prices."""
    base_ts = 1_700_000_000
    base_ts = base_ts - (base_ts % WINDOW_SECONDS)
    n = 500
    rng = np.random.RandomState(42)

    log_returns = rng.normal(0, 0.001, n)
    log_prices = np.log(35000.0) + np.cumsum(log_returns)
    prices = np.exp(log_prices)

    return pd.DataFrame(
        {
            "symbol": "BTC/USD",
            "granularity": WINDOW_SECONDS,
            "window_open_ts": [base_ts + i * WINDOW_SECONDS for i in range(n)],
            "open": prices,
            "high": prices * (1 + rng.uniform(0.0001, 0.001, n)),
            "low": prices * (1 - rng.uniform(0.0001, 0.001, n)),
            "close": prices * (1 + rng.normal(0, 0.0005, n)),
            "volume": rng.uniform(0.1, 10.0, n),
            "amount": 0.0,
            "source": "coinbase",
        }
    )


@pytest.fixture
def large_labels(large_candle_series: pd.DataFrame) -> pd.DataFrame:
    """Resolution labels for the large candle series (close vs open)."""
    df = large_candle_series
    return pd.DataFrame(
        {
            "window_open_ts": df["window_open_ts"],
            "oracle_close": df["close"],
            "coinbase_close": df["close"],
            "outcome_up": df["close"] > df["open"],
        }
    )
