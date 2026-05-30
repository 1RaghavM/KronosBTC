# Strikecast Phase 0: Data Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data ingestion layer — Coinbase candles, Polymarket market metadata, and resolution labels — stored in Parquet, queryable via DuckDB, with gap detection and grid-alignment enforcement.

**Architecture:** `strikecast/` is a new top-level Python package inside the existing Kronos repo. It imports from `model/` via an adapter (deferred to Phase 2). The data layer consists of a `CandleSource` protocol with a Coinbase implementation via `ccxt`, a read-only Polymarket client via `httpx`, and a `DataStore` class managing three Parquet tables. All timestamps are Unix epoch seconds aligned to a 300-second grid.

**Tech Stack:** Python 3.10+, ccxt (Coinbase), httpx (Polymarket REST), pyarrow (Parquet), duckdb (queries), pydantic + PyYAML (config), pytest (tests), black + ruff + mypy (linting).

**Design spec:** `docs/superpowers/specs/2026-05-30-strikecast-design.md`
**Source specs:** `specs/steering.md`, `specs/requirements.md`, `specs/design.md`, `specs/quality.md`

---

## File map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `strikecast/__init__.py` | Package marker |
| Create | `strikecast/__main__.py` | Entry point for `python -m strikecast` |
| Create | `strikecast/constants.py` | Named constants (WINDOW_SECONDS, etc.) |
| Create | `strikecast/config.py` | Pydantic config model + YAML loader |
| Create | `strikecast/data/__init__.py` | Subpackage marker |
| Create | `strikecast/data/store.py` | Parquet read/write, grid alignment, dedup, gap detection, DuckDB queries |
| Create | `strikecast/data/candle_source.py` | CandleSource Protocol + CoinbaseSource (ccxt) |
| Create | `strikecast/data/paginator.py` | Paginated candle fetching with rate limiting |
| Create | `strikecast/data/polymarket_read.py` | Read-only Polymarket Gamma/CLOB client |
| Create | `strikecast/cli.py` | CLI with `--phase data` support |
| Create | `config/default.yaml` | Default config file |
| Create | `requirements-strikecast.txt` | Strikecast-specific dependencies |
| Create | `tests/strikecast/__init__.py` | Test package marker |
| Create | `tests/strikecast/conftest.py` | Shared fixtures |
| Create | `tests/strikecast/test_config.py` | Config loading and validation tests |
| Create | `tests/strikecast/test_store.py` | Store CRUD, dedup, grid alignment, gap detection tests |
| Create | `tests/strikecast/test_candle_source.py` | CoinbaseSource tests with mocked ccxt |
| Create | `tests/strikecast/test_paginator.py` | Paginator tests |
| Create | `tests/strikecast/test_polymarket_read.py` | Polymarket client tests with mocked HTTP |
| Create | `tests/strikecast/test_no_order_path.py` | NFR-001: import guard (non-negotiable) |
| Create | `tests/strikecast/test_grid_alignment.py` | FR-007: grid alignment (non-negotiable) |
| Create | `tests/strikecast/test_cli.py` | CLI integration test |
| Modify | `.gitignore` | Add `data/` directory |

---

### Task 1: Project scaffolding & dependencies

**Files:**
- Create: `strikecast/__init__.py`, `strikecast/__main__.py`, `strikecast/constants.py`
- Create: `strikecast/data/__init__.py`, `strikecast/estimators/__init__.py`, `strikecast/calibration/__init__.py`, `strikecast/eval/__init__.py`
- Create: `tests/strikecast/__init__.py`, `tests/strikecast/conftest.py`
- Create: `requirements-strikecast.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Create the directory structure**

```bash
mkdir -p strikecast/data strikecast/estimators strikecast/calibration strikecast/eval
mkdir -p tests/strikecast
mkdir -p config
```

- [ ] **Step 2: Create package marker files**

`strikecast/__init__.py`:
```python
"""Strikecast: calibrated probability engine for 5-min BTC binary outcomes."""
```

`strikecast/data/__init__.py`:
```python
"""Data ingestion and storage layer."""
```

`strikecast/estimators/__init__.py`:
```python
"""Probability estimators."""
```

`strikecast/calibration/__init__.py`:
```python
"""Post-hoc probability calibration."""
```

`strikecast/eval/__init__.py`:
```python
"""Evaluation, scoring, and reporting."""
```

`tests/strikecast/__init__.py`:
```python
```

- [ ] **Step 3: Create constants module**

`strikecast/constants.py`:
```python
WINDOW_SECONDS: int = 300
DEFAULT_SAMPLE_COUNT: int = 1000
DEFAULT_TEMPERATURE: float = 1.0
DEFAULT_TOP_P: float = 0.9
ROUND_TRIP_COST: float = 0.02
DEFAULT_GARCH_LOOKBACK: int = 2016
CANDLE_COLUMNS: list[str] = [
    "symbol",
    "granularity",
    "window_open_ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
]
MARKET_COLUMNS: list[str] = [
    "window_open_ts",
    "condition_id",
    "token_id_up",
    "token_id_down",
    "price_to_beat",
    "price_up",
    "price_down",
    "captured_ts",
]
LABEL_COLUMNS: list[str] = [
    "window_open_ts",
    "oracle_close",
    "coinbase_close",
    "outcome_up",
]
```

- [ ] **Step 4: Create `__main__.py`**

`strikecast/__main__.py`:
```python
from strikecast.cli import main

main()
```

- [ ] **Step 5: Create requirements file**

`requirements-strikecast.txt`:
```
# Core
ccxt>=4.0.0
httpx>=0.27.0
pyarrow>=15.0.0
duckdb>=0.10.0
pydantic>=2.0.0
pyyaml>=6.0.0
tqdm>=4.60.0

# Already in Kronos requirements.txt
# numpy, pandas, torch, matplotlib

# Testing
pytest>=8.0.0
pytest-cov>=5.0.0

# Linting (dev)
black>=24.0.0
ruff>=0.4.0
mypy>=1.10.0
```

- [ ] **Step 6: Create shared test fixtures**

`tests/strikecast/conftest.py`:
```python
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
```

- [ ] **Step 7: Update .gitignore**

Add to the end of `.gitignore`:
```
# Strikecast runtime data
data/
```

- [ ] **Step 8: Install dependencies and commit**

```bash
pip install -r requirements-strikecast.txt
git add strikecast/ tests/strikecast/ config/ requirements-strikecast.txt .gitignore
git commit -m "feat(strikecast): project scaffolding with constants and test fixtures"
```

---

### Task 2: Config system

**Files:**
- Create: `strikecast/config.py`
- Create: `config/default.yaml`
- Create: `tests/strikecast/test_config.py`

- [ ] **Step 1: Write the failing tests**

`tests/strikecast/test_config.py`:
```python
from pathlib import Path

import pytest
import yaml


def test_load_config_from_yaml(tmp_path: Path) -> None:
    from strikecast.config import StrikecastConfig, load_config

    config_data = {
        "seed": 123,
        "symbol": "BTC/USD",
        "granularity": 300,
        "data": {
            "source": "coinbase",
            "start": "2025-12-01",
            "end": "2026-05-30",
            "data_dir": "data/",
            "rate_limit_req_per_sec": 10,
        },
    }
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml.dump(config_data))

    cfg = load_config(config_file)
    assert isinstance(cfg, StrikecastConfig)
    assert cfg.seed == 123
    assert cfg.symbol == "BTC/USD"
    assert cfg.data.source == "coinbase"
    assert cfg.data.start == "2025-12-01"
    assert cfg.data.rate_limit_req_per_sec == 10


def test_load_config_uses_defaults(tmp_path: Path) -> None:
    from strikecast.config import load_config

    config_file = tmp_path / "minimal.yaml"
    config_file.write_text(yaml.dump({}))

    cfg = load_config(config_file)
    assert cfg.seed == 42
    assert cfg.granularity == 300
    assert cfg.estimators.sample_count == 1000
    assert cfg.eval.train_frac == 0.60


def test_config_rejects_negative_granularity(tmp_path: Path) -> None:
    from strikecast.config import load_config

    config_file = tmp_path / "bad.yaml"
    config_file.write_text(yaml.dump({"granularity": -1}))

    with pytest.raises(ValueError, match="granularity"):
        load_config(config_file)


def test_config_rejects_invalid_split_fracs(tmp_path: Path) -> None:
    from strikecast.config import load_config

    config_file = tmp_path / "bad.yaml"
    config_file.write_text(
        yaml.dump({"eval": {"train_frac": 0.8, "val_frac": 0.5}})
    )

    with pytest.raises(ValueError, match="sum"):
        load_config(config_file)


def test_config_rejects_nonpositive_sample_count(tmp_path: Path) -> None:
    from strikecast.config import load_config

    config_file = tmp_path / "bad.yaml"
    config_file.write_text(yaml.dump({"estimators": {"sample_count": 0}}))

    with pytest.raises(ValueError, match="sample_count"):
        load_config(config_file)


def test_default_yaml_loads_successfully() -> None:
    from strikecast.config import load_config

    cfg = load_config("config/default.yaml")
    assert cfg.seed == 42
    assert cfg.data.source == "coinbase"
    assert cfg.model.checkpoint == "NeoQuasar/Kronos-small"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/strikecast/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'strikecast.config'`

- [ ] **Step 3: Implement config module**

`strikecast/config.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator


class DataConfig(BaseModel):
    source: str = "coinbase"
    start: str = "2025-12-01"
    end: str = "2026-05-30"
    data_dir: str = "data/"
    rate_limit_req_per_sec: int = 10


class PolymarketConfig(BaseModel):
    enabled: bool = True
    mode: Literal["historical"] = "historical"


class ModelConfig(BaseModel):
    checkpoint: str = "NeoQuasar/Kronos-small"
    tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base"
    device: str = "auto"
    max_context: int = 512


class EstimatorsConfig(BaseModel):
    sample_count: int = 1000
    temperature: float = 1.0
    top_p: float = 0.9
    garch_lookback: int = 2016

    @model_validator(mode="after")
    def validate_sample_count(self) -> EstimatorsConfig:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        return self


class CalibrationConfig(BaseModel):
    method: Literal["isotonic", "platt"] = "isotonic"


class EvalConfig(BaseModel):
    train_frac: float = 0.60
    val_frac: float = 0.20
    purge_windows: int = 1
    embargo_windows: int = 1
    edge_threshold: float = 0.02
    bootstrap_samples: int = 10000
    moneyness_near_threshold: float = 0.001
    moneyness_far_threshold: float = 0.01

    @model_validator(mode="after")
    def validate_split_fracs(self) -> EvalConfig:
        total = self.train_frac + self.val_frac
        if total >= 1.0:
            raise ValueError(
                f"train_frac + val_frac must sum to < 1.0, got {total}"
            )
        return self


class StrikecastConfig(BaseModel):
    seed: int = 42
    symbol: str = "BTC/USD"
    granularity: int = 300
    data: DataConfig = DataConfig()
    polymarket: PolymarketConfig = PolymarketConfig()
    model: ModelConfig = ModelConfig()
    estimators: EstimatorsConfig = EstimatorsConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    eval: EvalConfig = EvalConfig()

    @model_validator(mode="after")
    def validate_granularity(self) -> StrikecastConfig:
        if self.granularity <= 0:
            raise ValueError("granularity must be positive")
        return self


def load_config(path: str | Path) -> StrikecastConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return StrikecastConfig(**raw)
```

- [ ] **Step 4: Create the default config YAML**

`config/default.yaml`:
```yaml
seed: 42
symbol: "BTC/USD"
granularity: 300

data:
  source: "coinbase"
  start: "2025-12-01"
  end: "2026-05-30"
  data_dir: "data/"
  rate_limit_req_per_sec: 10

polymarket:
  enabled: true
  mode: "historical"

model:
  checkpoint: "NeoQuasar/Kronos-small"
  tokenizer: "NeoQuasar/Kronos-Tokenizer-base"
  device: "auto"
  max_context: 512

estimators:
  sample_count: 1000
  temperature: 1.0
  top_p: 0.9
  garch_lookback: 2016

calibration:
  method: "isotonic"

eval:
  train_frac: 0.60
  val_frac: 0.20
  purge_windows: 1
  embargo_windows: 1
  edge_threshold: 0.02
  bootstrap_samples: 10000
  moneyness_near_threshold: 0.001
  moneyness_far_threshold: 0.01
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/strikecast/test_config.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add strikecast/config.py config/default.yaml tests/strikecast/test_config.py
git commit -m "feat(strikecast): config system with Pydantic validation and default YAML"
```

---

### Task 3: Candle store (write, read, grid alignment, dedup)

**Files:**
- Create: `strikecast/data/store.py`
- Create: `tests/strikecast/test_store.py`

- [ ] **Step 1: Write the failing tests**

`tests/strikecast/test_store.py`:
```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strikecast.constants import WINDOW_SECONDS


class TestCandleStore:
    def test_append_and_read_round_trip(
        self, tmp_data_dir: Path, sample_candles: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_candles(sample_candles)
        result = store.read_candles()

        assert len(result) == len(sample_candles)
        assert list(result.columns) == list(sample_candles.columns)
        pd.testing.assert_frame_equal(
            result.sort_values("window_open_ts").reset_index(drop=True),
            sample_candles.sort_values("window_open_ts").reset_index(drop=True),
        )

    def test_rejects_off_grid_timestamps(
        self, tmp_data_dir: Path, sample_candles: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore, GridAlignmentError

        bad = sample_candles.copy()
        bad.loc[0, "window_open_ts"] = bad.loc[0, "window_open_ts"] + 1

        store = DataStore(tmp_data_dir)
        with pytest.raises(GridAlignmentError):
            store.append_candles(bad)

    def test_deduplicates_on_rewrite(
        self, tmp_data_dir: Path, sample_candles: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_candles(sample_candles)
        store.append_candles(sample_candles)
        result = store.read_candles()

        assert len(result) == len(sample_candles)

    def test_append_extends_existing_data(
        self, tmp_data_dir: Path, sample_candles: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        first_half = sample_candles.iloc[:10].copy()
        second_half = sample_candles.iloc[10:].copy()

        store = DataStore(tmp_data_dir)
        store.append_candles(first_half)
        store.append_candles(second_half)
        result = store.read_candles()

        assert len(result) == len(sample_candles)

    def test_rejects_missing_columns(self, tmp_data_dir: Path) -> None:
        from strikecast.data.store import DataStore

        bad = pd.DataFrame({"window_open_ts": [1_700_000_000], "open": [100.0]})
        store = DataStore(tmp_data_dir)
        with pytest.raises(ValueError, match="Missing columns"):
            store.append_candles(bad)

    def test_read_empty_returns_empty_dataframe(self, tmp_data_dir: Path) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        result = store.read_candles()
        assert len(result) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/strikecast/test_store.py::TestCandleStore -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'strikecast.data.store'`

- [ ] **Step 3: Implement the data store**

`strikecast/data/store.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from strikecast.constants import (
    CANDLE_COLUMNS,
    LABEL_COLUMNS,
    MARKET_COLUMNS,
    WINDOW_SECONDS,
)


class GridAlignmentError(Exception):
    pass


TableName = Literal["candles", "pm_markets", "resolution_labels"]

_SCHEMAS: dict[TableName, list[str]] = {
    "candles": CANDLE_COLUMNS,
    "pm_markets": MARKET_COLUMNS,
    "resolution_labels": LABEL_COLUMNS,
}

_DEDUP_KEYS: dict[TableName, list[str]] = {
    "candles": ["symbol", "granularity", "window_open_ts"],
    "pm_markets": ["window_open_ts"],
    "resolution_labels": ["window_open_ts"],
}


class DataStore:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    def _table_path(self, table: TableName) -> Path:
        return self.data_dir / table / f"{table}.parquet"

    def _validate_grid(self, df: pd.DataFrame) -> None:
        offgrid = df["window_open_ts"] % WINDOW_SECONDS != 0
        if offgrid.any():
            bad_ts = df.loc[offgrid, "window_open_ts"].tolist()
            raise GridAlignmentError(
                f"Timestamps not aligned to {WINDOW_SECONDS}s grid: {bad_ts[:5]}"
            )

    def _validate_columns(self, df: pd.DataFrame, table: TableName) -> None:
        expected = set(_SCHEMAS[table])
        actual = set(df.columns)
        missing = expected - actual
        if missing:
            raise ValueError(f"Missing columns for {table}: {missing}")

    def _write(self, df: pd.DataFrame, table: TableName) -> None:
        self._validate_columns(df, table)
        self._validate_grid(df)

        path = self._table_path(table)
        df = df[_SCHEMAS[table]].copy()

        if path.exists():
            existing = pq.read_table(path).to_pandas()
            df = pd.concat([existing, df], ignore_index=True)

        dedup_keys = _DEDUP_KEYS[table]
        df = df.drop_duplicates(subset=dedup_keys, keep="last")
        df = df.sort_values("window_open_ts").reset_index(drop=True)

        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

    def _read(self, table: TableName) -> pd.DataFrame:
        path = self._table_path(table)
        if not path.exists():
            return pd.DataFrame(columns=_SCHEMAS[table])
        return pq.read_table(path).to_pandas()

    def append_candles(self, df: pd.DataFrame) -> None:
        self._write(df, "candles")

    def read_candles(self) -> pd.DataFrame:
        return self._read("candles")

    def append_markets(self, df: pd.DataFrame) -> None:
        self._write(df, "pm_markets")

    def read_markets(self) -> pd.DataFrame:
        return self._read("pm_markets")

    def append_labels(self, df: pd.DataFrame) -> None:
        self._write(df, "resolution_labels")

    def read_labels(self) -> pd.DataFrame:
        return self._read("resolution_labels")

    def query(self, sql: str) -> pd.DataFrame:
        import duckdb

        conn = duckdb.connect()
        for table in _SCHEMAS:
            path = self._table_path(table)
            if path.exists():
                conn.execute(
                    f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path}')"
                )
        return conn.execute(sql).fetchdf()

    def detect_gaps(
        self, symbol: str = "BTC/USD", granularity: int = WINDOW_SECONDS
    ) -> pd.DataFrame:
        candles = self.read_candles()
        if candles.empty:
            return pd.DataFrame(columns=["gap_start_ts", "gap_end_ts", "missing_count"])

        subset = candles[
            (candles["symbol"] == symbol) & (candles["granularity"] == granularity)
        ]
        if subset.empty:
            return pd.DataFrame(columns=["gap_start_ts", "gap_end_ts", "missing_count"])

        ts = subset["window_open_ts"].sort_values().values
        ts_min, ts_max = int(ts[0]), int(ts[-1])

        expected = set(range(ts_min, ts_max + granularity, granularity))
        actual = set(int(t) for t in ts)
        missing = sorted(expected - actual)

        if not missing:
            return pd.DataFrame(columns=["gap_start_ts", "gap_end_ts", "missing_count"])

        gaps: list[dict[str, int]] = []
        gap_start = missing[0]
        prev = missing[0]
        count = 1

        for m in missing[1:]:
            if m == prev + granularity:
                prev = m
                count += 1
            else:
                gaps.append(
                    {"gap_start_ts": gap_start, "gap_end_ts": prev, "missing_count": count}
                )
                gap_start = m
                prev = m
                count = 1
        gaps.append(
            {"gap_start_ts": gap_start, "gap_end_ts": prev, "missing_count": count}
        )

        return pd.DataFrame(gaps)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/strikecast/test_store.py::TestCandleStore -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add strikecast/data/store.py tests/strikecast/test_store.py
git commit -m "feat(strikecast): candle store with Parquet persistence, grid alignment, and dedup"
```

---

### Task 4: Gap detection & data quality

**Files:**
- Modify: `tests/strikecast/test_store.py` (add gap detection tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/strikecast/test_store.py`:
```python
class TestGapDetection:
    def test_no_gaps_returns_empty(
        self, tmp_data_dir: Path, sample_candles: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_candles(sample_candles)
        gaps = store.detect_gaps()
        assert len(gaps) == 0

    def test_detects_single_gap(
        self, tmp_data_dir: Path, sample_candles: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        gapped = sample_candles.drop(index=[5, 6, 7]).reset_index(drop=True)
        store = DataStore(tmp_data_dir)
        store.append_candles(gapped)
        gaps = store.detect_gaps()

        assert len(gaps) == 1
        assert gaps.iloc[0]["missing_count"] == 3

    def test_detects_multiple_gaps(
        self, tmp_data_dir: Path, sample_candles: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        gapped = sample_candles.drop(index=[3, 10, 11]).reset_index(drop=True)
        store = DataStore(tmp_data_dir)
        store.append_candles(gapped)
        gaps = store.detect_gaps()

        assert len(gaps) == 2
        assert gaps.iloc[0]["missing_count"] == 1
        assert gaps.iloc[1]["missing_count"] == 2

    def test_empty_store_returns_empty_gaps(self, tmp_data_dir: Path) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        gaps = store.detect_gaps()
        assert len(gaps) == 0
```

- [ ] **Step 2: Run tests to verify they pass**

The `detect_gaps` method was already implemented in Task 3's `store.py`.

```bash
pytest tests/strikecast/test_store.py::TestGapDetection -v
```

Expected: all 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/strikecast/test_store.py
git commit -m "test(strikecast): gap detection tests for data quality reporting"
```

---

### Task 5: Market & label stores

**Files:**
- Modify: `tests/strikecast/test_store.py` (add market + label tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/strikecast/test_store.py`:
```python
class TestMarketStore:
    def test_append_and_read_round_trip(
        self, tmp_data_dir: Path, sample_markets: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_markets(sample_markets)
        result = store.read_markets()

        assert len(result) == len(sample_markets)
        pd.testing.assert_frame_equal(
            result.sort_values("window_open_ts").reset_index(drop=True),
            sample_markets.sort_values("window_open_ts").reset_index(drop=True),
        )

    def test_deduplicates_on_rewrite(
        self, tmp_data_dir: Path, sample_markets: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_markets(sample_markets)
        store.append_markets(sample_markets)
        result = store.read_markets()
        assert len(result) == len(sample_markets)


class TestLabelStore:
    def test_append_and_read_round_trip(
        self, tmp_data_dir: Path, sample_labels: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_labels(sample_labels)
        result = store.read_labels()

        assert len(result) == len(sample_labels)
        pd.testing.assert_frame_equal(
            result.sort_values("window_open_ts").reset_index(drop=True),
            sample_labels.sort_values("window_open_ts").reset_index(drop=True),
        )

    def test_rejects_off_grid_label(
        self, tmp_data_dir: Path, sample_labels: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore, GridAlignmentError

        bad = sample_labels.copy()
        bad.loc[0, "window_open_ts"] = bad.loc[0, "window_open_ts"] + 7

        store = DataStore(tmp_data_dir)
        with pytest.raises(GridAlignmentError):
            store.append_labels(bad)


class TestDuckDBQuery:
    def test_query_candles(
        self, tmp_data_dir: Path, sample_candles: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_candles(sample_candles)
        result = store.query("SELECT count(*) AS n FROM candles")
        assert result.iloc[0]["n"] == len(sample_candles)

    def test_query_join_candles_and_labels(
        self,
        tmp_data_dir: Path,
        sample_candles: pd.DataFrame,
        sample_labels: pd.DataFrame,
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_candles(sample_candles)
        store.append_labels(sample_labels)

        result = store.query(
            """
            SELECT c.window_open_ts, c.close, l.oracle_close
            FROM candles c
            JOIN resolution_labels l ON c.window_open_ts = l.window_open_ts
            """
        )
        assert len(result) == len(sample_labels)
```

- [ ] **Step 2: Run tests to verify they pass**

The `append_markets`, `read_markets`, `append_labels`, `read_labels`, and `query` methods were already implemented in Task 3.

```bash
pytest tests/strikecast/test_store.py -v
```

Expected: all tests PASS (6 candle + 4 gap + 4 market/label + 2 DuckDB = 16 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/strikecast/test_store.py
git commit -m "test(strikecast): market store, label store, and DuckDB query tests"
```

---

### Task 6: CoinbaseSource + paginator

**Files:**
- Create: `strikecast/data/candle_source.py`
- Create: `strikecast/data/paginator.py`
- Create: `tests/strikecast/test_candle_source.py`
- Create: `tests/strikecast/test_paginator.py`

- [ ] **Step 1: Write the failing tests for CoinbaseSource**

`tests/strikecast/test_candle_source.py`:
```python
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from strikecast.constants import WINDOW_SECONDS


def _make_ohlcv_response(base_ts: int, n: int = 5) -> list[list]:
    """Create a fake ccxt ohlcv response (list of [ts_ms, o, h, l, c, v])."""
    return [
        [
            (base_ts + i * WINDOW_SECONDS) * 1000,
            35000.0 + i,
            35010.0 + i,
            34990.0 + i,
            35005.0 + i,
            1.5 + i * 0.1,
        ]
        for i in range(n)
    ]


class TestCoinbaseSource:
    def test_fetch_returns_correct_schema(self) -> None:
        from strikecast.data.candle_source import CoinbaseSource

        base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = _make_ohlcv_response(base_ts, 5)

        source = CoinbaseSource(exchange=mock_exchange)
        df = source.fetch("BTC/USD", WINDOW_SECONDS, base_ts, base_ts + 5 * WINDOW_SECONDS)

        assert len(df) == 5
        assert list(df.columns) == [
            "symbol", "granularity", "window_open_ts",
            "open", "high", "low", "close", "volume", "amount", "source",
        ]
        assert (df["symbol"] == "BTC/USD").all()
        assert (df["source"] == "coinbase").all()
        assert (df["window_open_ts"] % WINDOW_SECONDS == 0).all()

    def test_fetch_filters_to_requested_range(self) -> None:
        from strikecast.data.candle_source import CoinbaseSource

        base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = _make_ohlcv_response(base_ts, 10)

        source = CoinbaseSource(exchange=mock_exchange)
        df = source.fetch(
            "BTC/USD", WINDOW_SECONDS, base_ts, base_ts + 5 * WINDOW_SECONDS
        )

        assert len(df) == 5
        assert df["window_open_ts"].max() < base_ts + 5 * WINDOW_SECONDS

    def test_fetch_empty_response(self) -> None:
        from strikecast.data.candle_source import CoinbaseSource

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = []

        source = CoinbaseSource(exchange=mock_exchange)
        df = source.fetch("BTC/USD", WINDOW_SECONDS, 1_700_000_000, 1_700_001_000)

        assert len(df) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/strikecast/test_candle_source.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'strikecast.data.candle_source'`

- [ ] **Step 3: Implement CoinbaseSource**

`strikecast/data/candle_source.py`:
```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from strikecast.constants import CANDLE_COLUMNS, WINDOW_SECONDS


@runtime_checkable
class CandleSource(Protocol):
    def fetch(
        self, symbol: str, granularity: int, start: int, end: int
    ) -> pd.DataFrame: ...


_TIMEFRAME_MAP: dict[int, str] = {
    60: "1m",
    300: "5m",
    900: "15m",
    3600: "1h",
    86400: "1d",
}


class CoinbaseSource:
    def __init__(self, exchange: object | None = None) -> None:
        if exchange is None:
            import ccxt

            exchange = ccxt.coinbase()
        self._exchange = exchange

    def fetch(
        self, symbol: str, granularity: int, start: int, end: int
    ) -> pd.DataFrame:
        timeframe = _TIMEFRAME_MAP.get(granularity)
        if timeframe is None:
            raise ValueError(
                f"Unsupported granularity {granularity}. "
                f"Supported: {list(_TIMEFRAME_MAP.keys())}"
            )

        ohlcv = self._exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=start * 1000, limit=300
        )

        if not ohlcv:
            return pd.DataFrame(columns=CANDLE_COLUMNS)

        df = pd.DataFrame(
            ohlcv, columns=["timestamp_ms", "open", "high", "low", "close", "volume"]
        )
        df["window_open_ts"] = (df["timestamp_ms"] // 1000).astype(int)
        df["symbol"] = symbol
        df["granularity"] = granularity
        df["amount"] = 0.0
        df["source"] = "coinbase"

        df = df[(df["window_open_ts"] >= start) & (df["window_open_ts"] < end)]
        return df[CANDLE_COLUMNS].reset_index(drop=True)
```

- [ ] **Step 4: Run CoinbaseSource tests**

```bash
pytest tests/strikecast/test_candle_source.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Write the failing tests for paginator**

`tests/strikecast/test_paginator.py`:
```python
from unittest.mock import MagicMock, call

import pandas as pd
import pytest

from strikecast.constants import CANDLE_COLUMNS, WINDOW_SECONDS


def _make_source_fetch(base_ts: int, total: int, batch: int = 300) -> MagicMock:
    """Create a mock CandleSource that returns batches of candles."""
    call_count = 0

    def fetch(symbol: str, granularity: int, start: int, end: int) -> pd.DataFrame:
        nonlocal call_count
        batch_start = base_ts + call_count * batch * granularity
        n = min(batch, (total - call_count * batch))
        if n <= 0:
            return pd.DataFrame(columns=CANDLE_COLUMNS)
        call_count += 1
        return pd.DataFrame(
            {
                "symbol": symbol,
                "granularity": granularity,
                "window_open_ts": [batch_start + i * granularity for i in range(n)],
                "open": 35000.0,
                "high": 35010.0,
                "low": 34990.0,
                "close": 35005.0,
                "volume": 1.0,
                "amount": 0.0,
                "source": "coinbase",
            }
        )

    mock = MagicMock()
    mock.fetch = MagicMock(side_effect=fetch)
    return mock


class TestPaginator:
    def test_single_batch(self) -> None:
        from strikecast.data.paginator import fetch_all_candles

        base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
        source = _make_source_fetch(base_ts, total=50)

        result = fetch_all_candles(
            source=source,
            symbol="BTC/USD",
            granularity=WINDOW_SECONDS,
            start_ts=base_ts,
            end_ts=base_ts + 50 * WINDOW_SECONDS,
            rate_limit=1000.0,
        )
        assert len(result) == 50

    def test_multiple_batches(self) -> None:
        from strikecast.data.paginator import fetch_all_candles

        base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
        source = _make_source_fetch(base_ts, total=500, batch=300)

        result = fetch_all_candles(
            source=source,
            symbol="BTC/USD",
            granularity=WINDOW_SECONDS,
            start_ts=base_ts,
            end_ts=base_ts + 500 * WINDOW_SECONDS,
            rate_limit=1000.0,
        )
        assert len(result) == 500
        assert source.fetch.call_count == 2

    def test_empty_source(self) -> None:
        from strikecast.data.paginator import fetch_all_candles

        source = MagicMock()
        source.fetch.return_value = pd.DataFrame(columns=CANDLE_COLUMNS)

        result = fetch_all_candles(
            source=source,
            symbol="BTC/USD",
            granularity=WINDOW_SECONDS,
            start_ts=1_700_000_000,
            end_ts=1_700_100_000,
            rate_limit=1000.0,
        )
        assert len(result) == 0
```

- [ ] **Step 6: Implement paginator**

`strikecast/data/paginator.py`:
```python
from __future__ import annotations

import logging
import time

import pandas as pd

from strikecast.constants import CANDLE_COLUMNS
from strikecast.data.candle_source import CandleSource

logger = logging.getLogger(__name__)


def fetch_all_candles(
    source: CandleSource,
    symbol: str,
    granularity: int,
    start_ts: int,
    end_ts: int,
    rate_limit: float = 10.0,
    batch_size: int = 300,
) -> pd.DataFrame:
    all_batches: list[pd.DataFrame] = []
    current_start = start_ts
    request_count = 0

    while current_start < end_ts:
        batch = source.fetch(symbol, granularity, current_start, end_ts)

        if batch.empty:
            break

        all_batches.append(batch)
        request_count += 1

        last_ts = int(batch["window_open_ts"].max())
        next_start = last_ts + granularity

        if next_start <= current_start:
            break
        current_start = next_start

        logger.info(
            "Fetched batch %d: %d candles (up to ts=%d)",
            request_count,
            len(batch),
            last_ts,
        )

        if current_start < end_ts and rate_limit > 0:
            time.sleep(1.0 / rate_limit)

    if not all_batches:
        return pd.DataFrame(columns=CANDLE_COLUMNS)

    result = pd.concat(all_batches, ignore_index=True)
    result = result[
        (result["window_open_ts"] >= start_ts) & (result["window_open_ts"] < end_ts)
    ]
    return result.drop_duplicates(subset=["window_open_ts"]).reset_index(drop=True)
```

- [ ] **Step 7: Run all candle source + paginator tests**

```bash
pytest tests/strikecast/test_candle_source.py tests/strikecast/test_paginator.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add strikecast/data/candle_source.py strikecast/data/paginator.py tests/strikecast/test_candle_source.py tests/strikecast/test_paginator.py
git commit -m "feat(strikecast): CoinbaseSource with ccxt and paginated candle fetching"
```

---

### Task 7: Polymarket read-only client

**Files:**
- Create: `strikecast/data/polymarket_read.py`
- Create: `tests/strikecast/test_polymarket_read.py`

**Context:** The Polymarket Gamma API is at `gamma-api.polymarket.com`. For Phase 0 we query resolved BTC 5-minute markets for historical backfill. We use `httpx` for HTTP calls — no `py-clob-client` dependency, eliminating any risk of importing order/signing classes (NFR-001).

- [ ] **Step 1: Write the failing tests**

`tests/strikecast/test_polymarket_read.py`:
```python
from unittest.mock import MagicMock, patch

import httpx
import pytest

from strikecast.constants import WINDOW_SECONDS


SAMPLE_GAMMA_RESPONSE = [
    {
        "id": "event_1",
        "slug": "will-btc-5min-up-or-down",
        "markets": [
            {
                "id": "cond_abc",
                "question": "Will BTC go up?",
                "outcomes": ["Up", "Down"],
                "outcomePrices": "[0.52, 0.48]",
                "clobTokenIds": "[\"tok_up_1\", \"tok_down_1\"]",
                "closed": True,
                "startDate": "2026-01-15T00:00:00Z",
                "endDate": "2026-01-15T00:05:00Z",
            }
        ],
        "description": "Price to beat: $42000.00",
    }
]


class TestFetchMarketMetadata:
    def test_returns_market_dict(self) -> None:
        from strikecast.data.polymarket_read import fetch_market_metadata

        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_GAMMA_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("strikecast.data.polymarket_read.httpx.get", return_value=mock_response):
            result = fetch_market_metadata(
                start_ts=1705276800,
                end_ts=1705276800 + 300,
            )

        assert result is not None
        assert len(result) >= 1
        row = result.iloc[0]
        assert "condition_id" in result.columns
        assert "token_id_up" in result.columns
        assert "price_to_beat" in result.columns
        assert "price_up" in result.columns

    def test_returns_empty_when_no_markets(self) -> None:
        from strikecast.data.polymarket_read import fetch_market_metadata

        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()

        with patch("strikecast.data.polymarket_read.httpx.get", return_value=mock_response):
            result = fetch_market_metadata(
                start_ts=1705276800,
                end_ts=1705276800 + 300,
            )

        assert len(result) == 0


class TestModuleSafety:
    def test_does_not_import_py_clob_client(self) -> None:
        """polymarket_read.py must never import py-clob-client."""
        import ast
        from pathlib import Path

        source = Path("strikecast/data/polymarket_read.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "clob" not in alias.name.lower(), (
                        f"polymarket_read.py imports '{alias.name}' — "
                        "py-clob-client must not be imported"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module and "clob" in node.module.lower():
                    raise AssertionError(
                        f"polymarket_read.py imports from '{node.module}' — "
                        "py-clob-client must not be imported"
                    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/strikecast/test_polymarket_read.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'strikecast.data.polymarket_read'`

- [ ] **Step 3: Implement the Polymarket read-only client**

`strikecast/data/polymarket_read.py`:
```python
"""Read-only Polymarket client for BTC 5-minute Up/Down markets.

SAFETY: This module uses httpx for HTTP calls. It does NOT import
py-clob-client, and it MUST NEVER import any order, signing, or
wallet module. NFR-001 is enforced by test_no_order_path.py.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import httpx
import pandas as pd

from strikecast.constants import MARKET_COLUMNS, WINDOW_SECONDS

logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"


def fetch_market_metadata(
    start_ts: int,
    end_ts: int,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch resolved BTC 5-min Up/Down markets from Gamma API.

    Queries the Gamma events endpoint for closed BTC 5-minute markets
    within the given time range. Returns a DataFrame matching the
    pm_markets schema.

    Args:
        start_ts: Window start (Unix seconds, grid-aligned).
        end_ts: Window end (Unix seconds, grid-aligned).
        timeout: HTTP timeout in seconds.

    Returns:
        DataFrame with columns from MARKET_COLUMNS. Empty if no markets found.
    """
    resp = httpx.get(
        f"{GAMMA_API_BASE}/events",
        params={
            "tag": "btc-5-minute",
            "closed": True,
            "limit": 100,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    events = resp.json()

    rows: list[dict] = []
    for event in events:
        for market in event.get("markets", []):
            parsed = _parse_market(event, market)
            if parsed is None:
                continue
            if start_ts <= parsed["window_open_ts"] < end_ts:
                rows.append(parsed)

    if not rows:
        return pd.DataFrame(columns=MARKET_COLUMNS)
    return pd.DataFrame(rows)[MARKET_COLUMNS]


def _parse_market(event: dict, market: dict) -> dict | None:
    """Extract structured market data from a Gamma API event+market pair."""
    try:
        end_date = market.get("endDate", "")
        if not end_date:
            return None

        dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        window_close_ts = int(dt.timestamp())
        window_open_ts = window_close_ts - WINDOW_SECONDS

        if window_open_ts % WINDOW_SECONDS != 0:
            return None

        price_to_beat = _extract_price_to_beat(event.get("description", ""))
        if price_to_beat is None:
            return None

        outcome_prices = _parse_json_list(market.get("outcomePrices", "[]"))
        clob_token_ids = _parse_json_list(market.get("clobTokenIds", "[]"))

        if len(outcome_prices) < 2 or len(clob_token_ids) < 2:
            return None

        return {
            "window_open_ts": window_open_ts,
            "condition_id": market.get("id", ""),
            "token_id_up": clob_token_ids[0],
            "token_id_down": clob_token_ids[1],
            "price_to_beat": price_to_beat,
            "price_up": float(outcome_prices[0]),
            "price_down": float(outcome_prices[1]),
            "captured_ts": int(datetime.now(timezone.utc).timestamp()),
        }
    except (ValueError, IndexError, KeyError) as exc:
        logger.debug("Skipping unparseable market: %s", exc)
        return None


def _extract_price_to_beat(description: str) -> float | None:
    """Extract the strike price from event description text."""
    match = re.search(r"\$?([\d,]+(?:\.\d+)?)", description)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


def _parse_json_list(raw: str) -> list[str]:
    """Parse a JSON-encoded list string like '[\"a\", \"b\"]'."""
    import json

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def fetch_resolution_labels(
    candles_df: pd.DataFrame,
    markets_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive resolution labels from Coinbase candles and market metadata.

    For Phase 0, the oracle_close is approximated by the Coinbase close.
    The Chainlink oracle integration is added in Phase 4 when paper-PnL
    requires the actual resolution source.

    Args:
        candles_df: Candle data with window_open_ts and close columns.
        markets_df: Market metadata with window_open_ts and price_to_beat.

    Returns:
        DataFrame with columns from LABEL_COLUMNS.
    """
    if candles_df.empty or markets_df.empty:
        from strikecast.constants import LABEL_COLUMNS
        return pd.DataFrame(columns=LABEL_COLUMNS)

    merged = pd.merge(
        markets_df[["window_open_ts", "price_to_beat"]],
        candles_df[["window_open_ts", "close"]],
        on="window_open_ts",
        how="inner",
    )

    return pd.DataFrame(
        {
            "window_open_ts": merged["window_open_ts"],
            "oracle_close": merged["close"],
            "coinbase_close": merged["close"],
            "outcome_up": merged["close"] > merged["price_to_beat"],
        }
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/strikecast/test_polymarket_read.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add strikecast/data/polymarket_read.py tests/strikecast/test_polymarket_read.py
git commit -m "feat(strikecast): read-only Polymarket client via httpx (no py-clob-client)"
```

---

### Task 8: Non-negotiable safety tests

**Files:**
- Create: `tests/strikecast/test_no_order_path.py`
- Create: `tests/strikecast/test_grid_alignment.py`

These are integrity guarantees that run in pre-commit. They must exist before any result is trusted.

- [ ] **Step 1: Write the import guard test (NFR-001)**

`tests/strikecast/test_no_order_path.py`:
```python
"""NFR-001: Zero execution surface.

Scans all Python source files under strikecast/ and asserts that no
order-placement, signing, or wallet symbol from py-clob-client (or any
Binance client) is imported. A failure here is a release blocker.
"""
import ast
from pathlib import Path

import pytest

STRIKECAST_ROOT = Path("strikecast")

BANNED_MODULE_FRAGMENTS = [
    "py_clob_client.order",
    "py_clob_client.signing",
    "py_clob_client.signer",
    "py_clob_client.wallet",
    "py_order_utils",
    "binance",
]

BANNED_NAME_FRAGMENTS = [
    "create_order",
    "place_order",
    "submit_order",
    "sign_order",
    "cancel_order",
    "ApiSigner",
    "ClobClient",
    "BinanceClient",
    "private_key",
]


def _collect_python_files() -> list[Path]:
    return sorted(STRIKECAST_ROOT.rglob("*.py"))


def _scan_file(path: Path) -> list[str]:
    """Return a list of violation descriptions found in the file."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for banned in BANNED_MODULE_FRAGMENTS:
                    if banned in alias.name.lower():
                        violations.append(
                            f"{path}:{node.lineno} imports '{alias.name}'"
                        )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for banned in BANNED_MODULE_FRAGMENTS:
                if banned in module.lower():
                    violations.append(
                        f"{path}:{node.lineno} imports from '{module}'"
                    )
            for alias in node.names:
                for banned in BANNED_NAME_FRAGMENTS:
                    if banned.lower() in alias.name.lower():
                        violations.append(
                            f"{path}:{node.lineno} imports name '{alias.name}' from '{module}'"
                        )
        elif isinstance(node, ast.Name):
            for banned in BANNED_NAME_FRAGMENTS:
                if node.id.lower() == banned.lower():
                    violations.append(
                        f"{path}:{node.lineno} references name '{node.id}'"
                    )

    return violations


class TestNoOrderPath:
    def test_no_banned_imports_in_strikecast(self) -> None:
        all_violations: list[str] = []
        for py_file in _collect_python_files():
            all_violations.extend(_scan_file(py_file))

        if all_violations:
            msg = "NFR-001 VIOLATION: order/signing/wallet symbols found:\n"
            msg += "\n".join(f"  - {v}" for v in all_violations)
            pytest.fail(msg)

    def test_strikecast_has_python_files(self) -> None:
        """Sanity check: the scan actually found files to scan."""
        files = _collect_python_files()
        assert len(files) > 0, "No .py files found under strikecast/"
```

- [ ] **Step 2: Write the grid alignment test (FR-007)**

`tests/strikecast/test_grid_alignment.py`:
```python
"""FR-007: Every stored window_open_ts must be aligned to the 300s grid.

Loads all three Parquet stores and verifies that every timestamp
satisfies ts % 300 == 0.
"""
from pathlib import Path

import pandas as pd
import pytest

from strikecast.constants import WINDOW_SECONDS


class TestGridAlignment:
    def test_candle_timestamps_aligned(
        self, tmp_data_dir: Path, sample_candles: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_candles(sample_candles)
        candles = store.read_candles()

        misaligned = candles[candles["window_open_ts"] % WINDOW_SECONDS != 0]
        assert len(misaligned) == 0, (
            f"Found {len(misaligned)} candle timestamps not aligned to "
            f"{WINDOW_SECONDS}s grid: {misaligned['window_open_ts'].tolist()[:5]}"
        )

    def test_market_timestamps_aligned(
        self, tmp_data_dir: Path, sample_markets: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_markets(sample_markets)
        markets = store.read_markets()

        misaligned = markets[markets["window_open_ts"] % WINDOW_SECONDS != 0]
        assert len(misaligned) == 0, (
            f"Found {len(misaligned)} market timestamps not aligned to "
            f"{WINDOW_SECONDS}s grid"
        )

    def test_label_timestamps_aligned(
        self, tmp_data_dir: Path, sample_labels: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_labels(sample_labels)
        labels = store.read_labels()

        misaligned = labels[labels["window_open_ts"] % WINDOW_SECONDS != 0]
        assert len(misaligned) == 0, (
            f"Found {len(misaligned)} label timestamps not aligned to "
            f"{WINDOW_SECONDS}s grid"
        )
```

- [ ] **Step 3: Run both safety tests**

```bash
pytest tests/strikecast/test_no_order_path.py tests/strikecast/test_grid_alignment.py -v
```

Expected: all 5 tests PASS (2 order path + 3 grid alignment).

- [ ] **Step 4: Commit**

```bash
git add tests/strikecast/test_no_order_path.py tests/strikecast/test_grid_alignment.py
git commit -m "test(strikecast): non-negotiable safety tests (NFR-001 import guard, FR-007 grid alignment)"
```

---

### Task 9: CLI data phase

**Files:**
- Create: `strikecast/cli.py`
- Create: `tests/strikecast/test_cli.py`

- [ ] **Step 1: Write the failing tests**

`tests/strikecast/test_cli.py`:
```python
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


class TestCLIDataPhase:
    def test_run_data_phase_creates_candle_store(self, tmp_path: Path) -> None:
        from strikecast.cli import run_data_phase
        from strikecast.config import StrikecastConfig
        from strikecast.data.store import DataStore

        config = StrikecastConfig(
            data={"data_dir": str(tmp_path), "start": "2026-01-01", "end": "2026-01-02"},
            polymarket={"enabled": False},
        )
        for subdir in ["candles", "pm_markets", "resolution_labels", "reports"]:
            (tmp_path / subdir).mkdir(exist_ok=True)

        mock_source = MagicMock()
        mock_source.fetch.return_value = _make_minimal_candles(
            start_ts=1735689600, n=10
        )

        with patch(
            "strikecast.cli.CoinbaseSource", return_value=mock_source
        ):
            run_data_phase(config)

        store = DataStore(tmp_path)
        candles = store.read_candles()
        assert len(candles) > 0

    def test_cli_main_parses_phase_flag(self) -> None:
        from strikecast.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "--config", "config/default.yaml", "--phase", "data"])
        assert args.phase == "data"
        assert args.config == "config/default.yaml"

    def test_cli_main_default_phase_is_all(self) -> None:
        from strikecast.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "--config", "config/default.yaml"])
        assert args.phase == "all"


def _make_minimal_candles(start_ts: int, n: int = 10):
    import pandas as pd
    from strikecast.constants import WINDOW_SECONDS

    start_ts = start_ts - (start_ts % WINDOW_SECONDS)
    return pd.DataFrame(
        {
            "symbol": "BTC/USD",
            "granularity": WINDOW_SECONDS,
            "window_open_ts": [start_ts + i * WINDOW_SECONDS for i in range(n)],
            "open": 42000.0,
            "high": 42010.0,
            "low": 41990.0,
            "close": 42005.0,
            "volume": 1.0,
            "amount": 0.0,
            "source": "coinbase",
        }
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/strikecast/test_cli.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'strikecast.cli'`

- [ ] **Step 3: Implement the CLI**

`strikecast/cli.py`:
```python
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np

from strikecast.config import StrikecastConfig, load_config
from strikecast.data.candle_source import CoinbaseSource
from strikecast.data.paginator import fetch_all_candles
from strikecast.data.store import DataStore

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strikecast",
        description="Strikecast: calibrated probability engine for BTC binary outcomes",
    )
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run the Strikecast pipeline")
    run_parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config YAML file",
    )
    run_parser.add_argument(
        "--phase",
        type=str,
        choices=["all", "data", "baseline", "kronos", "finetune", "decision"],
        default="all",
        help="Which phase to run (default: all)",
    )

    return parser


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _ensure_data_dirs(data_dir: Path) -> None:
    for subdir in ["candles", "pm_markets", "resolution_labels", "reports"]:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)


def _ts_from_date(date_str: str) -> int:
    from datetime import datetime, timezone

    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def run_data_phase(config: StrikecastConfig) -> None:
    data_dir = Path(config.data.data_dir)
    _ensure_data_dirs(data_dir)
    store = DataStore(data_dir)

    start_ts = _ts_from_date(config.data.start)
    end_ts = _ts_from_date(config.data.end)

    logger.info("Fetching candles from %s to %s", config.data.start, config.data.end)
    source = CoinbaseSource()
    candles = fetch_all_candles(
        source=source,
        symbol=config.symbol,
        granularity=config.granularity,
        start_ts=start_ts,
        end_ts=end_ts,
        rate_limit=float(config.data.rate_limit_req_per_sec),
    )

    if not candles.empty:
        store.append_candles(candles)
        logger.info("Stored %d candles", len(candles))
    else:
        logger.warning("No candles fetched")

    gaps = store.detect_gaps(symbol=config.symbol, granularity=config.granularity)
    if not gaps.empty:
        total_missing = int(gaps["missing_count"].sum())
        total_expected = (end_ts - start_ts) // config.granularity
        pct = 100.0 * total_missing / total_expected if total_expected > 0 else 0
        logger.warning(
            "Data quality: %d gaps (%d missing windows, %.2f%%)",
            len(gaps),
            total_missing,
            pct,
        )
    else:
        logger.info("Data quality: no gaps detected")

    if config.polymarket.enabled:
        logger.info("Polymarket ingestion: historical mode (Phase 0)")
        from strikecast.data.polymarket_read import (
            fetch_market_metadata,
            fetch_resolution_labels,
        )

        try:
            markets = fetch_market_metadata(start_ts=start_ts, end_ts=end_ts)
            if not markets.empty:
                store.append_markets(markets)
                logger.info("Stored %d Polymarket markets", len(markets))

                candle_data = store.read_candles()
                labels = fetch_resolution_labels(candle_data, markets)
                if not labels.empty:
                    store.append_labels(labels)
                    logger.info("Stored %d resolution labels", len(labels))
            else:
                logger.info("No Polymarket markets found for this window")
        except Exception:
            logger.exception("Polymarket ingestion failed (non-fatal for Phase 0)")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)
    _set_seed(config.seed)
    logger.info("Loaded config from %s (seed=%d)", args.config, config.seed)

    if args.phase in ("all", "data"):
        run_data_phase(config)

    if args.phase == "data":
        logger.info("Phase 0 (data) complete.")
        return

    if args.phase in ("all", "baseline"):
        logger.info("Phase 1 (baseline): not yet implemented")

    if args.phase in ("all", "kronos"):
        logger.info("Phase 2 (kronos): not yet implemented")

    if args.phase in ("all", "finetune"):
        logger.info("Phase 3 (finetune): not yet implemented")

    if args.phase in ("all", "decision"):
        logger.info("Phase 4 (decision): not yet implemented")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/strikecast/test_cli.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Verify `python -m strikecast` works**

```bash
python -m strikecast --help
```

Expected: prints the argparse help message showing `run` subcommand.

- [ ] **Step 6: Commit**

```bash
git add strikecast/cli.py strikecast/__main__.py tests/strikecast/test_cli.py
git commit -m "feat(strikecast): CLI skeleton with --phase data for Phase 0 pipeline"
```

---

### Task 10: Pre-commit & tooling setup

**Files:**
- Create: `.pre-commit-config.yaml`
- Create: `pyproject.toml` (strikecast linting config section only)

- [ ] **Step 1: Create pre-commit config**

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks:
      - id: black
        args: ["--line-length=100"]
        files: ^strikecast/

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.8
    hooks:
      - id: ruff
        args: ["--fix"]
        files: ^strikecast/

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        args: ["--strict", "--ignore-missing-imports"]
        files: ^strikecast/
        additional_dependencies:
          - pydantic>=2.0.0
          - pandas-stubs
          - types-PyYAML

  - repo: local
    hooks:
      - id: no-order-path
        name: NFR-001 import guard
        entry: pytest tests/strikecast/test_no_order_path.py -x -q
        language: system
        pass_filenames: false
        files: ^strikecast/

      - id: grid-alignment
        name: FR-007 grid alignment
        entry: pytest tests/strikecast/test_grid_alignment.py -x -q
        language: system
        pass_filenames: false
        files: ^strikecast/data/store\.py
```

- [ ] **Step 2: Add ruff + black + mypy config to pyproject.toml**

Create `pyproject.toml` at the repo root (if it doesn't exist, or add to existing):
```toml
[tool.black]
line-length = 100
target-version = ["py310"]

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "B", "UP", "SIM", "PD"]

[tool.ruff.isort]
known-first-party = ["strikecast"]

[tool.mypy]
strict = true
ignore_missing_imports = true
files = ["strikecast/"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=strikecast --cov-report=term-missing"
```

- [ ] **Step 3: Install pre-commit and run**

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Fix any formatting issues raised by black or ruff, then re-run until clean.

- [ ] **Step 4: Run the full test suite**

```bash
pytest tests/strikecast/ -v
```

Expected: all tests PASS (config: 6, store: 16, candle_source: 3, paginator: 3, polymarket: 3, safety: 5, cli: 3 = ~39 tests).

- [ ] **Step 5: Commit**

```bash
git add .pre-commit-config.yaml pyproject.toml
git commit -m "chore(strikecast): pre-commit hooks (black, ruff, mypy, safety tests) and tool config"
```

---

## Phase 0: Definition of Done

From `specs/quality.md`:

- [x] Candle store populated via Coinbase paginated backfill
- [x] Market store populated via Polymarket Gamma API (historical)
- [x] Resolution label store populated (Coinbase close as Phase 0 proxy)
- [x] Gap report generated and logged (target: < 0.5% missing)
- [x] Grid alignment test green
- [x] Import guard test green (no order/signing surface)
- [x] All timestamps aligned to 300s grid

---

## Phase roadmap (subsequent plans)

Each phase gets its own implementation plan written after the previous phase is complete.

**Phase 1: Baselines + Scoring Harness** (~1 week)
- `estimators/base.py`: ProbResult dataclass, Estimator Protocol
- `estimators/random_walk.py`: analytic baseline
- `estimators/garch_mc.py`: GARCH(1,1) Monte-Carlo baseline via `arch`
- `eval/splits.py`: purged + embargoed walk-forward split
- `eval/scoring.py`: Brier, log loss, ECE, BSS, directional accuracy + kill flag
- `eval/report.py`: JSON + Markdown run report
- `test_no_leakage.py`, `test_reproducibility.py`, `test_calibration_split_disjoint.py`
- CLI `--phase baseline`

**Phase 2: Kronos Zero-Shot** (~3 days)
- `kronos_adapter.py`: thin wrapper around `model/` imports
- `estimators/kronos_binary.py`: MC probability wrapper around KronosPredictor
- Zero-shot evaluation on test set, compare to Phase 1 baselines

**Phase 3: Fine-Tune + Calibrate** (~2 weeks)
- Fine-tune Kronos-small tokenizer + predictor on BTC 5-min data
- `calibration/calibrator.py`: isotonic primary / Platt fallback
- `calibration/reliability.py`: CORP/PAV reliability diagrams + ECE
- Walk-forward evaluation with calibrated predictions

**Phase 4: Decision** (~3 days)
- Apply kill criterion (K2 > 0?)
- `eval/paper_pnl.py`: edge-gate paper-PnL simulation vs Polymarket prices
- Final run report with all KPIs, CIs, and reliability diagrams
- Document result (positive or negative) with full statistical rigor
