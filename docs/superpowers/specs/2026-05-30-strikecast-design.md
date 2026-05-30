# Strikecast — Implementation Design

> Validated design for the Strikecast calibrated probability engine. Built on top of the Kronos foundation model within the existing Kronos repository.
>
> **Source specs:** `specs/steering.md`, `specs/requirements.md`, `specs/design.md`, `specs/quality.md`
> **Date:** 2026-05-30

---

## Decisions from brainstorming

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Repo structure | Sibling in Kronos repo (Approach A) | Simplest for single-dev research; no submodule overhead |
| Kronos integration | Direct import via adapter (`kronos_adapter.py`) | No changes to `model/`, adapter tracks coupling surface (NFR-009) |
| Coinbase access | Public endpoint first (10 req/s) | Zero setup friction; upgrade to authenticated later if needed |
| Polymarket mode | Historical backfill only (Phase 0) | Sufficient for backtesting; live snapshotting deferred |
| GPU target | Apple Silicon (MPS) | Primary dev hardware; CPU fallback for unsupported ops |
| Data window | 6 months (~52k 5-min windows) | Enough for 60/20/20 split with statistical power; ~17s backfill |

---

## 1. Project structure

```
Kronos/                          # existing repo
├── model/                       # upstream Kronos code (untouched)
├── strikecast/                  # new top-level package
│   ├── __init__.py
│   ├── __main__.py              # dispatches to cli.py for `python -m strikecast`
│   ├── constants.py             # named constants (WINDOW_SECONDS, DEFAULT_SAMPLE_COUNT, etc.)
│   ├── kronos_adapter.py        # thin wrapper around model/ imports
│   ├── config.py                # Pydantic config, loads YAML
│   ├── data/
│   │   ├── __init__.py
│   │   ├── candle_source.py     # CandleSource Protocol + CoinbaseSource (ccxt)
│   │   ├── paginator.py         # 300/req pagination logic
│   │   ├── polymarket_read.py   # READ-ONLY Gamma/CLOB client
│   │   └── store.py             # Parquet read/write, gap detection, grid alignment
│   ├── estimators/
│   │   ├── __init__.py
│   │   ├── base.py              # ProbResult dataclass, Estimator Protocol
│   │   ├── random_walk.py       # FR-030
│   │   ├── garch_mc.py          # FR-031
│   │   └── kronos_binary.py     # FR-010..015, wraps adapter
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── calibrator.py        # isotonic primary / Platt fallback
│   │   └── reliability.py       # CORP/PAV diagrams + ECE computation
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── splits.py            # purged + embargoed walk-forward
│   │   ├── scoring.py           # Brier/logloss/ECE/BSS/dir-acc + kill flag
│   │   ├── paper_pnl.py         # edge-gate sim vs market price
│   │   └── report.py            # JSON + Markdown run report
│   └── cli.py                   # single entrypoint: python -m strikecast run
├── config/
│   └── default.yaml             # Pydantic-validated config file
├── tests/
│   ├── strikecast/              # Strikecast test directory
│   │   ├── test_no_leakage.py
│   │   ├── test_no_order_path.py
│   │   ├── test_reproducibility.py
│   │   ├── test_grid_alignment.py
│   │   └── test_calibration_split_disjoint.py
│   └── ...                      # existing Kronos tests (untouched)
├── data/                        # gitignored, runtime data
│   ├── candles/
│   ├── pm_markets/
│   ├── resolution_labels/
│   └── reports/
└── specs/                       # existing spec files
```

### Kronos adapter (`kronos_adapter.py`)

The adapter does three things:

1. Adds `model/` to `sys.path` and re-exports `KronosPredictor`, `KronosTokenizer`, `Kronos`.
2. Provides a factory `load_predictor(checkpoint, device) -> KronosPredictor` that handles HuggingFace download + MPS-aware device placement.
3. Documents the exact upstream API surface depended on, so NFR-009 diff tracking reduces to "what changed in this one file?"

No changes to `model/` itself. Patches for MPS compatibility (if needed) go through the adapter or as minimal, tracked edits.

---

## 2. Data layer (Phase 0)

### Coinbase candle ingestion

- **`CandleSource` Protocol** defines `fetch(symbol, granularity, start, end) -> pd.DataFrame`.
- **`CoinbaseSource`** via `ccxt`: calls `fetch_ohlcv("BTC/USD", timeframe="5m", since=..., limit=300)`. Public endpoint, 10 req/s.
- **Paginator:** walks backward from `end` to `start` in 300-candle chunks, reverses. For 6 months (~52k candles), ~174 requests, ~17 seconds. Simple sleep-based throttle at 10 req/s. Progress logged.
- **Deduplication:** on write, dedup on `(symbol, granularity, window_open_ts)`. Re-runs skip already-fetched ranges.
- **Gap detection:** scan for missing timestamps in the 300s grid. Gaps logged to a data-quality report. Gaps > 1 window flagged but never silently forward-filled (FR-003).
- **Grid alignment:** `window_open_ts % 300 == 0` enforced at write time. Off-grid data rejected.

### Polymarket read-only ingestion

- **`polymarket_read.py`** — a single module, two functions:
  - `fetch_market_metadata(window_open_ts) -> dict` — Gamma API for BTC Up/Down market metadata (token IDs, price-to-beat, Up/Down prices).
  - `fetch_resolution_label(window_open_ts) -> dict` — Chainlink oracle price at window boundary from RTDS feed, plus Coinbase close for basis tracking.
- **Historical backfill only.** Queries resolved markets, not live ones.
- **Read-only enforcement:** never imports `py-clob-client`'s order/signing classes. Verified by `test_no_order_path.py`.

### Storage (`store.py`)

- **Three Parquet tables:** `candles`, `pm_markets`, `resolution_labels` — schemas match `design.md` §3 exactly.
- **Location:** `data/{candles,pm_markets,resolution_labels}/` — all gitignored.
- **Read:** DuckDB queries Parquet directly via `store.query(sql) -> pd.DataFrame`.
- **Write:** `append_candles(df)`, `append_markets(df)`, `append_labels(df)` — each validates schema + grid alignment, deduplicates, then writes Parquet.

---

## 3. Estimators (Phases 1–3)

### Shared contract

```python
@dataclass
class ProbResult:
    p: float            # calibrated (or raw if no calibrator)
    p_raw: float        # always uncalibrated
    ci_low: float       # bootstrap CI lower bound
    ci_high: float      # bootstrap CI upper bound
    n_samples: int      # MC samples (0 for analytic baselines)

class Estimator(Protocol):
    def estimate(self, lookback_df, x_ts, target_ts, strike, mode) -> ProbResult: ...
```

All estimators return the same shape. Scoring engine doesn't branch on estimator type.

### Random-walk baseline (Phase 1, FR-030)

- Analytic. `up_or_down` mode returns `p = 0.5`. Off-the-money: normal CDF with zero drift and realized vol from lookback.
- No CI (analytic): `ci_low = ci_high = p`, `n_samples = 0`.

### GARCH-MC baseline (Phase 1, FR-031)

- Fits `arch_model(returns, vol='Garch', p=1, q=1, dist='Normal')` on lookback log-returns (default 2016 bars = 1 week).
- Simulates `sample_count` (1000) one-step-ahead returns, computes fraction above strike.
- Bootstrap CI: resample simulated terminals 1000 times, take 2.5th/97.5th percentiles.
- CPU-only (`arch` uses scipy). Single-window GARCH fit < 100ms.

### Kronos MC estimator (Phases 2–3, FR-010..015)

- Wraps `KronosPredictor` via adapter. `predict()` with `pred_len=1`, `sample_count=1000`, `T=1.0`, `top_p=0.9`.
- `p_raw = mean(close_i > strike)`. Bootstrap CI same method as GARCH-MC.
- Context clamping: warns and truncates if lookback exceeds `max_context` (512 for small).
- Optional `Calibrator`: if present, `p = calibrator.apply(p_raw)`.
- Batch mode via `estimate_batch()` using `KronosPredictor.predict_batch()` for backtesting throughput.
- MPS auto-detection inherited from upstream. Adapter falls back to CPU with warning if unsupported ops hit.

### Phase ordering

1. Phase 1: random-walk + GARCH-MC + full scoring harness. Baseline KPIs. Scoreboard before contestant.
2. Phase 2: Kronos zero-shot (no calibration). Compare to baselines.
3. Phase 3: fine-tune Kronos-small + calibration layer. Final evaluation.

---

## 4. Calibration

### Calibrator (`calibration/calibrator.py`)

- Single `Calibrator` class with strategy parameter (`"isotonic"` or `"platt"`).
- **Isotonic:** `sklearn.isotonic.IsotonicRegression`, `out_of_bounds="clip"`. Flexible, handles any monotonic distortion. Validation split has ~10k windows — no overfitting concern.
- **Platt:** `sklearn.linear_model.LogisticRegression` on `p_raw`. 2-parameter sigmoid, stable on small data. Fallback if isotonic overfits.
- **Persistence (FR-023):** `save()`/`load()` via `joblib` with metadata (checkpoint hash, data window, method, fit date).

### Split discipline (FR-020)

- **Three-way time-ordered split:** train 60% / validation 20% / test 20%.
- Purge ≥ 1 window + embargo ≥ 1 window at each boundary.
- Calibrator fitted only on validation-split predictions. Never sees test data.
- `test_calibration_split_disjoint.py` asserts pairwise disjoint timestamp sets.

### Reliability diagrams (`calibration/reliability.py`)

- CORP/PAV approach (not equal-width binning). Pool Adjacent Violators for consistent reliability curves.
- Two diagrams per run: uncalibrated + calibrated.
- ECE: weighted mean absolute deviation between predicted and observed frequency across PAV bins.

---

## 5. Evaluation & scoring

### Splits (`eval/splits.py`)

- Single walk-forward split (not k-fold for v1). `generate_splits()` returns `(train_ts, val_ts, test_ts)`.
- Deterministic (time-ordered), so reproducibility is automatic.

### Scoring engine (`eval/scoring.py`)

Per estimator, per moneyness bucket:

| Metric | KPI |
|--------|-----|
| Brier score | K1 |
| Brier skill score vs GARCH-MC | K2 |
| ECE (PAV-binned) | K3 |
| Log loss (clipped) | K4 |
| Directional accuracy | K5 |

- **Moneyness buckets:** near-the-money (≤ 0.1%), far-the-money (> 1%), all.
- **Bootstrap 95% CI** on every metric (10,000 resamples). Report builder raises if CI is missing (NFR-007).
- **Kill criterion (FR-044):** `brier_skill_score_vs_garch <= 0` → `FAILED-KILL-CRITERION`. Automatic.

### Paper-PnL (`eval/paper_pnl.py`)

- Compare `p_calibrated` to Polymarket `price_up`. Bet when `|p - price_up| > edge_threshold` (default 0.02).
- Fixed-fraction sizing. No Kelly, no portfolio optimization.
- Resolution uses Chainlink oracle price (matches Polymarket settlement).
- Reports: cumulative PnL, hit rate, number of bets, per-bet Sharpe. All with CIs.
- Report always states label source (Chainlink for paper-PnL, Coinbase for model scoring) + Coinbase-Chainlink basis distribution.

### Run report (`eval/report.py`)

- JSON + Markdown per run.
- Contents: data window, git commit, checkpoint hash, seed, config snapshot, full KPI table with CIs, reliability diagrams (base64 PNG), basis distribution, paper-PnL summary, kill criterion (PASS/FAIL).
- Location: `data/reports/{run_id}/` — gitignored.

---

## 6. Config, CLI & reproducibility

### Config (`config/default.yaml` + `config.py`)

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
  device: "auto"                # MPS > CPU
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

Pydantic `BaseSettings` validates all values at startup. Every tunable knob in config or named constants — no magic numbers.

### CLI (`cli.py`)

Single entrypoint: `python -m strikecast run --config config/default.yaml`

Pipeline order:
1. Load + validate config
2. Set global seed (random, numpy, torch)
3. Pull/update candle data
4. Pull/update Polymarket historical data
5. Data quality report (gap check)
6. Generate splits
7. Run all estimators on test windows
8. Fit calibrator on validation split
9. Re-score with calibrated probabilities
10. Apply kill criterion
11. Paper-PnL simulation
12. Write JSON + Markdown report

`--phase` flag for partial runs (e.g., `--phase data`, `--phase baseline`).

### Reproducibility (NFR-003)

- Seed pinning: `torch.manual_seed`, `np.random.seed`, `random.seed`, plus `torch.use_deterministic_algorithms(True)` where MPS supports it.
- MPS determinism caveat: not all ops support deterministic mode. `test_reproducibility.py` runs on CPU to verify bit-identical Brier (NFR-003). MPS reproducibility is best-effort, documented.
- Config snapshot saved per run. Dependency versions locked in `requirements.txt`.

---

## 7. Testing strategy

### Non-negotiable tests

| Test | Verifies | Method |
|------|----------|--------|
| `test_no_order_path.py` | NFR-001: zero execution surface | `ast.parse` + `importlib` scan of `strikecast/` for order/signing/wallet symbols |
| `test_no_leakage.py` | NFR-002: no look-ahead | Permute future bars, assert `p` is bit-identical |
| `test_reproducibility.py` | NFR-003: deterministic | Two identical runs on CPU, Brier matches to 1e-9 |
| `test_grid_alignment.py` | FR-007: 300s grid | Assert `ts % 300 == 0` for all stored timestamps |
| `test_calibration_split_disjoint.py` | FR-020: split integrity | Assert pairwise disjoint timestamp sets |

### Unit tests

- Estimators: synthetic DataFrames, assert `ProbResult` validity (p in [0,1], CI bounds, n_samples).
- Scoring: hand-computed golden values to 1e-9.
- Store: write/read round-trip, dedup, grid rejection, gap detection.
- Config: Pydantic rejects invalid values.
- Calibrator: identity on well-calibrated data, correction on overconfident data.

### Property tests (Hypothesis)

- Estimator: any valid input produces `0 <= p <= 1`.
- Calibration monotonicity: isotonic preserves order.
- Split completeness: train + val + test cover all input timestamps.

### Integration tests

- Data clients: `vcrpy` recorded fixtures for Coinbase and Polymarket responses.
- Mini pipeline: `--phase baseline` on 200-window fixture, assert report JSON exists with expected keys.

### Coverage & pre-commit

- `pytest-cov --cov-fail-under=85` on `strikecast/`. `model/` excluded.
- Pre-commit: Black, Ruff, mypy `--strict`, nbstripout, `test_no_order_path.py`, `test_grid_alignment.py`.
