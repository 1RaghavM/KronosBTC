# requirements.md — Strikecast

> The **what**. Functional requirements in EARS form, non-functional requirements with thresholds, user stories, and an explicit non-goals list. See [`STEERING.md`](./STEERING.md) for the why, [`design.md`](./design.md) for the how, [`QUALITY.md`](./QUALITY.md) for the bar.
>
> **Priority key:** P0 = required for the kill-criterion decision; P1 = needed for a complete, trustworthy result; P2 = nice-to-have / hardening.
> **EARS pattern:** "The system shall [do X] by [mechanism] [when/while condition]."

---

## 1. User stories

- **US-1** — As a *quant researcher*, I want a single function that returns P(close > strike) for the next 5-min BTC window so I can treat Kronos as a probability source instead of a path generator.
- **US-2** — As a *quant researcher*, I want every model scored against a GARCH baseline and a coin flip on the identical test set so I can tell whether the foundation model earns its complexity.
- **US-3** — As a *skeptical analyst*, I want reliability diagrams and a Brier/ECE report so I can trust the numbers are calibrated, not just accurate.
- **US-4** — As a *US-based developer*, I want all data and trading paths to be jurisdiction-safe (no Binance, no live Polymarket orders) so the project is legally clean.
- **US-5** — As a *paper trader*, I want to compare my model's probability against Polymarket's live implied probability and simulate PnL so I can estimate edge without placing a real order.
- **US-6** — As a *model maintainer*, I want a reproducible fine-tune + evaluate pipeline so I can re-run the whole experiment from raw data with one command and get identical results.
- **US-7** — As a *careful experimenter*, I want guarantees against look-ahead leakage so a good backtest number is real, not an artifact.

---

## 2. Functional requirements (EARS)

### 2.1 Data ingestion & storage

| ID | Priority | Requirement |
|----|----------|-------------|
| FR-001 | P0 | The system shall retrieve historical BTC-USD 5-minute OHLCV candles by calling the Coinbase Advanced Trade candles endpoint with `granularity=300`, paginating in ≤ 300-candle requests. |
| FR-002 | P0 | The system shall persist all retrieved candles to a local columnar store (Parquet) keyed by `(symbol, granularity, window_open_ts)`, deduplicating on that key, so re-runs do not re-fetch. |
| FR-003 | P0 | The system shall detect and record gaps in the candle series (missing windows) and shall never silently forward-fill price across a gap larger than one window without flagging it in a data-quality report. |
| FR-004 | P1 | The system shall abstract the exchange behind a `CandleSource` interface (default: Coinbase via `ccxt`) so an alternate US-legal venue (e.g. Kraken) can be swapped without changing downstream code. |
| FR-005 | P0 | The system shall ingest Polymarket BTC "Up or Down" market metadata (token IDs, window-open timestamp, "price to beat") and current outcome prices by calling the Gamma and CLOB read endpoints, **read-only**, never authenticating an order path. |
| FR-006 | P0 | The system shall record the Polymarket resolution oracle price (Chainlink BTC/USD captured at the window boundary) for each evaluated window, stored separately from Coinbase candles, to serve as the ground-truth label for paper-PnL evaluation. |
| FR-007 | P1 | The system shall align Coinbase candle windows to the same fixed 300-second epoch grid Polymarket uses (Unix timestamps divisible by 300) so a candle and a market refer to the same window. |

### 2.2 Probability estimation (Kronos wrapper)

| ID | Priority | Requirement |
|----|----------|-------------|
| FR-010 | P0 | The system shall produce a probability P(close > strike) for a target future window by drawing `sample_count` ≥ 500 independent forecast paths from `KronosPredictor.predict` and computing the fraction of sampled closes that exceed the strike. |
| FR-011 | P0 | The system shall accept an arbitrary strike per request and shall also support strike = window-open price (the Polymarket "Up or Down" convention) as a named mode. |
| FR-012 | P0 | The system shall expose sampling controls (`T` temperature, `top_p`, `sample_count`) as configurable parameters with documented defaults (`T=1.0`, `top_p=0.9`, `sample_count=1000`). |
| FR-013 | P1 | The system shall return, alongside the point probability, a bootstrap confidence interval on that probability derived from the sampled paths, so downstream gating can account for Monte-Carlo sampling error. |
| FR-014 | P1 | The system shall support batched probability estimation across many windows via `KronosPredictor.predict_batch` for efficient backtesting. |
| FR-015 | P0 | The system shall clamp the lookback context to the model's `max_context` (512 for small/base, 2048 for mini) and shall warn rather than silently truncate when input exceeds it. |

### 2.3 Calibration

| ID | Priority | Requirement |
|----|----------|-------------|
| FR-020 | P0 | The system shall fit a post-hoc calibration map (isotonic regression as primary, Platt/sigmoid as fallback) on a validation split **disjoint** from both the training and test windows. |
| FR-021 | P0 | The system shall apply the fitted calibration map to all raw model probabilities before they are scored or used for paper PnL. |
| FR-022 | P1 | The system shall produce a CORP/isotonic reliability diagram and an ECE figure for both the uncalibrated and calibrated probabilities so the effect of calibration is auditable. |
| FR-023 | P2 | The system shall persist the fitted calibration map as a versioned artifact tied to the model checkpoint and data window that produced it. |

### 2.4 Baselines

| ID | Priority | Requirement |
|----|----------|-------------|
| FR-030 | P0 | The system shall implement a random-walk baseline that emits P = 0.5 for the at-the-money strike and the analytic no-drift normal probability for off-the-money strikes, as a floor benchmark. |
| FR-031 | P0 | The system shall implement a GARCH Monte-Carlo digital pricer: fit a GARCH(1,1)/Realized-GARCH model with the `arch` package on the return series, simulate terminal price paths, and compute P(close > strike) as the fraction above strike. |
| FR-032 | P0 | The system shall score the random-walk, GARCH-MC, and Kronos estimators on the identical test windows with the identical labels and report them in one table. |

### 2.5 Evaluation & backtesting

| ID | Priority | Requirement |
|----|----------|-------------|
| FR-040 | P0 | The system shall split data using purged + embargoed walk-forward folds, where each test fold is preceded by a purge gap of ≥ 1 window and an embargo of ≥ 1 window, to eliminate label leakage. |
| FR-041 | P0 | The system shall compute, per estimator and per moneyness bucket, Brier score, log loss, ECE, directional accuracy, and Brier skill score relative to the GARCH-MC baseline (KPIs K1–K5, K7 in `STEERING.md`). |
| FR-042 | P0 | The system shall label each evaluated window's outcome by comparing the resolution-source close to the strike, using the **Chainlink** oracle price for Polymarket-aligned evaluation and the Coinbase close for model-internal evaluation, and shall report which source was used. |
| FR-043 | P1 | The system shall simulate paper PnL (KPI K6) by entering a paper position only when `|P_model − P_market|` exceeds a configurable edge threshold that covers an assumed 2% round-trip cost, sizing by a fixed fraction, and shall report cumulative PnL, hit rate, and per-bet Sharpe. |
| FR-044 | P0 | The system shall apply the kill criterion automatically: if Kronos Brier skill score vs GARCH-MC ≤ 0 on the test set, the evaluation report shall flag the run as FAILED-KILL-CRITERION. |
| FR-045 | P1 | The system shall emit a machine-readable (JSON) and human-readable (Markdown) evaluation report per run, stamped with data window, model checkpoint hash, and git commit. |

### 2.6 Fine-tuning

| ID | Priority | Requirement |
|----|----------|-------------|
| FR-050 | P1 | The system shall fine-tune the Kronos tokenizer and predictor on BTC 5-min data using the upstream `finetune/` pipeline, starting from Kronos-small, with all paths and hyperparameters in a single config file. |
| FR-051 | P1 | The system shall keep the fine-tune data split aligned with the evaluation split so no test-window candle is ever seen during training. |
| FR-052 | P2 | The system shall support resuming fine-tuning from a checkpoint and shall log training/validation loss to a local file (Comet optional, default off). |

### 2.7 Reproducibility & ops

| ID | Priority | Requirement |
|----|----------|-------------|
| FR-060 | P0 | The system shall reproduce a full experiment (data → baseline → Kronos → calibration → report) from a single entrypoint command given a config file and a fixed random seed. |
| FR-061 | P1 | The system shall pin all dependencies and record the resolved environment (lockfile) so results are reproducible on another machine. |
| FR-062 | P2 | The system shall cache intermediate artifacts (candles, sampled paths, fitted models) and skip recomputation when inputs are unchanged. |

---

## 3. Non-functional requirements

| ID | Priority | Category | Requirement & threshold |
|----|----------|----------|--------------------------|
| NFR-001 | P0 | Legal | Zero code paths capable of placing a Polymarket/Binance order. Verified by absence of any signing/order client in the dependency graph (CI check). |
| NFR-002 | P0 | Correctness | No look-ahead leakage. Verified by a leakage unit test: shuffling future bars must not change any past-window probability. |
| NFR-003 | P0 | Reproducibility | Two runs with the same seed + config produce identical Brier scores to 1e-9. |
| NFR-004 | P1 | Performance | Backtest over ≥ 50,000 windows with `sample_count=1000` completes in ≤ 12h on one GPU (batched). |
| NFR-005 | P1 | Performance | Single-window probability (inference, `sample_count=1000`) returns in ≤ 5s on the target GPU. (Research-grade; not a latency product.) |
| NFR-006 | P1 | Data quality | < 0.5% missing windows over the training period after gap handling; any larger gap excluded and logged, not interpolated. |
| NFR-007 | P0 | Statistical validity | Every reported edge (K2, K6) accompanied by a confidence interval or significance test; point estimates alone are not a result. |
| NFR-008 | P1 | Resource | Peak GPU memory ≤ target card (e.g. 16 GB) for Kronos-small fine-tune at the configured batch size; fall back to gradient accumulation otherwise. |
| NFR-009 | P2 | Maintainability | Kronos fork changes isolated behind an adapter module; upstream diff ≤ 400 lines so rebasing onto new Kronos releases stays tractable. |
| NFR-010 | P1 | Observability | Every run writes a structured log with data window, seed, checkpoint hash, and all KPI values. |

---

## 4. What this project is explicitly NOT building

- **Not** a live trading bot. No execution, no wallet, no private keys, no order signing. (NFR-001.)
- **Not** a multi-asset or multi-horizon validated system. BTC/USD, 5-minute only for v1.
- **Not** a Binance-dependent pipeline. Any borrowed code referencing Binance is rewritten against Coinbase.
- **Not** a low-latency production service. Inference is research-grade.
- **Not** a from-scratch model. Fine-tuning the open-source Kronos checkpoints only; Kronos-large is unavailable.
- **Not** a portfolio/risk-management system. Paper-PnL uses a single simple edge-gate + fixed-fraction rule purely to express KPI K6.
- **Not** a hosted product for other users. Single-developer research.
- **Not** a redistribution of exchange data. Raw Coinbase/Polymarket data stays local.

---

### Traceability to KPIs
- K1–K5, K7 ← FR-041, FR-042, FR-040, FR-020/021.
- K2 (beats GARCH) ← FR-031, FR-032, FR-044.
- K6 (paper PnL vs market) ← FR-005, FR-006, FR-043.
- Reproducibility ← FR-060, FR-061, NFR-003.

### Sources
- Coinbase candle pagination/limits: <https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles>
- Polymarket read-only pricing & Chainlink resolution: <https://docs.polymarket.com/api-reference/pricing/get-market-price>, <https://gist.github.com/Archetapp/7680adabc48f812a561ca79d73cbac69>
- `arch` GARCH for crypto: <https://www.mdpi.com/2227-9091/11/12/211>
- Calibration (isotonic/Platt, Brier, ECE): <https://www.blog.trainindata.com/probability-calibration-in-machine-learning/>
- Purged + embargoed CV: <https://en.wikipedia.org/wiki/Purged_cross-validation>
- Kronos sampling (`sample_count`, `T`, `top_p`): <https://github.com/shiyu-coder/Kronos>
