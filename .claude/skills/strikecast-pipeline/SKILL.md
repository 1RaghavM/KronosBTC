---
name: strikecast-pipeline
description: >
  The end-to-end Strikecast estimation and evaluation workflow for turning Kronos forecast paths
  into calibrated binary probabilities with rigorous scoring. Use this skill whenever working on:
  Monte Carlo probability estimation from KronosPredictor sampled paths, calibration (isotonic/Platt),
  purged+embargoed walk-forward evaluation, Brier/logloss/ECE/BSS scoring, kill criterion checks,
  reliability diagrams, confidence intervals, paper-PnL simulation, or the evaluation report.
  Also use when implementing any estimator (Kronos, GARCH-MC, random-walk), the scoring engine,
  or the calibration layer. If you're touching strikecast/estimators/, strikecast/calibration/,
  or strikecast/eval/, this skill applies.
---

# Strikecast Pipeline

This skill captures the core Strikecast workflow: converting Kronos sampled OHLCV paths into
calibrated binary probabilities, then evaluating them with statistical rigor. It exists because
the same pipeline logic is specified across steering.md, requirements.md, design.md, and quality.md
— this is the single source of truth.

## The estimation flow

The central question Strikecast answers: **P(BTC close > strike at end of next 5-min window)**.

### Step 1: Monte Carlo sampling

Use `KronosPredictor.predict()` (or `predict_batch()` for backtesting) to draw independent
forecast paths. The fraction of sampled closes exceeding the strike IS the raw probability.

Key parameters and their defaults:
- `sample_count >= 500` (default 1000) — determines MC variance. At 1000, the CI on p is ~+-1.5pp
- `T = 1.0` — sampling temperature
- `top_p = 0.9` — nucleus sampling
- `max_context` — 512 for small/base, 2048 for mini. Warn (don't silently truncate) if input exceeds it

The `ProbResult` return contract:
```python
@dataclass
class ProbResult:
    p: float            # calibrated probability
    p_raw: float        # before calibration
    ci_low: float       # bootstrap CI lower bound
    ci_high: float      # bootstrap CI upper bound
    n_samples: int      # sample_count used
```

### Step 2: Bootstrap confidence interval

Compute a bootstrap CI over the sampled closes, not just the point estimate. Every probability
the system emits must have a CI — point estimates alone are not a result (NFR-007).

Target: CI half-width <= +-0.02 absolute at sample_count=1000.

### Step 3: Calibration

Apply a post-hoc calibration map to convert raw model probabilities into well-calibrated ones.

- **Primary**: isotonic regression (flexible, handles any monotonic distortion)
- **Fallback**: Platt/sigmoid scaling (more stable with small data)
- **Critical**: fit ONLY on a validation split that is disjoint from both training and test windows
- The fitted calibrator is a versioned artifact tied to the model checkpoint + data window

### Step 4: Baselines (score alongside, always)

No Kronos result is ever reported alone. Every evaluation includes side-by-side:
1. **Random-walk**: P=0.5 for at-the-money; analytic normal CDF for off-the-money
2. **GARCH-MC**: fit GARCH(1,1) or Realized-GARCH via `arch` package, simulate paths, compute fraction above strike
3. **Kronos** (raw and calibrated)

All three use the identical test windows and identical labels.

## The evaluation flow

### Walk-forward splitting

Use purged + embargoed walk-forward folds. This is non-negotiable because labels depend on future bars.

- **Purge gap**: >= 1 window between train and test
- **Embargo**: >= 1 window after test before next train fold
- Assert programmatically that train/validation/test window-id sets are pairwise disjoint

### Scoring metrics (per estimator, per moneyness bucket)

| KPI | Metric | Baseline | P0 Target |
|-----|--------|----------|-----------|
| K1 | Brier score (near-the-money) | 0.2500 (coin flip) | <= 0.2490 |
| K2 | Brier skill score vs GARCH-MC | 0.0 | > 0.0 |
| K3 | ECE (15 bins) | — | <= 0.020 |
| K4 | Log loss (near-the-money) | 0.6931 (coin flip) | <= 0.6910 |
| K5 | Directional accuracy (near-the-money) | 50.0% | >= 51.0% |
| K6 | Paper PnL vs Polymarket (after 2% RT cost) | break-even | > 0 over >= 2000 bets |
| K7 | Far-the-money sanity (strike +-1%+) | — | ECE <= 0.01 |

"Near-the-money" = strike within +-0.10% of spot at window open.

### Kill criterion

**Automatic and non-negotiable**: if after full fine-tuning and calibration, Kronos Brier skill
score vs GARCH-MC <= 0 on the test set, the run is flagged `FAILED-KILL-CRITERION`.

This is an acceptable, documented outcome — a credible negative result is a successful project deliverable.

### Reliability diagrams

Produce CORP/PAV isotonic reliability diagrams (not classical equal-width binning, which is unstable)
for both uncalibrated and calibrated probabilities. The effect of calibration must be visually auditable.

### Paper-PnL simulation

Enter a paper position only when `|P_model - P_market|` exceeds a configurable edge threshold
that covers the assumed 2% round-trip cost. Size by fixed fraction. Report:
- Cumulative PnL
- Hit rate
- Per-bet Sharpe

Uses Chainlink oracle prices for resolution (matching how Polymarket actually resolves),
NOT Coinbase closes. The Coinbase-Chainlink basis is measured and reported.

### Run report

Every run produces both:
- **JSON** (machine-readable) — for automated kill-criterion checks
- **Markdown** (human-readable) — for review

Both stamped with: data window, model checkpoint hash, git commit, seed, full KPI table with CIs,
reliability diagrams, and the KILL flag.

## Conventions

- Probabilities are `float` in `[0, 1]` — never percentages in code (convert only at display)
- Functions returning probabilities must name them `p_*` and document calibrated vs raw
- Timestamps are `int` Unix epoch seconds, UTC, aligned to the 300s grid
- `WINDOW_SECONDS = 300`, `DEFAULT_SAMPLE_COUNT = 1000`, `ROUND_TRIP_COST = 0.02` — named constants, no magic numbers
- If `sample_count` is raised to cut MC variance, re-verify NFR-004/005 (performance) in the same change

## Leakage prevention checklist

Run these checks on every change that touches data, splits, or estimators:
- [ ] Purge gap >= 1 window present in every fold (asserted programmatically)
- [ ] Embargo >= 1 window present in every fold (asserted programmatically)
- [ ] Train/validation/test window-id sets are pairwise disjoint
- [ ] `test_no_leakage.py` passes: permuting bars strictly after a window leaves that window's probability bit-identical
- [ ] `test_reproducibility.py` passes: same seed + config => identical Brier to 1e-9
