# QUALITY.md — Strikecast

> The **bar** everything must meet. Concrete tools, configs, thresholds, and checklists. See [`STEERING.md`](./STEERING.md), [`requirements.md`](./requirements.md), [`design.md`](./design.md).

---

## 1. Code style & linting

| Concern | Tool | Config / rule |
|---------|------|---------------|
| Format | **Black** | line length 100, default else. `black --check` in CI. |
| Lint + import sort | **Ruff** | rule sets `E,F,I,B,UP,SIM,PD`; `--fix` locally, `--no-fix` gate in CI. |
| Types | **mypy** `--strict` on `strikecast/` (the fork is exempt) | every public function in §5 of `design.md` fully annotated; no `Any` in estimator/calibration signatures. |
| Docstrings | **Google style** | every public function/class; must state units (USD, seconds, probability 0–1) and whether a probability is calibrated. |
| Notebooks | **nbstripout** pre-commit | no committed cell outputs; exploratory work lives in `notebooks/` and is never imported by `strikecast/`. |
| Pre-commit | **pre-commit** | runs black, ruff, mypy, nbstripout, and the two guard tests (§4) on every commit. |

**Hard style rules**
- No magic numbers for financial constants. `WINDOW_SECONDS = 300`, `DEFAULT_SAMPLE_COUNT = 1000`, `ROUND_TRIP_COST = 0.02` live in `config` or named constants.
- Probabilities are always `float` in `[0, 1]`; never percentages in code. Convert only at display.
- Timestamps are always `int` Unix epoch seconds, UTC, aligned to the 300s grid. No naive local datetimes cross a module boundary.
- A function that returns a probability must name it `p_*` and document calibrated vs raw.

---

## 2. Testing requirements

| Layer | Framework | Coverage / threshold |
|-------|-----------|----------------------|
| Unit | **pytest** | line coverage ≥ **85%** on `strikecast/` (excluding `kronos_fork/`), enforced by `pytest-cov --cov-fail-under=85`. |
| Property | **hypothesis** | estimators: any `p` returned is in `[0,1]`; calibrated `p` monotonic in raw `p` (isotonic invariant). |
| Statistical | custom | scoring functions checked against hand-computed Brier/log-loss on toy fixtures to 1e-9. |
| Integration | pytest + recorded fixtures (`vcrpy`) | data clients tested against recorded Coinbase/Polymarket responses; no live network in CI. |

**Non-negotiable tests (must exist and pass before any result is trusted):**
- `test_no_leakage.py` (NFR-002): permuting bars strictly after a window must leave that window's probability bit-identical. A failure here invalidates every backtest number — treat as a release blocker, not a flaky test.
- `test_no_order_path.py` (NFR-001): static import-graph scan; fails if any order-placement / wallet / signing symbol from `py-clob-client` (or any Binance order client) is importable from `strikecast/`.
- `test_reproducibility.py` (NFR-003): same seed + config ⇒ identical Brier to 1e-9.
- `test_grid_alignment.py`: every stored `window_open_ts % 300 == 0`.
- `test_calibration_split_disjoint.py` (FR-020): assert train/validation/test window-id sets are pairwise disjoint.

**Backtest hygiene gates (CI-enforced on the eval pipeline):**
- Purge gap ≥ 1 window and embargo ≥ 1 window present in every fold, asserted programmatically.
- Every reported edge (K2, K6) emitted with a CI / significance test (NFR-007); the report builder raises if a point estimate is reported without one.

---

## 3. PR / review checklist

Every PR must check all boxes:

- [ ] Black + Ruff + mypy `--strict` clean; pre-commit passes.
- [ ] New/changed public functions have Google docstrings with units and calibrated/raw labeling.
- [ ] Coverage ≥ 85%; new logic has tests, not just lines touched.
- [ ] No look-ahead: if the change touches data, splits, or estimators, `test_no_leakage.py` reasoning is stated in the PR description.
- [ ] No order/signing surface added (import-guard still green).
- [ ] No Binance dependency introduced; data path stays US-legal.
- [ ] Any new result includes a CI/significance figure and names the label source (Coinbase vs Chainlink).
- [ ] Kronos-fork diff still ≤ 400 lines and behind the adapter (NFR-009); if not, justify.
- [ ] Config changes documented in `config/default.yaml` comments.
- [ ] If a KPI moved, the run report (JSON + Markdown) is attached and the seed is recorded.

---

## 4. Security & legal non-negotiables

1. **No execution surface.** Zero code capable of placing an order on Polymarket, Binance, or any venue. Enforced by `test_no_order_path.py`. This is a hard blocker, not a guideline.
2. **No secrets in repo.** API keys via environment variables only; `.env` is gitignored; a `gitleaks` pre-commit scan blocks accidental key commits.
3. **Least privilege keys.** Coinbase keys used are **view/market-data only**; never create keys with trade or withdraw scope for this project.
4. **Read-only Polymarket.** The Polymarket module instantiates clients without a private key/signer. No wallet, no Polygon transaction code, anywhere.
5. **Local data only.** Raw Coinbase/Polymarket data is not redistributed or committed; it stays in the gitignored data dir.
6. **Dependency integrity.** Dependencies pinned with hashes; `pip-audit` (or `uv`'s audit) runs in CI; Kronos pinned to an exact commit, not a floating branch.
7. **No PII.** The project handles only market data; no user data is collected or stored.

---

## 5. Performance thresholds (concrete)

| Metric | Threshold | Source req |
|--------|-----------|------------|
| Single-window probability, `sample_count=1000` | ≤ 5 s on target GPU | NFR-005 |
| Full backtest, ≥ 50k windows, batched | ≤ 12 h on one GPU | NFR-004 |
| Candle backfill, 1 year of 5-min data | ≤ 30 min (respecting 30 req/s, 300/req) | FR-001 |
| Peak GPU memory, Kronos-small fine-tune | ≤ 16 GB (else gradient accumulation) | NFR-008 |
| Monte-Carlo CI half-width on `p` at `sample_count=1000` | ≤ ±0.02 absolute | FR-013 |
| Reproducibility of Brier across identical runs | identical to 1e-9 | NFR-003 |
| Missing-window rate after gap handling | < 0.5% | NFR-006 |

**Performance review rule:** if `sample_count` is raised to cut MC variance, re-verify NFR-004/005 in the same PR; cost scales linearly with samples.

---

## 6. Documentation standards

- **README.md**: one-command repro (`strikecast run --config config/default.yaml`), the kill criterion stated up front, and the legal posture (research/paper-only, US, no live trading) in the first paragraph.
- **Every run** produces a Markdown report containing: data window, git commit, model checkpoint hash, seed, the full KPI table (K1–K7) with CIs, reliability diagrams (uncalibrated + calibrated), the Coinbase–Chainlink basis distribution, and the KILL flag. A result that exists only in a notebook is not a result.
- **Decisions** that change a tradeoff in `design.md` §6 are recorded as short ADRs in `docs/adr/NNNN-title.md`.
- **The oracle gap** (Coinbase vs Chainlink) must be re-stated in any report that quotes paper PnL, so no number is read out of context.
- **Negative results** are documented with the same rigor as positive ones. "Kronos did not beat GARCH at 5 min, Brier skill = −0.001 (95% CI [−0.004, +0.002])" is a complete, publishable deliverable.

---

## 7. Definition of Done (per phase)

- **Phase 0 (data):** candle + market + label stores populated, gap report < 0.5% missing, grid-alignment test green.
- **Phase 1 (baseline):** random-walk + GARCH-MC scored on test set; scoring harness emits full KPI table with CIs; leakage + reproducibility tests green.
- **Phase 2 (Kronos zero-shot):** `KronosBinaryEstimator` returns calibrated `ProbResult` with CI; zero-shot KPIs reported next to baselines.
- **Phase 3 (fine-tune + calibrate):** fine-tuned checkpoint + fitted calibrator (on disjoint validation split); reliability diagrams show ECE within target; train/test split provably disjoint.
- **Phase 4 (decision):** kill criterion applied automatically; if passed, paper-PnL vs live Polymarket prices reported with CI and Sharpe; final report committed.

---

### Sources
- Calibration evaluation (Brier, ECE, isotonic vs Platt, calibrate on held-out set): <https://www.blog.trainindata.com/probability-calibration-in-machine-learning/>, <https://www.pnas.org/doi/10.1073/pnas.2016191118>
- Leakage-free evaluation (purged + embargoed CV): <https://en.wikipedia.org/wiki/Purged_cross-validation>
- Coinbase rate limits (10/30 req/s, 300/req): <https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles>
- Kronos fine-tune pipeline & sampling: <https://github.com/shiyu-coder/Kronos>
- GARCH baseline (`arch`): <https://www.mdpi.com/2227-9091/11/12/211>
