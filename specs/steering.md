# STEERING.md — Strikecast

> **Working codename:** Strikecast — a calibrated probability engine for 5-minute BTC binary outcomes, built on the open-source [Kronos](https://github.com/shiyu-coder/Kronos) foundation model.
> **Companion docs:** [`requirements.md`](./requirements.md) (the what), [`design.md`](./design.md) (the how), [`QUALITY.md`](./QUALITY.md) (the bar).
> **Status:** Pre-build planning. Single-developer research project. US jurisdiction — **research and paper-trading only, no live order placement.**

---

## 1. Mission

Turn Kronos from a candlestick *path generator* into a **calibrated probability estimator** that answers one question — "what is P(BTC close is above strike X at the end of the next 5-minute window)?" — well enough to beat both a naive coin-flip and a volatility-model baseline, and to disagree usefully with Polymarket's implied price.

## 2. The user and the core problem

**Primary user:** the developer (you), running a personal quant-research loop.

**The core problem this project exists to solve:** Kronos outputs sampled OHLCV paths, not probabilities. A single averaged path is near-useless for a binary question. The product gap is the *wrapper and the honesty layer* — the Monte Carlo aggregation that converts sampled paths into a probability, plus the calibration and benchmarking that tell you whether that probability is trustworthy and whether it carries any edge over the market price.

**Why this is non-trivial:** at a 5-minute horizon BTC is close to a random walk, and the interesting strikes (near spot) are exactly where signal is thinnest. The deliverable is not "a model that predicts BTC." It is "an apparatus that measures, with statistical rigor, whether *any* edge exists at this horizon — and quantifies it." A credible negative result is a successful outcome of this project.

## 3. Success metrics (KPIs)

All metrics are measured on a held-out, time-ordered test set using purged + embargoed walk-forward evaluation (see `design.md` §7). "Near-the-money" = strike within ±0.10% of spot at window open, which is the Polymarket "Up or Down" regime.

| # | Metric | Baseline | Target (P0) | Stretch (P1) |
|---|--------|----------|-------------|--------------|
| K1 | **Brier score**, near-the-money 5-min | 0.2500 (coin flip) | ≤ 0.2490 | ≤ 0.2470 |
| K2 | **Brier skill score** vs GARCH-MC baseline | 0.0 | > 0.0 (beats GARCH) | ≥ +0.02 |
| K3 | **Expected Calibration Error (ECE)**, 15 bins | — | ≤ 0.020 | ≤ 0.010 |
| K4 | **Log loss**, near-the-money | 0.6931 (coin flip) | ≤ 0.6910 | ≤ 0.6870 |
| K5 | **Directional accuracy**, near-the-money | 50.0% | ≥ 51.0% | ≥ 53.0% |
| K6 | **Paper PnL** vs Polymarket implied prob, after 2% round-trip cost assumption, edge-threshold gating | break-even | > 0 over ≥ 2,000 bets | Sharpe ≥ 1.0 (per-bet annualized) |
| K7 | **Far-the-money sanity** (strike ±1% +): model agrees with trivial answer | — | ECE ≤ 0.01 | — |

> **Read these honestly.** K1's target gap (0.2500 → 0.2490) is *deliberately tiny*. At 5 minutes near the money, a 0.004 Brier improvement is a real, hard-won edge, not a disappointment. The decision-critical KPIs are **K2 (beats a real vol model)** and **K3 (well-calibrated)**. A model that is well-calibrated but barely better than a coin flip is still useful for K6; a model that is accurate but overconfident is dangerous and fails the project.

**Kill criterion:** if after full fine-tuning and calibration the model cannot achieve K2 > 0 (i.e. cannot beat the GARCH Monte-Carlo digital-pricing baseline) on the test set, the foundation-model approach is not justified for this horizon. Document the negative result and stop. This is an explicit, acceptable end state.

## 4. Hard constraints

### Legal / jurisdiction (non-negotiable)
- **US-based developer.** Polymarket is not available for trading to US persons, and Binance.com is not accessible. The system **must not place orders** on Polymarket or any restricted venue. All Polymarket interaction is **read-only** (public market prices and the resolution oracle feed) for benchmarking.
- **No Binance dependency.** All training and inference candle data must come from US-accessible sources (Coinbase Advanced Trade API is the primary; see `design.md` §4). Any third-party code that hardcodes Binance must be replaced.
- Nothing in this project constitutes financial advice or a trading service for others. It is personal research.

### Data & licensing
- Kronos is MIT-licensed ([repo](https://github.com/shiyu-coder/Kronos), [model card](https://huggingface.co/NeoQuasar/Kronos-base)) — modification and redistribution of derivatives are permitted with attribution. Forking and editing the repo is in scope and expected.
- Coinbase market data is subject to Coinbase API terms; cache locally, respect rate limits (10 req/s public, 30 req/s authenticated, 300 candles/request — [docs](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles)).

### Budget & compute
- Single consumer/cloud GPU. Fine-tuning starts on **Kronos-small (24.7M params)** for fast iteration; Kronos-base (102.3M) only if small shows promise. Kronos-large is not open-source — out of scope.
- Target: full data pull + fine-tune + backtest cycle runnable on one GPU in < 24h wall-clock.

### Timeline (suggested phasing, not contractual)
- **Phase 0 (data):** US-legal data pipeline + Polymarket read-only ingestion. ~1 week.
- **Phase 1 (baseline):** random-walk + GARCH Monte-Carlo digital pricer + full evaluation harness. ~1 week. *Build the scoreboard before the contestant.*
- **Phase 2 (Kronos zero-shot):** wrap `KronosPredictor` in Monte-Carlo probability estimator, evaluate zero-shot. ~3 days.
- **Phase 3 (fine-tune + calibrate):** fine-tune on BTC 5-min, add post-hoc calibration layer. ~2 weeks.
- **Phase 4 (decision):** apply kill criterion; if it passes, paper-trade against live Polymarket prices.

## 5. Out of scope (explicit)

- ❌ Live order placement, wallet integration, private keys, or any execution path. (Legal constraint.)
- ❌ Multi-asset support. BTC/USD only for v1. (ETH etc. is a later concern.)
- ❌ Horizons other than 5 minutes for v1. The harness should be horizon-parameterized, but only 5-min is validated.
- ❌ Latency-optimized production inference. This is research-grade; sub-second inference is not required (Polymarket trades are paper-only).
- ❌ Kronos-large (not open-source) and pre-training from scratch (no budget; fine-tuning only).
- ❌ Portfolio optimization, risk-factor neutralization, position sizing models beyond a simple edge-threshold + fixed-fraction rule for the paper-PnL KPI.
- ❌ A user-facing product, accounts, or serving multiple users.
- ❌ Reproducing or redistributing Coinbase/Polymarket raw data publicly.

## 6. Architectural principles (the north star for every decision)

1. **Calibration over accuracy.** When forced to choose, prefer a model that is honest about its uncertainty over one with higher raw hit-rate. Every probability the system emits must survive a reliability-diagram check.
2. **The baseline is sacred.** No Kronos result is reported without a side-by-side GARCH-MC and random-walk number on the *identical* test set. A foundation model that can't beat `arch`'s GARCH at 5 minutes has not earned its GPU.
3. **Probabilities, not paths.** Kronos is treated as a Monte-Carlo path simulator. The system never acts on a single forecast path or a point estimate. `sample_count` is high (≥ 500); the *distribution* is the product.
4. **The market price is the benchmark that pays.** Truth is for calibration; the Polymarket implied probability is what determines paper PnL. Optimize and report edge **vs the market price**, not vs reality alone.
5. **No leakage, ever.** Label depends on a future bar, so all evaluation uses purged + embargoed walk-forward splits ([López de Prado purged CV](https://en.wikipedia.org/wiki/Purged_cross-validation)). Any backtest that touches future data is treated as a defect, not a result.
6. **Mind the oracle gap.** Polymarket resolves on a **Chainlink** BTC/USD oracle, but the model trains/infers on **Coinbase** candles. This basis is a first-class modeled risk (see `design.md` §6), not an afterthought. Resolution labels for paper-PnL come from the Chainlink feed, not Coinbase.
7. **A clean negative is a win.** The project succeeds if it produces a statistically defensible answer to "is there edge here?" — including "no." Don't torture the backtest into a false positive.
8. **Fork, don't vendor-lock.** Kronos is edited in a fork. Keep changes minimal, isolated behind an adapter, and rebased onto upstream so improvements (new checkpoints, longer context) can be pulled in.

---

### Sources
- Kronos repo & model zoo: <https://github.com/shiyu-coder/Kronos>; paper: <https://arxiv.org/abs/2508.02739> (AAAI 2026).
- Coinbase Advanced Trade candles API: <https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles>
- Polymarket CLOB/Gamma pricing: <https://docs.polymarket.com/api-reference/pricing/get-market-price>; BTC Up/Down market structure & Chainlink resolution discussed at <https://gist.github.com/Archetapp/7680adabc48f812a561ca79d73cbac69>
- Calibration methodology (CORP/isotonic reliability diagrams): <https://www.pnas.org/doi/10.1073/pnas.2016191118>
- GARCH for crypto volatility (`arch` package, Realized-GARCH): <https://www.mdpi.com/2227-9091/11/12/211>
- Purged/embargoed cross-validation: <https://en.wikipedia.org/wiki/Purged_cross-validation>
