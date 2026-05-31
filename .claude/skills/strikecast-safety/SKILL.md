---
name: strikecast-safety
description: >
  Legal and security guardrails for the Strikecast project. Use this skill before every PR,
  code review, dependency change, or new module in the Strikecast codebase. Covers: no order/signing
  surface (Polymarket or any venue), read-only Polymarket access, no Binance dependencies,
  US-legal data sources only, no secrets in repo, least-privilege API keys, oracle gap awareness,
  and Kronos fork isolation. If you're adding a dependency, creating a new module under strikecast/,
  touching strikecast/data/polymarket_read.py, or reviewing a PR, this skill applies. Also use
  when someone asks about Strikecast's legal posture, data sourcing, or security model.
---

# Strikecast Safety Guards

These guardrails are hard constraints — not guidelines, not best practices. They exist because
Strikecast operates in a legally sensitive space (US-based developer, prediction markets, crypto
exchange data) and a single violation could make the entire project unusable.

Every item below is repeated across steering.md, requirements.md, design.md, and quality.md
because each was deemed important enough to state in every context. This skill is the single
enforcement point.

## The non-negotiables

### 1. No execution surface — anywhere, ever

Zero code paths capable of placing an order on Polymarket, Binance, or any trading venue.
This is not about discipline — it's about making trading *impossible by construction*.

**What this means concretely:**
- No wallet integration, no private keys, no order signing
- The `py-clob-client` library (if used) must only have read methods imported
- No `signing`, `order`, `create_order`, `place_order`, or equivalent symbols importable from any Strikecast module
- Enforced by `test_no_order_path.py`: a static import-graph scan that fails if any order/signing symbol from `py-clob-client` (or any exchange order client) is reachable from `strikecast/`

**On every PR, verify:**
- [ ] `test_no_order_path.py` still passes
- [ ] No new dependency introduces an order/execution capability
- [ ] No new import touches signing, wallet, or transaction code

### 2. Read-only Polymarket

All Polymarket interaction is strictly read-only — public market prices and the resolution oracle
feed, used solely for benchmarking paper-PnL against implied probabilities.

**Allowed:**
- `GET` requests to Gamma API (event/market metadata, token IDs, prices)
- `GET` requests to CLOB (midpoint prices, price reads)
- WebSocket subscription to `crypto_prices_chainlink` (BTC/USD oracle prices for resolution labels)

**Forbidden:**
- Any authenticated endpoint
- Any POST/PUT/DELETE to CLOB
- Instantiating a client with a private key or signer
- Any Polygon transaction code

The Polymarket module (`strikecast/data/polymarket_read.py`) must be instantiated without
a private key/signer. The module name itself signals the constraint.

### 3. No Binance dependency

Binance.com is not accessible to US persons. Any code, third-party library, or borrowed snippet
that hardcodes Binance must be rewritten against a US-legal source.

**Primary data source:** Coinbase Advanced Trade API (via `ccxt` for venue abstraction)
**Acceptable alternatives:** Kraken, other US-legal exchanges (swappable via `CandleSource` interface)

On every dependency change, check that no transitive dependency pulls in Binance-specific code
that could create a compliance concern.

### 4. US-legal data sourcing

- Coinbase market data: respect API terms, cache locally, respect rate limits (10 req/s public, 30 req/s authenticated, 300 candles/request)
- Raw Coinbase/Polymarket data stays local — never redistributed or committed to the repo
- Data files (`.parquet`, `.h5`, `.feather`) are gitignored
- Nothing in this project constitutes financial advice or a trading service

### 5. No secrets in repo

- API keys via environment variables only
- `.env` is gitignored
- `gitleaks` pre-commit scan blocks accidental key commits
- Coinbase keys must be **view/market-data only** — never create keys with trade or withdraw scope

### 6. Dependency integrity

- All dependencies pinned with hashes
- `pip-audit` (or `uv`'s audit) runs in CI
- Kronos pinned to an exact commit, not a floating branch

### 7. Kronos fork isolation (NFR-009)

Changes to the Kronos codebase live behind an adapter module. The upstream diff must stay
<= 400 lines so rebasing onto new Kronos releases (new checkpoints, longer context) stays tractable.

**On every PR that touches `kronos_fork/`:**
- [ ] Diff against upstream is still <= 400 lines
- [ ] Changes are behind the adapter, not inline modifications
- [ ] If the diff exceeds 400 lines, justify in the PR description

## The oracle gap

Polymarket resolves on **Chainlink** BTC/USD oracle prices. The model trains and infers on
**Coinbase** candles. This basis risk is a first-class concern, not an afterthought.

- `resolution_labels` stores both `oracle_close` (Chainlink) and `coinbase_close`
- Model-internal scoring uses Coinbase (consistent with training data)
- Paper-PnL evaluation uses Chainlink (consistent with how bets actually resolve)
- The Coinbase-Chainlink basis distribution must be reported in any result that quotes paper PnL
- Any report that omits the oracle gap context is incomplete

## PR checklist (complete list)

Every PR must satisfy:

- [ ] Black + Ruff + mypy `--strict` clean; pre-commit passes
- [ ] `test_no_order_path.py` passes (no execution surface)
- [ ] `test_no_leakage.py` reasoning stated in PR description (if touching data/splits/estimators)
- [ ] No Binance dependency introduced; data path stays US-legal
- [ ] No secrets committed (gitleaks clean)
- [ ] Kronos-fork diff still <= 400 lines and behind adapter
- [ ] New/changed public functions have Google docstrings with units and calibrated/raw labeling
- [ ] Coverage >= 85% on `strikecast/`
- [ ] Config changes documented in `config/default.yaml` comments
- [ ] Any new result includes CI/significance and names the label source (Coinbase vs Chainlink)
- [ ] If a KPI moved, run report (JSON + Markdown) attached with seed recorded
