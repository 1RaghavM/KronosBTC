# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kronos is a decoder-only foundation model for financial candlestick (K-line) forecasting, pre-trained on data from 45+ global exchanges. It uses a two-stage architecture: a Binary Spherical Quantization (BSQ) tokenizer converts continuous OHLCV data into hierarchical discrete tokens, then an autoregressive Transformer predicts future tokens.

## Common Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run tests (downloads models from HuggingFace on first run)
```bash
pytest tests/ -v
```

### Run a single test
```bash
pytest tests/test_kronos_regression.py::test_kronos_predictor_regression -v
pytest tests/test_kronos_regression.py::test_kronos_predictor_mse -v
```

### Run Web UI
```bash
cd webui && python run.py   # serves on http://localhost:7070
# webui has its own requirements.txt (flask, plotly, etc.)
```

### Finetuning (multi-GPU with torchrun)
```bash
# 1. Preprocess data (requires pyqlib)
python finetune/qlib_data_preprocess.py

# 2. Finetune tokenizer, then predictor
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py

# 3. Backtest
python finetune/qlib_test.py --device cuda:0
```

## Architecture

### Core Model (`model/`)

Three public classes exported from `model/__init__.py`:

- **`KronosTokenizer`** (`model/kronos.py`): Encoder-decoder Transformer with BSQ. Encodes OHLCV sequences into hierarchical token pairs (s1, s2) via `encode()`, decodes back via `decode()`. Uses `half=True` for split indices (returns `[s1_indices, s2_indices]` tuple).

- **`Kronos`** (`model/kronos.py`): Autoregressive Transformer that predicts next-token logits. Uses a **dual-head architecture**: first predicts s1 (coarse) logits, then predicts s2 (fine) logits conditioned on s1 via a `DependencyAwareLayer` (cross-attention). Inference is split into `decode_s1()` -> `decode_s2()` for sequential token generation.

- **`KronosPredictor`** (`model/kronos.py`): High-level inference wrapper. Handles normalization (z-score per feature), timestamp encoding, and autoregressive generation. Supports single (`predict()`) and batch (`predict_batch()`) prediction. Auto-detects device (CUDA > MPS > CPU).

### Key internals (`model/module.py`)

- **`BSQuantizer`** / **`BinarySphericalQuantizer`**: Core quantization module. Tokens are split into s1 (coarse, `s1_bits`) and s2 (fine, `s2_bits`) components. The codebook is implicit (binary codes), not a learned embedding table.
- **`HierarchicalEmbedding`**: Embeds s1 and s2 token IDs separately, then fuses via linear projection.
- **`DualHead`**: Outputs s1 logits unconditionally and s2 logits via `cond_forward()`.
- **`TransformerBlock`**: Pre-norm (RMSNorm) block with RoPE attention and SwiGLU FFN.
- **`TemporalEmbedding`**: Encodes 5 time features (minute, hour, weekday, day, month).

### Inference flow (`auto_regressive_inference` in `model/kronos.py`)

1. Encode input with tokenizer (`half=True`) to get `(s1_ids, s2_ids)`
2. Loop for `pred_len` steps: predict s1 via `decode_s1()`, sample, then predict s2 via `decode_s2()`, sample
3. Uses a sliding buffer of size `max_context` for long sequences
4. Generates `sample_count` parallel paths and averages predictions

### Finetuning (`finetune/`)

- `config.py`: Central config with all paths and hyperparameters (must be edited before use)
- `dataset.py`: `QlibDataset` — loads pickled data, returns normalized OHLCV + timestamps
- `train_tokenizer.py` / `train_predictor.py`: DDP training scripts (use `torchrun`)
- `qlib_test.py`: Inference + backtesting with top-K strategy

### CSV Finetuning (`finetune_csv/`)

Alternative finetuning pipeline that works directly with CSV files instead of Qlib. Uses YAML config files in `finetune_csv/configs/`.

### Tests (`tests/`)

- `test_kronos_regression.py`: Regression tests that download `Kronos-small` and `Kronos-Tokenizer-base` from HuggingFace at pinned revisions, then verify deterministic outputs (top_k=1) against saved CSV baselines. Also includes MSE-based tests with random sampling.
- Test data lives in `tests/data/` (input CSV + expected output CSVs per context length).

## Model Variants

| Model | Tokenizer | Context | Params |
|-------|-----------|---------|--------|
| Kronos-mini | Kronos-Tokenizer-2k | 2048 | 4.1M |
| Kronos-small | Kronos-Tokenizer-base | 512 | 24.7M |
| Kronos-base | Kronos-Tokenizer-base | 512 | 102.3M |

Models are hosted on HuggingFace under `NeoQuasar/` and loaded via `PyTorchModelHubMixin`.

## Key Conventions

- Input DataFrames require columns `['open', 'high', 'low', 'close']`; `volume` and `amount` are optional (filled with zeros if missing).
- All normalization is z-score (per-feature mean/std), clipped to `[-clip, clip]` (default 5).
- Temporal features are computed from pandas timestamps: `[minute, hour, weekday, day, month]`.
- The tokenizer's `half=True` mode splits the codebook into two independent index spaces (s1, s2), which is how the predictor model consumes tokens.
