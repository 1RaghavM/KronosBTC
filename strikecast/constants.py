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
PREDICTION_COLUMNS: list[str] = [
    "run_id",
    "window_open_ts",
    "estimator",
    "strike",
    "p",
    "p_raw",
    "p_ci_low",
    "p_ci_high",
    "n_samples",
    "label",
    "moneyness",
]
