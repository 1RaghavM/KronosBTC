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
