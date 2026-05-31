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
                start_ts=1768435200,
                end_ts=1768435200 + 300,
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
                start_ts=1768435200,
                end_ts=1768435200 + 300,
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
