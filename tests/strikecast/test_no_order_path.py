"""NFR-001: Zero execution surface.

Scans all Python source files under strikecast/ and asserts that no
order-placement, signing, or wallet symbol from py-clob-client (or any
Binance client) is imported. A failure here is a release blocker.
"""
import ast
from pathlib import Path

import pytest

STRIKECAST_ROOT = Path("strikecast")

BANNED_MODULE_FRAGMENTS = [
    "py_clob_client.order",
    "py_clob_client.signing",
    "py_clob_client.signer",
    "py_clob_client.wallet",
    "py_order_utils",
    "binance",
]

BANNED_NAME_FRAGMENTS = [
    "create_order",
    "place_order",
    "submit_order",
    "sign_order",
    "cancel_order",
    "ApiSigner",
    "ClobClient",
    "BinanceClient",
    "private_key",
]


def _collect_python_files() -> list[Path]:
    return sorted(STRIKECAST_ROOT.rglob("*.py"))


def _scan_file(path: Path) -> list[str]:
    """Return a list of violation descriptions found in the file."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for banned in BANNED_MODULE_FRAGMENTS:
                    if banned in alias.name.lower():
                        violations.append(
                            f"{path}:{node.lineno} imports '{alias.name}'"
                        )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for banned in BANNED_MODULE_FRAGMENTS:
                if banned in module.lower():
                    violations.append(
                        f"{path}:{node.lineno} imports from '{module}'"
                    )
            for alias in node.names:
                for banned in BANNED_NAME_FRAGMENTS:
                    if banned.lower() in alias.name.lower():
                        violations.append(
                            f"{path}:{node.lineno} imports name '{alias.name}' from '{module}'"
                        )
        elif isinstance(node, ast.Name):
            for banned in BANNED_NAME_FRAGMENTS:
                if node.id.lower() == banned.lower():
                    violations.append(
                        f"{path}:{node.lineno} references name '{node.id}'"
                    )

    return violations


class TestNoOrderPath:
    def test_no_banned_imports_in_strikecast(self) -> None:
        all_violations: list[str] = []
        for py_file in _collect_python_files():
            all_violations.extend(_scan_file(py_file))

        if all_violations:
            msg = "NFR-001 VIOLATION: order/signing/wallet symbols found:\n"
            msg += "\n".join(f"  - {v}" for v in all_violations)
            pytest.fail(msg)

    def test_strikecast_has_python_files(self) -> None:
        """Sanity check: the scan actually found files to scan."""
        files = _collect_python_files()
        assert len(files) > 0, "No .py files found under strikecast/"
