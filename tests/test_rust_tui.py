from __future__ import annotations

from pathlib import Path

from alphagen_agent.cli import _dataset_categories, _rust_tui_command


def test_dataset_categories_parse_csv():
    assert _dataset_categories(None) is None
    assert _dataset_categories("") is None
    assert _dataset_categories("pv, analyst") == ["pv", "analyst"]


def test_rust_tui_honors_binary_override(monkeypatch):
    monkeypatch.setenv("ALPHAGEN_TUI_BIN", "/tmp/custom-alphagen-tui")

    assert _rust_tui_command() == ["/tmp/custom-alphagen-tui"]


def test_rust_tui_manifest_is_packaged():
    manifest = Path(__file__).resolve().parents[1] / "tui-rs" / "Cargo.toml"

    assert manifest.is_file()
