from __future__ import annotations

from alphagen_agent.tui import AlphaGenAgentTui, _DATASET_CATEGORIES, _MARKETS, _truncate


def test_tui_declares_core_bindings():
    actions = {binding[1] for binding in AlphaGenAgentTui.BINDINGS}
    assert {"generate", "run_batch", "refine", "backtest_pending", "refresh", "quit"} <= actions


def test_truncate_shortens_long_expressions():
    assert _truncate("rank(close)", 20) == "rank(close)"
    assert _truncate("abcdefghijklmnopqrstuvwxyz", 8) == "abcdefg..."


def test_tui_declares_dataset_categories():
    values = {value for _, value in _DATASET_CATEGORIES}
    assert {"all", "pv", "fundamental", "analyst", "model", "news", "option", "sentiment", "socialmedia"} <= values


def test_tui_declares_markets():
    values = {value for _, value in _MARKETS}
    assert {"default", "USA", "CHN", "EUR", "ASI", "GLB"} <= values
