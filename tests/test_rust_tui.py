from __future__ import annotations

import os
import signal
from pathlib import Path

from alphagen_agent.cli import _dataset_categories, _run_rust_tui, _rust_tui_command


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


def test_rust_tui_normalizes_signal_exit_code(monkeypatch):
    class Process:
        def wait(self):
            return -2

    monkeypatch.setattr("alphagen_agent.cli.subprocess.Popen", lambda *args, **kwargs: Process())

    assert _run_rust_tui(["alphagen-tui"], {}) == 130


def test_rust_tui_forwards_keyboard_interrupt(monkeypatch):
    class Process:
        def __init__(self):
            self.wait_calls = 0
            self.signal = None
            self.terminated = False

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise KeyboardInterrupt
            return -2

        def poll(self):
            return None

        def send_signal(self, value):
            self.signal = value

        def terminate(self):
            self.terminated = True

    process = Process()
    monkeypatch.setattr("alphagen_agent.cli.subprocess.Popen", lambda *args, **kwargs: process)

    assert _run_rust_tui(["alphagen-tui"], {}) == 130
    if os.name == "posix":
        assert process.signal == signal.SIGINT
    else:
        assert process.terminated is True
    assert process.wait_calls == 2
