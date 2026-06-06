from __future__ import annotations

import os
import sys

import pytest

from wq_agent import launcher


class InputQueue:
    def __init__(self, values: list[str]):
        self.values = values

    def __call__(self, prompt: str) -> str:
        assert prompt
        if not self.values:
            raise EOFError
        return self.values.pop(0)


def test_launcher_no_args_menu_can_exit():
    commands: list[list[str]] = []
    output: list[str] = []

    launcher.main(
        [],
        input_func=InputQueue(["0"]),
        print_func=lambda *parts: output.append(" ".join(str(p) for p in parts)),
        command_runner=commands.append,
    )

    assert commands == []
    assert any("wq-agent" in line for line in output)


def test_launcher_with_args_delegates_to_cli():
    commands: list[list[str]] = []

    launcher.main(
        ["wiki", "stats", "--verbose"],
        input_func=InputQueue([]),
        print_func=lambda *parts: None,
        command_runner=commands.append,
    )

    assert commands == [["wiki", "stats", "--verbose"]]


def test_launcher_run_menu_builds_command_with_defaults_and_idea():
    commands: list[list[str]] = []

    launcher.run_menu(
        input_func=InputQueue(["2", "", "", "analyst revision", "0"]),
        print_func=lambda *parts: None,
        command_runner=commands.append,
    )

    assert commands == [
        ["run", "--count", "18", "--batches", "1", "--idea", "analyst revision"]
    ]


def test_launcher_command_mode_accepts_full_wq_agent_prefix(monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(os, "name", "posix")

    launcher.run_menu(
        input_func=InputQueue(["5", 'wq-agent.exe generate --idea "low turnover" -n 3', "0"]),
        print_func=lambda *parts: None,
        command_runner=commands.append,
    )

    assert commands == [["generate", "--idea", "low turnover", "-n", "3"]]


def test_launcher_configures_cwd_to_exe_dir_when_frozen(tmp_path, monkeypatch):
    exe_dir = tmp_path / "dist" / "wq-agent"
    exe_dir.mkdir(parents=True)
    exe_path = exe_dir / "wq-agent.exe"
    exe_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_path))

    old_cwd = os.getcwd()
    try:
        root = launcher.configure_runtime_cwd()
        assert root == exe_dir
        assert os.getcwd() == str(exe_dir)
    finally:
        os.chdir(old_cwd)


@pytest.mark.skipif(os.name != "nt", reason="Windows command parsing uses CommandLineToArgvW")
def test_launcher_windows_command_line_parser_handles_quotes():
    assert launcher.split_command_line('generate --idea "low turnover" -n 3') == [
        "generate",
        "--idea",
        "low turnover",
        "-n",
        "3",
    ]
