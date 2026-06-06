from __future__ import annotations

import ctypes
import os
import shlex
import sys
from collections.abc import Callable, Sequence
from ctypes import wintypes
from pathlib import Path


InputFunc = Callable[[str], str]
PrintFunc = Callable[..., None]
CommandRunner = Callable[[list[str]], None]

DEFAULT_RUN_COUNT = 18
DEFAULT_RUN_BATCHES = 1


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def configure_runtime_cwd() -> Path:
    """In the frozen exe, keep runtime files beside wq-agent.exe."""
    if is_frozen():
        root = Path(sys.executable).resolve().parent
        os.chdir(root)
        return root
    return Path.cwd()


def run_cli(args: list[str]) -> None:
    from wq_agent.cli import app

    app(args=args, prog_name="wq-agent")


def build_run_command(input_func: InputFunc = input, print_func: PrintFunc = print) -> list[str]:
    count = _prompt_int(input_func, print_func, "每批生成数量 count [默认 18]: ", DEFAULT_RUN_COUNT)
    batches = _prompt_int(input_func, print_func, "批次数 batches [默认 1]: ", DEFAULT_RUN_BATCHES)
    idea = input_func("研究想法 idea [可留空]: ").strip()

    command = ["run", "--count", str(count), "--batches", str(batches)]
    if idea:
        command.extend(["--idea", idea])
    return command


def run_menu(
    input_func: InputFunc = input,
    print_func: PrintFunc = print,
    command_runner: CommandRunner = run_cli,
) -> None:
    while True:
        _print_menu(print_func)
        try:
            choice = input_func("请选择: ").strip()
        except (EOFError, KeyboardInterrupt):
            print_func("\n已退出。")
            return

        if choice == "0":
            print_func("已退出。")
            return
        if choice == "1":
            _run_from_menu(["gui"], command_runner, print_func)
        elif choice == "2":
            _run_from_menu(build_run_command(input_func, print_func), command_runner, print_func)
        elif choice == "3":
            _run_from_menu(["backtest", "--pending"], command_runner, print_func)
        elif choice == "4":
            _run_from_menu(["status"], command_runner, print_func)
        elif choice == "5":
            line = input_func("请输入 wq-agent 参数: ").strip()
            if not line:
                continue
            try:
                _run_from_menu(_normalize_cli_args(split_command_line(line)), command_runner, print_func)
            except ValueError as exc:
                print_func(f"命令解析失败: {exc}")
        else:
            print_func("无效选择，请重新输入。")


def main(
    argv: Sequence[str] | None = None,
    *,
    input_func: InputFunc = input,
    print_func: PrintFunc = print,
    command_runner: CommandRunner = run_cli,
) -> None:
    configure_runtime_cwd()
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        command_runner(args)
        return
    run_menu(input_func=input_func, print_func=print_func, command_runner=command_runner)


def split_command_line(line: str) -> list[str]:
    if not line.strip():
        return []
    if os.name == "nt":
        return _split_windows_command_line(line)
    return shlex.split(line)


def _split_windows_command_line(line: str) -> list[str]:
    ctypes.windll.shell32.CommandLineToArgvW.argtypes = [
        wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_int),
    ]
    ctypes.windll.shell32.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    ctypes.windll.kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    ctypes.windll.kernel32.LocalFree.restype = ctypes.c_void_p

    argc = ctypes.c_int()
    argv = ctypes.windll.shell32.CommandLineToArgvW(line, ctypes.byref(argc))
    if not argv:
        raise ValueError("无法解析命令行")
    try:
        return [argv[i] for i in range(argc.value)]
    finally:
        ctypes.windll.kernel32.LocalFree(argv)


def _normalize_cli_args(args: list[str]) -> list[str]:
    if args and Path(args[0]).name.lower() in {"wq-agent", "wq-agent.exe"}:
        return args[1:]
    return args


def _prompt_int(input_func: InputFunc, print_func: PrintFunc, prompt: str, default: int) -> int:
    while True:
        raw = input_func(prompt).strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print_func("请输入整数。")
            continue
        if value < 1:
            print_func("请输入大于 0 的整数。")
            continue
        return value


def _run_from_menu(command: list[str], command_runner: CommandRunner, print_func: PrintFunc) -> None:
    try:
        command_runner(command)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            print_func(f"命令退出，状态码: {exc.code}")


def _print_menu(print_func: PrintFunc) -> None:
    print_func("")
    print_func("wq-agent 一键启动菜单")
    print_func("======================")
    print_func("1 启动 GUI")
    print_func("2 运行完整流程 run")
    print_func("3 回测待处理 backtest --pending")
    print_func("4 查看状态 status")
    print_func("5 命令行模式")
    print_func("0 退出")


if __name__ == "__main__":
    main()
