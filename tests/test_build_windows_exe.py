from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_windows_exe.py"
SPEC = importlib.util.spec_from_file_location("build_windows_exe", SCRIPT_PATH)
assert SPEC is not None
build_windows_exe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(build_windows_exe)


def _patch_paths(monkeypatch, project_root: Path) -> Path:
    app_dist = project_root / "dist" / "wq-agent"
    monkeypatch.setattr(build_windows_exe, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(build_windows_exe, "DIST_ROOT", project_root / "dist")
    monkeypatch.setattr(build_windows_exe, "APP_DIST", app_dist)
    monkeypatch.setattr(build_windows_exe, "BUILD_ROOT", project_root / "build" / "pyinstaller")
    return app_dist


def _create_required_sources(project_root: Path) -> None:
    for rel in build_windows_exe.REQUIRED_SOURCE_PATHS:
        path = project_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("public", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")


def _create_required_distribution(app_dist: Path) -> None:
    for rel in build_windows_exe.REQUIRED_DIST_PATHS:
        path = app_dist / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("public", encoding="utf-8")


def test_clean_previous_build_refuses_to_delete_runtime_private_files(tmp_path, monkeypatch):
    app_dist = _patch_paths(monkeypatch, tmp_path)
    app_dist.mkdir(parents=True)
    (app_dist / ".env").write_text("OPENAI_API_KEY=secret", encoding="utf-8")

    with pytest.raises(SystemExit, match="Refusing to delete"):
        build_windows_exe._clean_previous_build()

    assert (app_dist / ".env").exists()


def test_clean_previous_build_removes_public_bundle_when_no_runtime_private_files(
    tmp_path, monkeypatch
):
    app_dist = _patch_paths(monkeypatch, tmp_path)
    build_root = tmp_path / "build" / "pyinstaller"
    app_dist.mkdir(parents=True)
    build_root.mkdir(parents=True)
    (app_dist / "wq-agent.exe").write_text("exe", encoding="utf-8")

    build_windows_exe._clean_previous_build()

    assert not app_dist.exists()
    assert not build_root.exists()


def test_assert_required_sources_fails_for_missing_public_resources(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="Missing required build resources"):
        build_windows_exe._assert_required_sources()


def test_copy_public_resources_filters_private_files(tmp_path, monkeypatch):
    app_dist = _patch_paths(monkeypatch, tmp_path)
    _create_required_sources(tmp_path)

    (tmp_path / "wiki" / "entries").mkdir()
    (tmp_path / "wiki" / "entries" / "secret.md").write_text("secret", encoding="utf-8")
    (tmp_path / "wiki" / "private.key").write_text("secret", encoding="utf-8")
    (tmp_path / "wiki" / "credentials.json").write_text("secret", encoding="utf-8")
    (tmp_path / "wiki" / ".codex").mkdir()
    (tmp_path / "wiki" / ".codex" / "secret.md").write_text("secret", encoding="utf-8")
    (tmp_path / "templates" / "alpha_templates.yaml").write_text("public", encoding="utf-8")
    (tmp_path / "templates" / ".env").write_text("secret", encoding="utf-8")

    build_windows_exe._copy_public_resources()

    assert (app_dist / "wiki" / "index.md").exists()
    assert (app_dist / "wiki" / "dictionary" / "base.txt").exists()
    assert (app_dist / "templates" / "alpha_templates.yaml").exists()
    assert (app_dist / ".env.example").exists()
    assert not (app_dist / "wiki" / "entries").exists()
    assert not (app_dist / "wiki" / "private.key").exists()
    assert not (app_dist / "wiki" / "credentials.json").exists()
    assert not (app_dist / "wiki" / ".codex").exists()
    assert not (app_dist / "templates" / ".env").exists()


def test_distribution_privacy_assertion_scans_resources_and_runtime_root(
    tmp_path, monkeypatch
):
    app_dist = _patch_paths(monkeypatch, tmp_path)
    _create_required_distribution(app_dist)
    (app_dist / "wiki" / "lessons").mkdir()
    (app_dist / "wiki" / "lessons" / "secret.md").write_text("secret", encoding="utf-8")
    (app_dist / "wq_agent.db").write_text("secret", encoding="utf-8")

    with pytest.raises(SystemExit, match="Distribution contains private/runtime files"):
        build_windows_exe._assert_distribution_is_private_safe()


def test_distribution_privacy_assertion_accepts_required_public_bundle(tmp_path, monkeypatch):
    app_dist = _patch_paths(monkeypatch, tmp_path)
    _create_required_distribution(app_dist)

    build_windows_exe._assert_distribution_is_private_safe()


def test_pyinstaller_command_includes_gui_upload_parser_hidden_imports(monkeypatch):
    commands = []

    def fake_run(command, cwd, check):
        commands.append(command)

    monkeypatch.setattr(build_windows_exe.subprocess, "run", fake_run)

    build_windows_exe._run_pyinstaller()

    command = commands[0]
    assert command.count("--hidden-import") >= 3
    assert "sqlite_vec" in command
    assert "pypdf" in command
    assert "docx" in command


def test_pyinstaller_command_uses_console_launcher_entry(monkeypatch):
    commands = []

    def fake_run(command, cwd, check):
        commands.append(command)

    monkeypatch.setattr(build_windows_exe.subprocess, "run", fake_run)

    build_windows_exe._run_pyinstaller()

    command = commands[0]
    assert "--console" in command
    assert "--windowed" not in command
    assert "--noconsole" not in command
    assert str(build_windows_exe.PROJECT_ROOT / "src" / "wq_agent" / "launcher.py") in command


def teardown_module() -> None:
    if SCRIPT_PATH.parent.joinpath("__pycache__").exists():
        shutil.rmtree(SCRIPT_PATH.parent / "__pycache__", ignore_errors=True)
