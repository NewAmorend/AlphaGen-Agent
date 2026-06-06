from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_ROOT = PROJECT_ROOT / "dist"
APP_DIST = DIST_ROOT / "wq-agent"
BUILD_ROOT = PROJECT_ROOT / "build" / "pyinstaller"

PUBLIC_RESOURCE_DIRS = ("templates", "wiki")
PUBLIC_RESOURCE_FILES = (".env.example",)
REQUIRED_SOURCE_PATHS = (
    Path("src/wq_agent/gui/static/index.html"),
    Path("src/wq_agent/gui/static/app.js"),
    Path("src/wq_agent/gui/static/styles.css"),
    Path("templates/alpha_templates.yaml"),
    Path("wiki/index.md"),
    Path("wiki/dictionary/base.txt"),
    Path("wiki/dictionary/synonyms.yaml"),
    Path(".env.example"),
)
REQUIRED_DIST_PATHS = (
    Path("wq-agent.exe"),
    Path("_internal/wq_agent/gui/static/index.html"),
    Path("_internal/wq_agent/gui/static/app.js"),
    Path("_internal/wq_agent/gui/static/styles.css"),
    Path("templates/alpha_templates.yaml"),
    Path("wiki/index.md"),
    Path("wiki/dictionary/base.txt"),
    Path("wiki/dictionary/synonyms.yaml"),
    Path(".env.example"),
)
FORBIDDEN_NAMES = {
    ".env",
    "wq_agent.db",
    "credentials.json",
    "private_wiki",
    ".git",
    ".claude",
    ".codex",
    "__pycache__",
}
FORBIDDEN_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".log", ".pem", ".key", ".pyc", ".pyo"}
RUNTIME_PRIVATE_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".log", ".pem", ".key"}
PUBLIC_RESOURCE_SUFFIXES = {".md", ".yaml", ".yml", ".txt"}


def main() -> None:
    _ensure_project_root()
    _assert_required_sources()
    _clean_previous_build()
    _run_pyinstaller()
    _copy_public_resources()
    _assert_distribution_is_private_safe()
    print(f"Built Windows folder distribution: {APP_DIST}")


def _ensure_project_root() -> None:
    if not (PROJECT_ROOT / "pyproject.toml").exists():
        raise SystemExit(f"Cannot find pyproject.toml at {PROJECT_ROOT}")


def _clean_previous_build() -> None:
    private_runtime_paths = _find_private_runtime_paths(APP_DIST)
    if private_runtime_paths:
        formatted = "\n".join(f" - {path}" for path in private_runtime_paths)
        raise SystemExit(
            "Refusing to delete an existing runtime distribution with private data:\n"
            f"{formatted}\n"
            "Move or back up these files before rebuilding."
        )
    shutil.rmtree(APP_DIST, ignore_errors=True)
    shutil.rmtree(BUILD_ROOT, ignore_errors=True)


def _run_pyinstaller() -> None:
    separator = ";" if os.name == "nt" else ":"
    static_source = PROJECT_ROOT / "src" / "wq_agent" / "gui" / "static"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--console",
        "--name",
        "wq-agent",
        "--distpath",
        str(DIST_ROOT),
        "--workpath",
        str(BUILD_ROOT),
        "--specpath",
        str(BUILD_ROOT),
        "--paths",
        str(PROJECT_ROOT / "src"),
        "--hidden-import",
        "sqlite_vec",
        "--collect-all",
        "sqlite_vec",
        "--add-data",
        f"{static_source}{separator}wq_agent/gui/static",
        str(PROJECT_ROOT / "src" / "wq_agent" / "launcher.py"),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _copy_public_resources() -> None:
    for dirname in PUBLIC_RESOURCE_DIRS:
        source = PROJECT_ROOT / dirname
        destination = APP_DIST / dirname
        if source.exists():
            shutil.copytree(source, destination, dirs_exist_ok=True, ignore=_ignore_private_paths)

    for filename in PUBLIC_RESOURCE_FILES:
        source = PROJECT_ROOT / filename
        if source.exists():
            shutil.copy2(source, APP_DIST / filename)


def _ignore_private_paths(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        path = Path(directory) / name
        if name in FORBIDDEN_NAMES:
            ignored.add(name)
        elif path.suffix.lower() in FORBIDDEN_SUFFIXES:
            ignored.add(name)
        elif path.is_file() and path.suffix.lower() not in PUBLIC_RESOURCE_SUFFIXES:
            ignored.add(name)
        elif name in {"entries", "lessons"} and Path(directory).name == "wiki":
            ignored.add(name)
    return ignored


def _assert_required_sources() -> None:
    missing = [path for path in REQUIRED_SOURCE_PATHS if not (PROJECT_ROOT / path).exists()]
    if missing:
        formatted = "\n".join(f" - {path}" for path in missing)
        raise SystemExit(f"Missing required build resources:\n{formatted}")


def _assert_distribution_is_private_safe() -> None:
    missing = [path for path in REQUIRED_DIST_PATHS if not (APP_DIST / path).exists()]
    if missing:
        formatted = "\n".join(f" - {path}" for path in missing)
        raise SystemExit(f"Distribution is missing required resources:\n{formatted}")

    violations: list[Path] = []
    for resource_dir in PUBLIC_RESOURCE_DIRS:
        root = APP_DIST / resource_dir
        if root.exists():
            violations.extend(_find_forbidden_resource_paths(root, root.relative_to(APP_DIST)))
    violations.extend(path.relative_to(APP_DIST) for path in _find_private_runtime_paths(APP_DIST))

    if violations:
        formatted = "\n".join(f" - {path}" for path in sorted(violations))
        raise SystemExit(f"Distribution contains private/runtime files:\n{formatted}")


def _find_private_runtime_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    violations: list[Path] = []
    for path in root.iterdir():
        if path.name in FORBIDDEN_NAMES:
            violations.append(path)
        elif path.is_file() and path.suffix.lower() in RUNTIME_PRIVATE_SUFFIXES:
            violations.append(path)
    return sorted(violations)


def _find_forbidden_resource_paths(root: Path, rel_root: Path) -> list[Path]:
    violations: list[Path] = []
    for path in root.rglob("*"):
        rel = rel_root / path.relative_to(root)
        parts = set(rel.parts)
        if parts & FORBIDDEN_NAMES:
            violations.append(rel)
            continue
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(rel)
            continue
        if rel.parts[:2] in {("wiki", "entries"), ("wiki", "lessons")}:
            violations.append(rel)
    return violations


if __name__ == "__main__":
    main()
