from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


PRIVATE_PATH_RE = re.compile(
    r"(^|/)(private_wiki|wiki/entries|wiki/lessons|\.claude|\.codex)(/|$)"
    r"|(^|/)\.env$"
    r"|\.(db|log)$"
)
LOCAL_RESEARCH_TRACE_RE = re.compile(
    r"(已提交|submitted|SUBMITTED|submission|提交)\s*[#＃]\d+"
)
SECRET_LIKE_RE = re.compile(
    r"AKIA[0-9A-Z]{16}"
    r"|sk-[A-Za-z0-9_-]{20,}"
    r"|(?:KIMI|DEEPSEEK|EMBEDDING|WQ)_(?:API_KEY|PASSWORD)\s*=\s*"
    r"(?!\s*(?:$|#|your_|<|\"))\S+"
)
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ALLOWED_PUBLIC_WIKI_FILES = {"wiki/.gitkeep"}

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "SUPPORT.md",
    "GOVERNANCE.md",
    "CHANGELOG.md",
    "pyproject.toml",
    ".env.example",
    ".github/workflows/ci.yml",
    ".github/workflows/release.yml",
]


def run_git(root: Path, *args: str) -> list[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return [line for line in proc.stdout.splitlines() if line]


def repo_root() -> Path:
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "not inside a git repository")
    return Path(proc.stdout.strip())


def is_text_path(path: str) -> bool:
    return Path(path).suffix.lower() in TEXT_SUFFIXES


def check_required(root: Path) -> list[str]:
    return [f"missing required file: {path}" for path in REQUIRED_FILES if not (root / path).exists()]


def check_tracked_private(tracked: list[str]) -> list[str]:
    failures = [f"private/generated file is tracked: {path}" for path in tracked if PRIVATE_PATH_RE.search(path)]
    failures.extend(
        f"public repo wiki must stay empty; tracked wiki content: {path}"
        for path in tracked
        if path.startswith("wiki/") and path not in ALLOWED_PUBLIC_WIKI_FILES
    )
    return failures


def check_untracked(root: Path, allow_untracked: bool) -> list[str]:
    if allow_untracked:
        return []
    status = run_git(root, "status", "--porcelain", "--untracked-files=all")
    untracked = [line[3:] for line in status if line.startswith("?? ")]
    return [f"untracked file before release: {path}" for path in untracked]


def check_text_content(root: Path, tracked: list[str]) -> list[str]:
    findings: list[str] = []
    for rel in tracked:
        if not is_text_path(rel):
            continue
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            findings.append(f"cannot read {rel}: {exc}")
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if SECRET_LIKE_RE.search(line):
                findings.append(f"secret-like value in {rel}:{lineno}")
            if rel.startswith("wiki/") and LOCAL_RESEARCH_TRACE_RE.search(line):
                findings.append(f"local research trace in {rel}:{lineno}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Run public-release hygiene checks.")
    parser.add_argument(
        "--allow-untracked",
        action="store_true",
        help="Do not fail on untracked files. Use only during local development, not releases.",
    )
    args = parser.parse_args()

    root = repo_root()
    tracked = run_git(root, "ls-files")
    failures: list[str] = []
    failures.extend(check_required(root))
    failures.extend(check_tracked_private(tracked))
    failures.extend(check_untracked(root, args.allow_untracked))
    failures.extend(check_text_content(root, tracked))

    if failures:
        print("Open-source preflight failed:", file=sys.stderr)
        for item in failures:
            print(f"- {item}", file=sys.stderr)
        return 1

    print("Open-source preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
