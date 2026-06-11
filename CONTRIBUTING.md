# Contributing

Thanks for helping improve wq-agent. This project is a Python CLI/TUI harness
for generating, backtesting, evaluating, and curating WorldQuant-style alpha
research workflows.

## Ground Rules

- Keep public contributions free of private credentials, API keys, account
  data, backtest databases, logs, and private research notes.
- Prefer small, focused pull requests with tests or a clear reason why tests do
  not apply.
- Preserve the public/private boundary described in the README. Files under
  `private_wiki/`, `wiki/entries/`, and `wiki/lessons/` are not for public PRs.
- Do not include proprietary WorldQuant content unless you have the right to
  publish it.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Fill `.env` only with local development credentials. Never commit `.env`.

## Local Checks

Run these before opening a pull request:

```bash
ruff check src tests
pytest
```

For changes that touch packaging metadata, also run:

```bash
python -m build
```

## Pull Request Flow

1. Open or link an issue for user-visible behavior changes.
2. Create a branch from `main`.
3. Make the smallest coherent change.
4. Add or update tests and docs.
5. Run the local checks.
6. Open a pull request with context, test results, and any follow-up work.

Maintainers should use squash merges for normal feature/fix PRs and keep the
merge commit history reserved for coordinated release or large integration
work.

## Commit Style

Use short, imperative commit subjects. Conventional prefixes are welcome:

- `feat:` user-visible capability
- `fix:` bug fix
- `docs:` documentation-only change
- `test:` test-only change
- `refactor:` behavior-preserving code change
- `ci:` automation and workflow changes

## Review Expectations

Reviewers focus on correctness, reproducibility, test coverage, public/private
data separation, and whether the CLI/TUI behavior remains understandable.

PRs should not be merged until CI passes or a maintainer documents why the
failing check is unrelated.
