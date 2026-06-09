# Open Source Process

This document describes the normal project flow for wq-agent.

## Public Repository Boundary

Public:

- `src/`, `tests/`, `templates/`, `docs/`
- `wiki/concepts/`, `wiki/operators/`, `wiki/fields/`, `wiki/patterns/`,
  `wiki/recipes/`, `wiki/papers/`, `wiki/datasets/`
- `.env.example`, `README.md`, and project governance files

Private:

- `.env`, credentials, tokens, session data
- `wq_agent.db`, generated SQLite files, and logs
- `private_wiki/`, `wiki/entries/`, `wiki/lessons/`
- Local agent/editor folders such as `.claude/` and `.codex/`

Before pushing to a public remote:

```bash
git ls-files | rg '^(private_wiki/|wiki/entries/|wiki/lessons/|\.claude/|\.codex/|\.env$)|\.(db|log)$'
```

The command should print nothing.

## Issue Triage

Use labels consistently:

- `bug`: broken or surprising behavior
- `enhancement`: new capability or workflow improvement
- `documentation`: docs-only work
- `good first issue`: small, well-scoped contribution
- `help wanted`: maintainers welcome outside help
- `security`: public tracking issue after private disclosure is resolved

Triage checklist:

1. Confirm the report is reproducible or ask for missing details.
2. Check that no private data was included.
3. Add labels and milestone if appropriate.
4. Link related issues or pull requests.
5. Close duplicates with a link to the canonical issue.

## Development Flow

1. Start from `main`.
2. Create a focused branch.
3. Keep commits reviewable.
4. Update tests and docs with behavior changes.
5. Run `ruff check src tests` and `pytest`.
6. Open a pull request using the template.

## Pull Request Review

Merge requirements:

- CI passes, or a maintainer documents why a failure is unrelated.
- Behavior changes include tests or a clear test rationale.
- Public docs are updated when commands, configuration, or workflows change.
- No private research data, credentials, databases, or logs are included.
- At least one maintainer approves non-trivial changes.

Preferred merge method: squash merge.

## Release Flow

1. Ensure `main` is green.
2. Update `CHANGELOG.md`.
3. Bump `version` in `pyproject.toml`.
4. Create and push a tag:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

5. Let the release workflow build artifacts.
6. Create GitHub release notes from the changelog.
7. Publish to package indexes only after artifacts and smoke tests pass.

Suggested smoke test:

```bash
python -m venv /tmp/wq-agent-smoke
source /tmp/wq-agent-smoke/bin/activate
pip install dist/wq_agent-*.whl
wq-agent --help
```

## Security Flow

1. Receive report privately.
2. Confirm impact and affected versions.
3. Patch on a private branch if public discussion would expose users.
4. Release the fix.
5. Publish a short advisory with impact, fixed version, and mitigation.

## Dependency Updates

Dependabot opens weekly dependency update pull requests. Treat runtime
dependencies more carefully than dev-only dependencies, especially integrations
that touch authentication, HTTP clients, SQLite, or vector indexing.
