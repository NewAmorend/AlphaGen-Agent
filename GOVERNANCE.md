# Governance

wq-agent currently uses a maintainer-led governance model.

## Roles

- **Users** report issues, suggest improvements, and share reproducible use
  cases.
- **Contributors** submit documentation, tests, fixes, and features through pull
  requests.
- **Maintainers** triage issues, review pull requests, make release decisions,
  and protect the public/private data boundary.

## Decision Making

Most decisions are made by lazy consensus: if a proposal has clear support and
no unresolved objections after reasonable review time, a maintainer may merge
it.

Maintainers may make immediate decisions for security fixes, CI breakages,
documentation corrections, and small low-risk maintenance work.

Large changes should start as an issue or design note before implementation.
Examples include storage schema changes, CLI command redesigns, new external
service integrations, and changes to the wiki retrieval architecture.

## Maintainer Responsibilities

- Keep `main` passing CI.
- Require tests or a documented test rationale for behavior changes.
- Keep private data out of public history.
- Label and triage incoming issues.
- Publish release notes for tagged releases.
- Handle security and conduct reports privately and promptly.
