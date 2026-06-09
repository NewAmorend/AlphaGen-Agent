# Security Policy

## Supported Versions

Security fixes are provided for the `main` branch until the project starts
publishing versioned maintenance lines.

## Reporting a Vulnerability

Please do not open a public issue for a suspected vulnerability. Report it
privately through GitHub Security Advisories for this repository, or contact a
maintainer privately if advisories are unavailable.

Include:

- Affected version or commit.
- Steps to reproduce.
- Impact and likely exposure.
- Any relevant logs with secrets removed.

## Sensitive Data

This project is designed for public code plus private research. Never include:

- `.env`, API keys, credentials, or session tokens.
- `wq_agent.db` or other local SQLite databases.
- `*.log` files that may contain prompts, responses, alpha expressions, account
  identifiers, or API responses.
- `private_wiki/`, `wiki/entries/`, or `wiki/lessons/` content unless you have
  explicitly sanitized and approved it for publication.

If sensitive data is committed, rotate the exposed secret first, then remove it
from the repository history before public release.
