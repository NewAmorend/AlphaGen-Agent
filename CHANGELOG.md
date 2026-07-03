# Changelog

This project follows a simple changelog format inspired by Keep a Changelog and
uses semantic versioning once public releases begin.

## [Unreleased]

### Changed

- Renamed the project, distribution, package, and primary CLI to AlphaGen Agent,
  `alphagen_agent`, and `alphagen-agent`; legacy Python and CLI names remain compatible.
- Reuse an existing `wq_agent.db` automatically when `alphagen_agent.db` is absent,
  preventing upgrades from silently starting with an empty database.

### Added

- Open source project process documents, GitHub issue/PR templates, CI, and
  dependency update configuration.

## [0.1.0] - 2026-06-09

### Added

- Initial public package metadata for the AlphaGen Agent CLI/TUI alpha research
  harness.
