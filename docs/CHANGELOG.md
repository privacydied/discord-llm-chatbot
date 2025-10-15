# Changelog

> **Fill me in**
> - [ ] Record changes for each release.

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Pure helper module for routing decisions with structured breadcrumbs.
- Documentation files: `AUDIT.md`, `REFACTOR.md`, `LOGGING.md`, `TESTPLAN.md`.

### Changed
- `contextual_brain_infer` delegates scope resolution to helper, improving observability without altering behavior.

### Fixed
- Reply anchoring & mention correctness
  - Replies now anchor to the human asker: reply parent in replies, newest human in threads; never self-anchor.
  - Exactly one notification per reply (reply-ping or explicit mention). Uses `AllowedMentions` and `mention_author=False` to avoid double-ping.
  - Locality-first context preserved; no cross-scope memory bleed. Routing/unroll/media gates unchanged.
  - Safe fallbacks: archived/missing parent or invalid thread state → plain send with no explicit mention; no crashes.

## [0.1.0] - 2024-01-01
### Added
- Initial release of the Discord LLM Chatbot
