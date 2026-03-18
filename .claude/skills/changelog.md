---
name: changelog
description: Generate or update a CHANGELOG.md following Keep a Changelog format with conventional commit parsing
---

# Changelog Skill

Generate and maintain a CHANGELOG.md following the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. Parse conventional commits from git history to categorize changes.

---

## Format

The changelog uses these exact categories in this order:

```markdown
# Changelog

All notable changes to the RAG Knowledge Assistant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New features that were added

### Changed
- Changes to existing features

### Deprecated
- Features that will be removed in future versions

### Removed
- Features that were removed

### Fixed
- Bug fixes

### Security
- Security-related changes

## [0.2.0] - 2026-03-15

### Added
- Streaming response support for /query endpoint
- Ollama provider for local LLM inference

### Fixed
- Chunk overlap not applied when using RecursiveCharacterTextSplitter

## [0.1.0] - 2026-03-01

### Added
- Initial RAG pipeline with FastAPI, LangChain, and Qdrant
- PDF and text document ingestion
- OpenAI embeddings and GPT-4o-mini generation
- Health check endpoint
- Docker Compose setup for Qdrant
```

---

## Conventional Commit Parsing

Map conventional commit prefixes to changelog categories:

| Commit prefix | Changelog category |
|---|---|
| `feat:` | Added |
| `fix:` | Fixed |
| `refactor:` | Changed |
| `perf:` | Changed |
| `docs:` | (skip — documentation-only changes) |
| `test:` | (skip — test-only changes) |
| `chore:` | (skip — maintenance tasks) |
| `ci:` | (skip — CI configuration) |
| `style:` | (skip — formatting only) |
| `build:` | Changed |
| `revert:` | Removed |
| `security:` or `sec:` | Security |
| `deprecate:` | Deprecated |
| `BREAKING CHANGE:` in body | Changed (note as breaking) |

---

## Workflow

When asked to update the changelog:

### Step 1: Find the last release tag

```bash
# Get the most recent tag
git describe --tags --abbrev=0

# If no tags exist, use the initial commit
git rev-list --max-parents=0 HEAD
```

### Step 2: Read git log since last tag

```bash
# Commits since last tag, one line each
git log v0.2.0..HEAD --oneline --no-merges

# With full messages for parsing conventional commits
git log v0.2.0..HEAD --format="%H %s%n%b" --no-merges
```

### Step 3: Categorize each commit

Parse the commit subject line for the conventional commit prefix. Map to the appropriate changelog category using the table above. Skip commits that map to documentation, tests, chores, CI, or style categories.

### Step 4: Write the entry

- Add entries under `## [Unreleased]` section.
- Each entry is a single bullet point starting with `- `.
- Write in imperative present tense: "Add streaming support" not "Added streaming support" or "Adds streaming support".
- Include the scope if present: `feat(ingestion): Add batch processing` becomes `- Add batch processing for document ingestion`.
- Group related commits into a single entry when they address the same feature or fix.
- Do not include commit hashes in the changelog (they add noise).

### Step 5: Verify

- Read the updated CHANGELOG.md to verify formatting.
- Ensure no duplicate entries.
- Ensure categories are in the correct order (Added, Changed, Deprecated, Removed, Fixed, Security).
- Remove empty categories (do not leave `### Added` with no entries).

---

## Release Workflow

When asked to cut a release:

1. Move all entries from `## [Unreleased]` to a new version section: `## [X.Y.Z] - YYYY-MM-DD`.
2. Determine version bump based on changes:
   - **Major (X):** Breaking changes present (BREAKING CHANGE in commit body).
   - **Minor (Y):** New features added (feat: commits).
   - **Patch (Z):** Only bug fixes (fix: commits) and non-breaking changes.
3. Add an empty `## [Unreleased]` section at the top.
4. Update version links at the bottom of the file:
   ```markdown
   [unreleased]: https://github.com/owner/repo/compare/v0.3.0...HEAD
   [0.3.0]: https://github.com/owner/repo/compare/v0.2.0...v0.3.0
   [0.2.0]: https://github.com/owner/repo/compare/v0.1.0...v0.2.0
   [0.1.0]: https://github.com/owner/repo/releases/tag/v0.1.0
   ```

---

## Edge Cases

- **No conventional commit prefix:** Categorize based on the change description. If unclear, put under Changed.
- **Multiple categories in one commit:** If a commit adds a feature AND fixes a bug, create entries in both categories.
- **Merge commits:** Skip merge commits (`--no-merges` flag in git log).
- **Squash merges from PRs:** Parse the PR title as the commit message.
- **No changes since last tag:** Report "No changes to document since last release."
- **First changelog creation:** Include all commits from the beginning of the repository.
