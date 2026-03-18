# Agent Workflows

This document defines how to route user requests to the appropriate workflow, the five standard workflow patterns used in the RAG Knowledge Assistant, and the handoff protocol that agents follow when passing work between stages.

---

## Decision Table

When a user request arrives, use this table to determine the correct workflow. Match the user's intent to the closest row and follow the corresponding routing.

| User wants... | Route to workflow | Agents involved |
|---|---|---|
| Design a new feature | Feature workflow | architect, implement, test-writer, reviewer |
| Implement a designed feature | Feature workflow (skip architect) | implement, test-writer, reviewer |
| Fix a bug | Bug fix workflow | debugger, fix, test-writer, reviewer |
| Investigate a bug (no fix yet) | Bug fix workflow (stop after debugger) | debugger |
| Add tests to existing code | Test-only workflow | test-writer |
| Review code or a pull request | Review-only workflow | reviewer |
| Improve performance | Performance workflow | performance-analyst, refactorer, reviewer |
| Audit security | Security workflow | security-analyst, fix, reviewer |
| Refactor code | Refactor workflow | refactorer, reviewer |
| Migrate API, schema, or dependency | Migration workflow | migrator, test-writer, reviewer |
| Complex multi-step task | Coordinated workflow | coordinator delegates to appropriate agents |
| Explain code or concepts | Direct response | No workflow needed; use explain skill directly |
| Update changelog | Direct response | No workflow needed; use changelog skill directly |
| Add observability | Direct response | No workflow needed; use add-observability skill directly |

### Ambiguous Requests

When the user's intent does not clearly match one row, ask for clarification. Common ambiguities:

- "Fix the performance" could be a bug fix (something is broken) or performance improvement (something is slow but correct). Ask: "Is the feature producing wrong results, or is it producing correct results too slowly?"
- "Update the API" could be a new feature (add endpoint), a migration (change existing endpoint), or a refactor (restructure without behavior change). Ask: "Are you adding new behavior, changing existing behavior, or restructuring without changing behavior?"
- "Clean up the code" could be a refactor (improve structure) or a style fix (formatting). Ask: "Are you looking to restructure the code or just clean up formatting and style?"

---

## Workflow Patterns

### 1. Feature Workflow

**Trigger:** User wants to design and implement a new capability.

**Stages:**

1. **Architect** — Define the feature's scope, identify which modules are affected, design the interfaces (function signatures, Pydantic models, API endpoints), and decide where new code lives in the file structure. Output: a design document with module layout, key interfaces, and data flow.

2. **Implement** — Write the production code following the architect's design. Follow the conventions in `conventions.md`. Use the patterns from the stack (LangChain LCEL chains, FastAPI dependency injection, Pydantic validation). Ensure all functions have type hints and docstrings.

3. **Test-writer** — Write tests for the implemented code following the patterns in the `write-test` skill. Cover the happy path, edge cases, and error conditions. Use `@pytest.mark.asyncio` for async code, `QdrantClient(":memory:")` for vector store tests, `httpx.AsyncClient` for API endpoint tests.

4. **Reviewer** — Review the implementation and tests using the `code-review` skill's 10-point checklist. Report findings categorized by severity. Verify that `uv run ruff check .` and `uv run pytest` pass.

**Example:** "Add a /query/stream endpoint that streams LLM responses."
- Architect: Design the streaming endpoint, define the response format, identify which chain methods to use (`astream` vs `ainvoke`).
- Implement: Write the route handler with `StreamingResponse`, update the chain to use `astream`, add the endpoint to the router.
- Test-writer: Write tests for the streaming endpoint, including tests for partial responses and error during streaming.
- Reviewer: Review for async correctness, error handling, typing, and test coverage.

---

### 2. Bug Fix Workflow

**Trigger:** User reports something is broken or producing wrong results.

**Stages:**

1. **Debugger** — Reproduce the issue. Read the relevant code. Identify the root cause by tracing the data flow from input to output. Check logs and error messages. Output: a diagnosis explaining what is wrong, why, and where in the code the bug lives (file path and line number).

2. **Fix** — Write the minimal code change that fixes the root cause. Do not refactor unrelated code. Do not add features. Fix only the identified bug. If the fix requires changing an interface, note the downstream impact.

3. **Test-writer** — Write a regression test that fails before the fix and passes after. This test ensures the bug does not recur. Add it to the appropriate test file.

4. **Reviewer** — Review the fix for correctness, side effects, and test coverage. Verify the regression test actually tests the bug scenario.

**Example:** "The /query endpoint returns 500 when the Qdrant collection doesn't exist."
- Debugger: Trace the query endpoint code. Find that `retriever.ainvoke()` raises `UnexpectedResponse` without a handler.
- Fix: Add a try/except for `UnexpectedResponse` that returns a 404 with a clear error message.
- Test-writer: Write a test that queries when no collection exists and asserts 404 response.
- Reviewer: Check that the fix handles only the specific error, not all exceptions.

---

### 3. Refactor Workflow

**Trigger:** User wants to improve code structure without changing behavior.

**Stages:**

1. **Refactorer** — Identify the refactoring type (extract, inline, rename, restructure, generalize) using the `refactor` skill. Plan the steps. Read all callers of affected code. Execute the refactoring in atomic steps, running `uv run pytest` after each step.

2. **Reviewer** — Verify that behavior is unchanged. Run `uv run pytest` and `uv run ruff check .`. Check that all imports are updated. Verify no tests needed modification (if tests changed, behavior may have changed).

**Example:** "Extract the prompt templates into a separate module."
- Refactorer: Identify all prompt templates in `generation/chain.py`. Create `generation/prompts.py`. Move templates. Update imports in `chain.py` and any tests. Run tests after each move.
- Reviewer: Verify all imports updated, no behavior change, tests pass without modification.

---

### 4. Security Workflow

**Trigger:** User wants a security audit or reports a security concern.

**Stages:**

1. **Security-analyst** — Run the full `security-check` skill checklist against the codebase. Check prompt injection, SSRF, API key exposure, dependency vulnerabilities, input validation, .env safety, and rate limiting. Output: a prioritized list of findings with severity ratings.

2. **Fix** — Address findings in order of severity (Critical first). Each fix is a minimal, targeted change.

3. **Reviewer** — Review the fixes for correctness. Re-run the security checks to verify findings are resolved. Check that fixes do not introduce new issues.

**Example:** "Audit the application for security issues."
- Security-analyst: Run all 8 checks from the security-check skill. Find: no rate limiting on /query, API key in docker-compose.yml, no file size validation on /ingest.
- Fix: Add slowapi rate limiting. Move API key to .env. Add file size validation.
- Reviewer: Re-run security checks. Verify all three findings are resolved.

---

### 5. Migration Workflow

**Trigger:** User wants to change a dependency, API schema, or data format.

**Stages:**

1. **Migrator** — Plan the migration. Identify all affected files. Design the migration path (backward-compatible if possible). If migrating a database schema (Qdrant collection), plan the data migration. Output: a step-by-step migration plan.

2. **Test-writer** — Write tests for the new behavior before implementing the migration. Tests should verify the migration target state.

3. **Reviewer** — Review the migration plan and implementation. Verify backward compatibility (if required). Check that rollback is possible. Run all tests.

**Example:** "Migrate from OpenAI embeddings to Ollama embeddings for local development."
- Migrator: Plan the provider switch. Identify all files that reference OpenAI embeddings. Design the provider abstraction (if not already present). Plan collection recreation (different embedding dimension).
- Test-writer: Write tests that verify the Ollama provider returns embeddings and integrates with the retrieval chain.
- Reviewer: Verify the provider abstraction works for both OpenAI and Ollama. Check that switching is configuration-only.

---

## Handoff Protocol

When one stage completes and hands off to the next, the completing agent produces a structured summary that the next agent reads before starting work.

### Summary Format

```
## Handoff: [Source Stage] -> [Target Stage]

### What was done
- [Bullet list of completed actions]

### Files changed
- [List of file paths that were created, modified, or deleted]

### Key decisions
- [Any decisions made that affect the next stage]

### Open questions
- [Anything the next agent needs to decide or investigate]

### Status
- Tests passing: [yes/no]
- Lint passing: [yes/no]
- Ready for next stage: [yes/no]
```

### Rules

1. **Every handoff includes test and lint status.** If tests are failing, the handoff must explain why and whether the next stage is expected to fix them.
2. **File paths are absolute.** The next agent should be able to read any referenced file immediately.
3. **Decisions are recorded, not just actions.** If the architect chose FastAPI dependency injection over middleware for a feature, the handoff explains why so the implementer does not question or reverse the decision.
4. **Open questions are explicit.** If the architect was unsure whether to use `astream` or `ainvoke`, the handoff says so and the implementer makes the decision.
5. **No implicit context.** The next agent may not have seen the user's original request. The handoff must contain enough context to work independently.

### Coordinator Role

For complex multi-step tasks that span multiple workflows or have dependencies between steps, a coordinator manages the overall process:

- Breaks the task into subtasks and assigns each to a workflow.
- Tracks completion status of each subtask.
- Handles dependencies (e.g., feature B depends on feature A being completed first).
- Collects handoff summaries from each stage and routes them to the next.
- Reports overall progress to the user.

The coordinator does not write code. It orchestrates the agents that do.
