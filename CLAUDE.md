# CLAUDE.md — RAG Knowledge Assistant Project Blueprint

## 1. Project Identity

This is a **RAG-based Knowledge Assistant** that ingests documents (PDF, plain text), chunks and embeds them into a Qdrant vector database, and answers user queries with cited source passages using OpenAI (gpt-4o-mini) or a local LLaMA 3.1 model via Ollama. The system is served as a REST API via FastAPI with endpoints for document ingestion, querying, document management, and health checks. There is no frontend — all interaction is via REST, with auto-generated OpenAPI docs at `/docs` serving as the developer-facing interface. Every endpoint except `/health` is authenticated via Bearer API keys (hashed with bcrypt, stored in config/env). LangChain orchestrates the entire RAG pipeline: document loading, text splitting, embedding generation, vector store operations, retrieval, prompt assembly, and LLM invocation.

## 2. Tech Stack

| Concern | Choice | Version | Why |
|---|---|---|---|
| Language | Python | 3.12 | Latest stable; required for modern asyncio patterns, `TaskGroup`, `ExceptionGroup` |
| API Framework | FastAPI | 0.115 | Async-native, automatic OpenAPI docs, dependency injection, lifespan events |
| RAG Orchestration | LangChain | 0.3.x | Chain composition (LCEL), document loaders, text splitters, retrievers, prompt templates |
| OpenAI Integration | langchain-openai | 0.2.x | `ChatOpenAI` and `OpenAIEmbeddings` wrappers with async support |
| Local LLM | langchain-ollama | 0.2.x | `ChatOllama` wrapper for local LLaMA 3.1 via Ollama — no API cost, full privacy |
| Vector Store | langchain-qdrant | 0.2.x | LangChain-native Qdrant integration with retriever interface |
| Document Loaders | langchain-community | 0.3.x | `PyPDFLoader`, `TextLoader` for document ingestion |
| Text Splitting | langchain-text-splitters | 0.3.x | `RecursiveCharacterTextSplitter` with configurable chunk size and overlap |
| Vector Database | Qdrant | 1.12 | Purpose-built vector DB with HNSW indexing, payload filtering, gRPC support |
| Qdrant SDK | qdrant-client | 1.12 | Python client for collection management, batch upserts, search operations |
| Default LLM | OpenAI gpt-4o-mini | — | Fast, cost-effective, 128k context window, strong instruction following |
| Default Embeddings | OpenAI text-embedding-3-small | — | 1536 dimensions, fast, cheap ($0.02/1M tokens), strong retrieval quality |
| Local LLM Alternative | Ollama + LLaMA 3.1 8B | — | Free, private, no API key required, runs on consumer hardware |
| Testing | pytest + pytest-asyncio + httpx | 8.x / 0.24 / 0.27 | Async test support, `TestClient` via httpx for FastAPI integration tests |
| Linting & Formatting | ruff | 0.8 | Single tool replaces flake8 + black + isort; fast, configurable, Rust-based |
| Package Manager | uv | 0.5 | Fast dependency resolution, replaces pip + venv, lockfile support |
| Logging | structlog | 24.x | Structured JSON logging with context variables bound per-request |
| Metrics | prometheus-client | 0.21 | Histogram for latency, counters for status codes, exposed at `/metrics` |
| Tracing | opentelemetry-sdk + instrumentation-fastapi | 1.x | Distributed tracing with spans for embedding, search, and LLM calls |
| CI | GitHub Actions | — | Lint, test, build on every push and PR |
| Deployment | Docker + docker-compose | — | Multi-stage build; app container + Qdrant container + optional Ollama container |

## 3. Project Structure

```
rag-knowledge-assistant/
├── src/
│   ├── api/              # FastAPI routes: /ingest, /query, /documents, /health
│   ├── ingestion/        # Document loading (PDF, text), chunking, embedding pipeline
│   ├── retrieval/        # Vector search via Qdrant, reranking, context assembly
│   ├── generation/       # LangChain chains, prompt templates, LLM provider switching
│   ├── storage/          # Qdrant client wrapper, collection management, metadata
│   ├── common/           # Config, typed errors, logging setup, auth dependencies
│   └── main.py           # FastAPI app entry point, middleware, lifespan
├── tests/
│   ├── unit/             # Pure logic tests — no Qdrant, no network
│   ├── integration/      # Tests against real Qdrant instance
│   └── fixtures/         # Sample PDFs, text files for testing
├── config/               # Environment-specific configs
├── scripts/              # setup, seed, migrate scripts
├── docs/                 # Architecture docs, ADRs
├── .claude/
│   ├── agents/           # Agent definitions
│   ├── commands/         # Custom slash commands
│   ├── context/          # Contextual knowledge files
│   ├── hooks/            # Pre/post tool-use hooks
│   ├── skills/           # Skill definitions
│   ├── settings.json     # Hook and permission configuration
│   └── PERSONA.md        # Engineer persona definition
├── .github/workflows/    # CI pipeline
├── pyproject.toml        # Project manifest (uv/pip compatible)
├── ruff.toml             # Linter + formatter config
├── docker-compose.yml    # App + Qdrant + optional Ollama
├── Dockerfile            # Multi-stage Python build
├── .env.example          # Template with all required env vars
├── .gitignore
├── .claudeignore
├── CLAUDE.md
└── CHANGELOG.md
```

## 4. Commands

```bash
# Setup
uv init
uv add fastapi uvicorn[standard] langchain langchain-openai langchain-ollama langchain-qdrant langchain-community langchain-text-splitters qdrant-client openai python-multipart pypdf structlog prometheus-client opentelemetry-sdk opentelemetry-instrumentation-fastapi opentelemetry-exporter-otlp bcrypt python-dotenv
uv add --dev pytest pytest-asyncio httpx ruff pytest-cov

# Development
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Testing
uv run pytest
uv run pytest tests/unit/test_ingestion.py -v
uv run pytest --tb=short -q -x
uv run pytest --cov=src --cov-report=html

# Linting
uv run ruff check .
uv run ruff check . --fix
uv run ruff format .

# Docker
docker-compose up -d qdrant
docker-compose up --build

# Clean
rm -rf .venv __pycache__ .pytest_cache .ruff_cache htmlcov .coverage
```

## 5. Architecture Decisions

All ADRs are documented in `docs/decisions.md`. Summary of key decisions:

- **Qdrant over pgvector** — Purpose-built vector DB with superior HNSW performance, payload filtering, and no need for PostgreSQL ops overhead.
- **LangChain for orchestration** — LCEL chain composition, built-in document loaders, text splitters, and retriever abstractions reduce boilerplate significantly.
- **OpenAI + Ollama dual-provider** — OpenAI for production quality, Ollama for local development and cost-free experimentation; both abstracted behind LangChain's ChatModel interface.
- **text-embedding-3-small over local sentence-transformers** — Simpler deployment (no GPU, no model download), 1536d vectors, strong retrieval quality at $0.02/1M tokens.
- **RecursiveCharacterTextSplitter** — Respects paragraph/sentence boundaries, configurable chunk size (default 1024) and overlap (default 128), best general-purpose splitter.
- **FastAPI lifespan for resource management** — Qdrant client and embedding model initialized once at startup, cleaned up at shutdown; no per-request connection overhead.
- **Bearer token auth with bcrypt** — Simple, stateless, sufficient for API-to-API integration; keys hashed with bcrypt and stored in environment/config.
- **structlog over stdlib logging** — Structured JSON output, context variable binding per request, zero-config correlation with OpenTelemetry trace IDs.

## 6. Multi-Agent System

| Agent | File | Purpose | Commands | Skills |
|---|---|---|---|---|
| Orchestrator | (default) | Coordinates all tasks, reads CLAUDE.md first | `/plan`, `/status` | Code generation, architecture |
| Code Agent | `.claude/agents/code.md` | Writes source code in `src/` | `/implement`, `/refactor` | Python, FastAPI, LangChain patterns |
| Test Agent | `.claude/agents/test.md` | Writes and runs tests in `tests/` | `/test`, `/coverage` | pytest, fixtures, mocking LLM |
| Review Agent | `.claude/agents/review.md` | Reviews code for correctness and style | `/review`, `/lint` | ruff, type checking, security audit |
| DevOps Agent | `.claude/agents/devops.md` | Docker, CI, deployment configs | `/deploy`, `/ci` | Docker, GitHub Actions, Qdrant ops |
| Docs Agent | `.claude/agents/docs.md` | Writes docstrings, ADRs, API docs | `/docs`, `/adr` | OpenAPI, markdown, changelogs |

**Hooks:**
- `PreToolUse` (Bash, on git commit/push) — runs `.claude/hooks/pre-commit.sh` for lint checks
- `PostToolUse` (Bash) — runs `.claude/hooks/on-bash-complete.sh` for error detection
- `Notification` — runs `.claude/hooks/on-notification.sh` for alerting

## 7. Engineering Rules

### Always Do

- Type-annotate every function signature including return types
- Use `Annotated[T, Depends(...)]` for all FastAPI dependency injection
- Use LangChain's LCEL (`RunnablePassthrough`, `|` pipe operator) for chain composition
- Use `RecursiveCharacterTextSplitter` with explicit `chunk_size` and `chunk_overlap` parameters
- Use Pydantic v2 models for all request/response schemas with `model_config = ConfigDict(strict=True)`
- Use `structlog.contextvars.bind_contextvars()` in middleware for per-request context
- Use `asyncio.to_thread()` for CPU-bound work (PDF parsing, embedding generation)
- Use `pathlib.Path` for all file system operations
- Use `httpx.AsyncClient` for outbound HTTP calls
- Batch embed documents (never embed one chunk at a time)
- Set a similarity score threshold (minimum 0.5 cosine) in addition to `top_k` limit
- Write docstrings for every public function: one-line summary, Args, Returns, Raises
- Keep functions under 40 lines; extract helpers when they grow
- Run `uv run ruff check . && uv run pytest` before every commit
- Use Qdrant payload filtering for metadata-scoped searches (by document ID, file type, date)

### Never Do

- Never use bare `except` — always catch the narrowest exception type
- Never log API keys, user queries containing PII, or raw LLM prompts at INFO level
- Never instantiate services inside route handlers — use dependency injection
- Never call synchronous blocking code inside `async def` without `asyncio.to_thread()`
- Never return raw dicts from route handlers — use Pydantic response models
- Never use `os.path` — use `pathlib.Path`
- Never use `requests` — use `httpx.AsyncClient`
- Never hardcode model names, chunk sizes, or Qdrant collection names — use config
- Never store vectors without metadata (source document ID, chunk index, file name)
- Never commit `.env` files — only `.env.example` with placeholder values
- Never skip the retrieval step and pass full documents to the LLM
- Never use LangChain's deprecated APIs (`LLMChain`, `RetrievalQA`) — use LCEL
- Never create Qdrant collections without specifying distance metric and vector size explicitly
- Never embed user queries with a different model than was used for document embeddings

## 8. Getting Started

After reading this CLAUDE.md, your first task is:

```
1. Run `uv init` and add all dependencies from Section 4
2. Create the directory structure from Section 3
3. Create ruff.toml with strict settings (select = ["E", "F", "W", "I", "N", "UP", "S", "B", "A", "C4", "SIM", "TCH"])
4. Create pyproject.toml [tool.pytest.ini_options] with asyncio_mode = "auto"
5. Create .env.example with: OPENAI_API_KEY, QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDING_MODEL, LLM_MODEL, LLM_PROVIDER, OLLAMA_BASE_URL, API_KEY, LOG_LEVEL
6. Create .gitignore for Python
7. Create docker-compose.yml with qdrant and app services
8. Create Dockerfile (multi-stage, uv-based)
9. Write src/main.py with FastAPI app skeleton + health endpoint
10. Write one passing test in tests/unit/test_health.py
11. Create .github/workflows/ci.yml
12. Make initial commit
```
