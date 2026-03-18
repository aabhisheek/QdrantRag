# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A **RAG-based Knowledge Assistant** REST API. Users upload documents (PDF, `.txt`, `.md`), which are chunked, embedded locally via FastEmbed, and stored in Qdrant. Query requests retrieve the top-k chunks by cosine similarity, then pass them as context to an LLM (Groq by default, OpenAI or Ollama as alternatives). Every endpoint except `GET /health` requires a Bearer token.

## Commands

```bash
# Run dev server
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/unit/test_health.py -v

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Lint + format
uv run ruff check .
uv run ruff check . --fix
uv run ruff format .

# Start Qdrant (required before running the app)
docker-compose up -d qdrant

# Full stack (app + Qdrant)
docker-compose up --build
```

## Architecture

**Request flow:**
1. `POST /ingest` → `IngestionPipeline.ingest_file()` → loads with `PyPDFLoader`/`TextLoader` → splits with `RecursiveCharacterTextSplitter` → adds metadata (`document_id`, `filename`, `file_type`, `chunk_index`) → `QdrantStore.add_documents()`
2. `POST /query` → `RetrieverService.retrieve()` (cosine similarity, score ≥ 0.5) → `RetrieverService.format_context()` → `GenerationChain.generate()` → LLM via LCEL chain

**Service wiring:** All services (`QdrantStore`, `IngestionPipeline`, `RetrieverService`, `GenerationChain`) are instantiated once in the FastAPI lifespan (`src/main.py`) and stored in `app.state`. Route handlers access them via `request.app.state.*` — never instantiate services inside handlers.

**LLM providers** (set via `LLM_PROVIDER` env var):
- `groq` (default) — `ChatGroq`, requires `GROQ_API_KEY`
- `openai` — `ChatOpenAI`, requires `OPENAI_API_KEY`
- `ollama` — `ChatOllama`, requires running Ollama at `OLLAMA_BASE_URL`

**Embeddings:** FastEmbed (`BAAI/bge-small-en-v1.5`, 384 dimensions) — runs locally, no API key. The collection is auto-recreated if `EMBEDDING_DIMENSIONS` changes (dimension mismatch detection in `QdrantStore.initialize()`).

**Auth:** `src/common/auth.py` — Bearer token checked via `verify_token` dependency. If `API_KEY` starts with `$2b$`, bcrypt comparison is used; otherwise plain string comparison. If `API_KEY` is empty, all tokens pass.

## Key Patterns

- **Config:** `src/common/config.py` — `Settings(BaseSettings)` loaded from `.env`, accessed via `get_settings()` (LRU-cached). Add new env vars here first.
- **Errors:** `src/common/errors.py` — `AppError` subclasses with `status_code`; handled globally in `main.py` via `@app.exception_handler(AppError)`.
- **Logging:** structlog throughout — `structlog.get_logger()` per module, `logger.info("event_name", key=value)`.
- **LCEL chain:** `GenerationChain` builds the chain in `__init__` as `{context, question} | prompt | llm | StrOutputParser()`. Called with `chain.ainvoke(...)`.
- **Tests:** Unit tests mock at the import level — `patch("src.storage.qdrant_store.QdrantClient")`, etc. See `test_health.py` for the `TestClient` + lifespan pattern.

## Tech Stack (Actual Versions)

| Component | Implementation |
|---|---|
| LLM (default) | Groq `llama-3.3-70b-versatile` via `langchain-groq` |
| Embeddings | FastEmbed `BAAI/bge-small-en-v1.5` (384d, local) |
| Vector DB | Qdrant via `langchain-qdrant` + `qdrant-client` |
| API | FastAPI + uvicorn |
| RAG orchestration | LangChain LCEL (`langchain>=1.2`) |
| Testing | pytest + pytest-asyncio (`asyncio_mode = "auto"`) + httpx |

## Engineering Rules

- Use `Annotated[T, Depends(...)]` for FastAPI DI
- Use LCEL (`|` pipe) for all LangChain chains — never `LLMChain` or `RetrievalQA`
- Use `asyncio.to_thread()` for synchronous blocking calls inside `async def`
- Use `pathlib.Path` for file paths, never `os.path`
- Use Pydantic v2 response models — never return raw dicts from route handlers
- Never hardcode model names, collection names, or chunk sizes — always use `settings.*`
- Always store chunks with metadata: `document_id`, `filename`, `file_type`, `chunk_index`
- Catch specific exceptions, never bare `except:`
