# Architecture Decision Records

This document contains the Architecture Decision Records (ADRs) for the RAG Knowledge Assistant. Each ADR captures a significant technical decision, the context that led to it, the alternatives considered, and the consequences accepted. New decisions are added at the bottom. Superseded decisions have their status updated but are never deleted — the history of why decisions were made is as valuable as the decisions themselves.

---

## ADR Template

Use this template when recording a new architecture decision:

```
## ADR-NNN: [Title]

**Status:** Proposed | Accepted | Deprecated | Superseded by ADR-NNN
**Date:** YYYY-MM-DD
**Context:** [Why this decision was needed. What problem or question triggered it.]
**Decision:** [What was decided. Be specific about the choice made.]
**Alternatives considered:** [What other options were evaluated and why they were rejected.]
**Consequences:** [What trade-offs were accepted. Both positive and negative impacts.]
```

---

## ADR-001: Python 3.12 as Primary Language

**Status:** Accepted
**Date:** 2026-03-01

**Context:** The RAG Knowledge Assistant needs a programming language with strong support for the AI/ML ecosystem, particularly LangChain, embedding models, vector database clients, and LLM SDKs. The team needs to choose a single primary language for the entire application (API server, data pipeline, tests, tooling).

**Decision:** Python 3.12 is the primary and only language for this project. All application code, tests, scripts, and tooling are written in Python. No other languages are used for application logic.

**Alternatives considered:**
- **TypeScript:** LangChain.js exists and is maturing, but the ecosystem is significantly smaller. Many Python-first libraries (sentence-transformers, some document loaders, scientific computing tools) have no TypeScript equivalent. The qdrant-client JS SDK has fewer features than the Python SDK. Server-side streaming with Node.js is well-supported, but the overall AI/ML ecosystem maturity favors Python.
- **Go:** Excellent for high-performance API servers. However, there is no mature LangChain equivalent. The go-openai library exists but provides only raw API access with no chain composition, prompt templating, or document loading abstractions. Building the RAG pipeline from scratch in Go would take significantly longer.
- **Rust:** Similar to Go — excellent performance but no RAG ecosystem. Would require FFI bindings to Python libraries, defeating the purpose.

**Consequences:**
- Positive: Access to the entire Python AI/ML ecosystem. Large hiring pool familiar with Python. Extensive LangChain documentation and examples in Python. Fast prototyping with dynamic typing. Strong async support in Python 3.12.
- Negative: Python is slower than Go/Rust for CPU-bound operations. The GIL limits true parallelism for CPU-bound tasks (mitigated by using async for I/O and ProcessPoolExecutor for CPU-bound work). Deployment size is larger than Go binaries. Type safety is opt-in (mitigated by type hints, Pydantic, and ruff).

---

## ADR-002: FastAPI Over Django and Flask

**Status:** Accepted
**Date:** 2026-03-01

**Context:** The application needs an HTTP framework for the API server. The API is JSON-only (no HTML templates), has no admin UI, does not need a SQL ORM for primary storage (Qdrant is the primary store), and requires async support for streaming LLM responses and concurrent Qdrant queries.

**Decision:** FastAPI 0.115 is the HTTP framework. It provides async-first request handling, automatic OpenAPI documentation, Pydantic validation integrated into the request/response cycle, and dependency injection for managing shared resources like the QdrantClient.

**Alternatives considered:**
- **Django + Django REST Framework:** Too heavy for this use case. Django's ORM, admin interface, template engine, middleware stack, and authentication system are unused overhead. Django's async support exists but is not first-class — many Django features still require sync execution. DRF adds another layer of abstraction on top.
- **Flask:** Lightweight and familiar, but lacks native async support. Flask 2.x added `async def` route support, but it runs each async handler in a new thread rather than on an event loop, losing the benefits of true async (connection sharing, streaming). Flask also lacks built-in Pydantic validation and OpenAPI generation.
- **Litestar:** Async-first like FastAPI, with similar features. Less community adoption, fewer tutorials, smaller ecosystem of extensions. A viable alternative but with less community support.

**Consequences:**
- Positive: Async request handling enables streaming LLM responses with `StreamingResponse`. Pydantic integration provides input validation, serialization, and OpenAPI docs automatically. Dependency injection manages the QdrantClient lifecycle cleanly. Large community and extensive documentation.
- Negative: FastAPI's dependency injection can be confusing for developers new to it. Starlette (the underlying ASGI framework) has occasional breaking changes. Middleware ordering can be non-obvious.

---

## ADR-003: Qdrant Over Pinecone, Weaviate, ChromaDB, and pgvector

**Status:** Accepted
**Date:** 2026-03-01

**Context:** The application needs a vector database for storing document embeddings and performing similarity search. Requirements: production-ready performance, payload filtering (filter by document source, page number, etc.), HNSW indexing with configurable parameters, a Python SDK that integrates with LangChain, and the ability to self-host for development and production.

**Decision:** Qdrant 1.12 with qdrant-client 1.12 is the vector database. It is deployed as a Docker container for both development and production. The langchain-qdrant package provides the LangChain integration via `QdrantVectorStore`.

**Alternatives considered:**
- **Pinecone:** Managed-only service with no self-hosting option. Vendor lock-in: all data is on Pinecone's infrastructure. Cost scales with vector count and query volume. Good SDK and LangChain integration. Rejected for vendor lock-in and cost at scale.
- **Weaviate:** Self-hostable with good features. However, it is heavier than Qdrant (requires more resources), has a more complex configuration model (schema definition, module system), and its Python SDK is more verbose. Good LangChain integration. Rejected for complexity relative to needs.
- **ChromaDB:** Lightweight and easy to start with. However, it is not production-ready for datasets beyond toy scale. Limited filtering capabilities. No built-in replication or sharding. Good for prototyping, not for production. Rejected for production readiness.
- **pgvector:** PostgreSQL extension for vector similarity search. Requires running and managing PostgreSQL. HNSW indexing was added relatively recently and is less mature than Qdrant's implementation. Filtering via SQL is powerful but adds SQL injection risk surface. No gRPC support. Rejected for maturity of vector-specific features and the overhead of managing PostgreSQL when we do not need relational data storage.

**Consequences:**
- Positive: Self-hosted Docker container — full control over data and infrastructure. Excellent Python SDK with both REST and gRPC support. Payload filtering for metadata-based queries. Configurable HNSW parameters (m, ef_construct) for tuning recall/speed trade-off. In-memory mode for testing (`QdrantClient(":memory:")`). Active development with frequent releases.
- Negative: Another Docker container to manage in production. Less managed-service convenience than Pinecone (must handle backups, scaling, monitoring). Smaller community than PostgreSQL ecosystem. Fewer managed hosting options.

---

## ADR-004: LangChain for RAG Orchestration

**Status:** Accepted
**Date:** 2026-03-01

**Context:** The RAG pipeline requires document loading (PDF, text), text splitting, embedding generation, vector store interaction, prompt template management, LLM invocation, and output parsing. These components need to be composed into a pipeline that is testable, configurable, and supports swapping providers (OpenAI to Ollama).

**Decision:** LangChain 0.3.x is the orchestration framework. Specifically: langchain-core for base abstractions (LCEL, RunnablePassthrough, StrOutputParser, ChatPromptTemplate), langchain-openai for OpenAI integration, langchain-ollama for Ollama integration, langchain-qdrant for vector store integration, langchain-community for document loaders, and langchain-text-splitters for chunking.

**Alternatives considered:**
- **Raw SDK calls (OpenAI SDK + qdrant-client + custom glue):** Maximum control, minimum abstraction. But requires building prompt templating, chain composition, provider switching, document loading, and text splitting from scratch. The code would be more verbose and less standardized. Debugging is harder without LangChain's callback system. Rejected for productivity cost.
- **LlamaIndex:** Another popular RAG framework. More opinionated about the RAG pipeline structure (index types, query engines). Less flexibility in chain composition compared to LangChain's LCEL. Stronger for structured data retrieval but more rigid for custom pipelines. Smaller ecosystem of integrations. Rejected for flexibility.
- **Haystack:** Good RAG framework with a pipeline abstraction. Less community adoption than LangChain. Fewer integrations with LLM providers and vector stores. Decent documentation but smaller ecosystem of tutorials and examples. Rejected for ecosystem size.

**Consequences:**
- Positive: LCEL provides a clean, composable pipe operator (`|`) for building chains. Built-in document loaders handle PDF, text, CSV, and many other formats. Text splitters are well-tested and configurable. Provider switching (OpenAI to Ollama) requires changing one import and one configuration value. Callbacks enable observability. Large community means problems are quickly answered.
- Negative: LangChain is a heavy dependency with many sub-packages. Breaking changes between versions have historically been a problem (mitigated in 0.3.x with the modular package structure). Some abstractions add overhead compared to raw SDK calls. The abstraction can hide what is actually happening (mitigated by understanding the explain skill's Layer 3 deep-dive).

---

## ADR-005: Custom Typed Exception Hierarchy

**Status:** Accepted
**Date:** 2026-03-01

**Context:** The application needs clear, consistent error reporting across API boundaries. When something fails — LLM timeout, Qdrant connection error, invalid file upload, malformed query — the API should return a structured error response with an appropriate HTTP status code and a message that helps the caller understand what went wrong and what to do about it.

**Decision:** A custom exception hierarchy with typed exceptions for each failure domain. Each exception class maps to a specific HTTP status code via FastAPI exception handlers. The hierarchy:

```python
class RAGError(Exception):
    """Base exception for all RAG application errors."""

class IngestionError(RAGError):
    """Raised when document ingestion fails."""
    # Maps to HTTP 422 (Unprocessable Entity) or 413 (Payload Too Large)

class RetrievalError(RAGError):
    """Raised when vector search or document retrieval fails."""
    # Maps to HTTP 404 (Not Found) or 502 (Bad Gateway)

class GenerationError(RAGError):
    """Raised when LLM generation fails."""
    # Maps to HTTP 502 (Bad Gateway) or 504 (Gateway Timeout)

class ConfigurationError(RAGError):
    """Raised when application configuration is invalid."""
    # Maps to HTTP 500 (Internal Server Error) — should not reach users

class AuthenticationError(RAGError):
    """Raised when API key validation fails."""
    # Maps to HTTP 401 (Unauthorized)
```

FastAPI exception handlers translate these into structured JSON responses:

```python
@app.exception_handler(RetrievalError)
async def retrieval_error_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "retrieval_error", "message": str(exc)},
    )
```

**Alternatives considered:**
- **HTTP status codes only (no custom exceptions):** Raising `HTTPException(status_code=502, detail="LLM failed")` directly in business logic. This couples business logic to HTTP semantics. The retrieval module should not know about HTTP status codes. Rejected for coupling concerns.
- **Generic Exception with message strings:** `raise Exception("LLM timeout")`. No type safety. Cannot selectively catch specific error types. Cannot map to different HTTP status codes based on exception type. Rejected for lack of type safety.
- **Error codes as return values:** `return {"error": "LLM_TIMEOUT", "data": None}`. Requires every caller to check for errors. Easy to forget, leading to silent failures. Not Pythonic. Rejected for error-handling ergonomics.

**Consequences:**
- Positive: Business logic raises domain-specific exceptions without knowing about HTTP. FastAPI handlers translate exceptions to appropriate HTTP responses. Each exception type is catchable individually for targeted error handling. Exception hierarchy allows catching all RAG errors with `except RAGError`. Testable — tests can assert specific exception types.
- Negative: More boilerplate than using `HTTPException` directly. Must remember to add exception handlers when adding new exception types. Must keep the handler mapping up to date. Slight overhead from the exception hierarchy (negligible in practice).
