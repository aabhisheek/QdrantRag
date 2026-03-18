# Conventions

This document defines the naming, file layout, import, testing, and commit conventions for the RAG Knowledge Assistant. All contributors and automated agents follow these conventions to maintain consistency across the codebase. When conventions conflict with external library patterns, the library's pattern wins within its integration code, but these conventions apply everywhere else.

---

## Naming Conventions

### Functions and Variables

Use `snake_case` for all function names and variable names. Function names should be verb phrases that describe what the function does. Variable names should be noun phrases that describe what the variable holds.

```python
# Functions — verb phrases
def retrieve_relevant_chunks(query: str) -> list[Document]: ...
def build_retrieval_chain(retriever, llm) -> RunnableSequence: ...
def ingest_document(file_path: str) -> int: ...
def format_documents_as_context(documents: list[Document]) -> str: ...
async def handle_query_request(request: QueryRequest) -> QueryResponse: ...

# Variables — noun phrases
query_embedding = embeddings.embed_query(question)
relevant_chunks = retriever.invoke(question)
formatted_context = format_documents_as_context(chunks)
collection_info = qdrant_client.get_collection("documents")
```

### Classes

Use `PascalCase` for all class names. Classes should be noun phrases. Pydantic models describe the data they hold. Exception classes end with `Error`.

```python
class QueryRequest(BaseModel): ...
class QueryResponse(BaseModel): ...
class DocumentSource(BaseModel): ...
class RetrievalError(Exception): ...
class GenerationError(Exception): ...
class OpenAIProvider: ...
class IngestionPipeline: ...
```

### Constants

Use `SCREAMING_SNAKE_CASE` for module-level constants. Constants should be defined at the top of the module, after imports.

```python
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
COLLECTION_NAME = "documents"
```

### Modules

Module names are singular nouns in `snake_case`. A module describes a single concept or responsibility. Use singular form: `retrieval` not `retrievals`, `ingestion` not `ingestions`.

```
src/
  retrieval/      # not retrievals/
  ingestion/      # not ingestions/
  generation/     # not generators/
  core/           # shared configuration, logging, metrics
  api/            # HTTP route handlers
  provider/       # LLM provider implementations
```

### LangChain-Specific Naming

Chain variables end with `_chain`. Prompt templates end with `_prompt` or `_template`. Retrievers end with `_retriever`. These suffixes make it immediately clear what type of LangChain component a variable holds.

```python
retrieval_chain = retriever | format_docs
generation_chain = prompt | llm | StrOutputParser()
full_rag_chain = {"context": retrieval_chain, "question": RunnablePassthrough()} | generation_chain

rag_prompt = ChatPromptTemplate.from_template("Context: {context}\nQuestion: {question}")
system_template = "You are a helpful assistant that answers questions based on provided context."

document_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
mmr_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})
```

---

## File Layout

The project uses a feature-grouped module structure under `src/`. Each feature module is a Python package (directory with `__init__.py`). The `__init__.py` file exports the module's public API.

```
project-root/
  src/
    __init__.py
    main.py                     # FastAPI app creation, lifespan, middleware
    api/
      __init__.py
      routes.py                 # Route handlers for /query, /ingest, /health
      dependencies.py           # FastAPI dependency injection functions
    core/
      __init__.py
      config.py                 # Settings class (pydantic-settings)
      logging.py                # structlog configuration
      metrics.py                # prometheus-client metric definitions
      tracing.py                # OpenTelemetry setup
      exceptions.py             # Custom exception hierarchy
    retrieval/
      __init__.py
      chain.py                  # Retrieval chain composition
      vector_store.py           # QdrantVectorStore setup and management
    ingestion/
      __init__.py
      pipeline.py               # Document loading, splitting, embedding, storing
      loaders.py                # Document loader factory (PDF, text, etc.)
      splitter.py               # Text splitter configuration
    generation/
      __init__.py
      chain.py                  # Generation chain (prompt + LLM + parser)
      prompts.py                # Prompt templates
    provider/
      __init__.py
      base.py                   # BaseLLMProvider protocol
      openai_provider.py        # OpenAI implementation
      ollama_provider.py        # Ollama implementation
  tests/
    __init__.py
    conftest.py                 # Shared fixtures
    api/
      __init__.py
      test_routes.py
    test_retrieval.py
    test_ingestion.py
    test_generation.py
    test_chunking.py
  docker-compose.yml
  Dockerfile
  pyproject.toml
  .env.example
  .gitignore
```

### __init__.py Exports

Each `__init__.py` exports the module's public API. Other modules import from the package, not from internal files.

```python
# src/retrieval/__init__.py
from src.retrieval.chain import build_retrieval_chain
from src.retrieval.vector_store import get_vector_store

__all__ = ["build_retrieval_chain", "get_vector_store"]
```

```python
# Importing from outside the module
from src.retrieval import build_retrieval_chain  # GOOD — imports from __init__.py
from src.retrieval.chain import build_retrieval_chain  # AVOID — imports from internal file
```

---

## Import Conventions

Imports are organized into three groups separated by blank lines, enforced by ruff's isort integration:

1. **Standard library** — `import os`, `from pathlib import Path`, `import asyncio`
2. **Third-party packages** — `from fastapi import FastAPI`, `from langchain_core.documents import Document`
3. **Local modules** — `from src.core.config import settings`, `from src.retrieval import build_retrieval_chain`

```python
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient

from src.core.config import settings
from src.core.exceptions import IngestionError, RetrievalError
from src.retrieval import build_retrieval_chain
```

### Rules

- **Absolute imports only.** Use `from src.retrieval.chain import ...`, not `from .chain import ...`. Relative imports cause confusion when modules are moved.
- **No wildcard imports.** `from module import *` obscures what is being used and can cause name collisions.
- **No unused imports.** Ruff catches these automatically. Remove them immediately.
- **Import from the package, not internal files**, when the package has an `__init__.py` that exports the symbol.
- **Run `uv run ruff check .`** to verify import organization.

---

## Testing Conventions

### File Naming

Test files are named `test_<module>.py` and placed under `tests/`, mirroring the `src/` structure for feature-specific tests.

```
tests/
  conftest.py              # Shared fixtures (qdrant client, async client, sample data)
  test_retrieval.py        # Tests for src/retrieval/
  test_ingestion.py        # Tests for src/ingestion/
  test_generation.py       # Tests for src/generation/
  test_chunking.py         # Tests for text splitting
  api/
    conftest.py            # API-specific fixtures
    test_routes.py         # Tests for src/api/routes.py
```

### Test Functions

Test functions are named `test_<what_is_being_tested>_<expected_behavior>`. They describe the scenario and expected outcome.

```python
def test_retrieval_returns_relevant_chunks(): ...
def test_retrieval_returns_empty_list_when_no_matches(): ...
def test_ingestion_creates_chunks_with_metadata(): ...
def test_query_endpoint_returns_422_for_empty_question(): ...
async def test_streaming_endpoint_returns_chunked_response(): ...
```

### Async Tests

All async tests must be decorated with `@pytest.mark.asyncio`. Use `AsyncMock` for mocking async functions.

```python
@pytest.mark.asyncio
async def test_chain_returns_answer():
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = "test answer"
    result = await mock_chain.ainvoke("test question")
    assert result == "test answer"
```

### Fixtures

Shared fixtures live in `conftest.py`. Fixtures should be as narrow as possible — provide exactly what the test needs, no more.

```python
@pytest.fixture
def qdrant_memory_client():
    return QdrantClient(":memory:")

@pytest_asyncio.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
```

### Test Grouping

Group related tests in classes when they share setup or test the same component. Use descriptive class names prefixed with `Test`.

```python
class TestRetrievalChain:
    def test_returns_documents_for_valid_query(self): ...
    def test_returns_empty_for_unrelated_query(self): ...
    def test_raises_error_for_missing_collection(self): ...

class TestIngestionPipeline:
    def test_splits_pdf_into_chunks(self): ...
    def test_preserves_metadata_through_splitting(self): ...
```

---

## Commit Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/) format. The commit subject line follows this pattern:

```
<type>(<optional scope>): <description>
```

### Types

| Type | When to use |
|---|---|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or updating tests |
| `docs` | Documentation only |
| `chore` | Maintenance tasks (dependency updates, config changes) |
| `ci` | CI/CD configuration changes |
| `perf` | Performance improvement |
| `style` | Formatting, whitespace (no logic change) |
| `build` | Build system or dependency changes |

### Rules

- Subject line in present tense, imperative mood: "add streaming support" not "added streaming support" or "adds streaming support".
- Subject line under 72 characters.
- Optional body separated by blank line, wrapping at 72 characters.
- Reference issue numbers in the body: `Closes #42`.
- Breaking changes noted with `BREAKING CHANGE:` in the body.

### Examples

```
feat(retrieval): add MMR search strategy for diverse results

Add Maximal Marginal Relevance search option to reduce redundancy
in retrieved chunks. Configurable via search_type parameter.

Closes #23
```

```
fix(ingestion): handle Unicode decode errors in text file loading

Text files with non-UTF-8 encoding caused UnicodeDecodeError during
ingestion. Now falls back to latin-1 encoding and logs a warning.

Closes #31
```

```
refactor(generation): extract prompt templates into dedicated module

Move all prompt templates from chain.py to prompts.py for
easier management and testing.
```

```
test: add integration tests for /query endpoint

Cover success, validation error, and missing collection scenarios
using httpx AsyncClient with ASGITransport.
```
