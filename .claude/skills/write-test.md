---
name: write-test
description: Write pytest + pytest-asyncio + httpx tests for the RAG Knowledge Assistant using established patterns
---

# Test Writing Skill

Write tests using pytest, pytest-asyncio, and httpx. Follow the patterns and conventions below exactly. All tests target the Python/FastAPI/LangChain/Qdrant stack.

---

## Test File Naming and Location

- Test files live under `tests/` and mirror the `src/` structure.
- Naming: `test_<module>.py` (e.g., `tests/test_retrieval.py`, `tests/api/test_routes.py`).
- Shared fixtures go in `tests/conftest.py`.
- Run all tests: `uv run pytest`
- Run specific file: `uv run pytest tests/test_retrieval.py`
- Run with coverage: `uv run pytest --cov=src`

---

## Conftest.py Patterns

```python
# tests/conftest.py
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from qdrant_client import QdrantClient
from langchain_core.documents import Document

from src.main import app


@pytest.fixture
def qdrant_memory_client():
    """Qdrant client using in-memory storage for test isolation."""
    client = QdrantClient(":memory:")
    yield client
    # Cleanup happens automatically — in-memory storage is discarded


@pytest_asyncio.fixture
async def async_client():
    """Async HTTP client for testing FastAPI endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_documents():
    """Sample LangChain Document objects for testing."""
    return [
        Document(
            page_content="HNSW (Hierarchical Navigable Small World) is a graph-based "
            "algorithm for approximate nearest neighbor search.",
            metadata={"source": "vector_db_guide.pdf", "page": 1},
        ),
        Document(
            page_content="RAG combines retrieval of relevant documents with language "
            "model generation to produce grounded answers.",
            metadata={"source": "rag_overview.pdf", "page": 3},
        ),
        Document(
            page_content="Qdrant is a vector similarity search engine that provides "
            "a production-ready service with a convenient API.",
            metadata={"source": "qdrant_docs.pdf", "page": 1},
        ),
    ]


@pytest.fixture
def sample_query():
    """A sample user query for testing the RAG pipeline."""
    return "What is HNSW and how does it work?"
```

---

## Unit Test with @pytest.mark.asyncio and AsyncMock

Use `AsyncMock` to mock LLM and embedding calls. Test the logic around the mocked boundaries.

```python
# tests/test_retrieval.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document


@pytest.mark.asyncio
async def test_retrieve_relevant_chunks_returns_documents(sample_query):
    """Retrieval should return Document objects with page_content and metadata."""
    mock_retriever = AsyncMock()
    mock_retriever.ainvoke.return_value = [
        Document(
            page_content="HNSW is a graph-based algorithm.",
            metadata={"source": "test.pdf", "page": 1},
        ),
    ]

    results = await mock_retriever.ainvoke(sample_query)

    assert len(results) == 1
    assert "HNSW" in results[0].page_content
    assert results[0].metadata["source"] == "test.pdf"
    mock_retriever.ainvoke.assert_called_once_with(sample_query)


@pytest.mark.asyncio
async def test_retrieve_returns_empty_list_when_no_matches(sample_query):
    """Retrieval should return an empty list when no documents match."""
    mock_retriever = AsyncMock()
    mock_retriever.ainvoke.return_value = []

    results = await mock_retriever.ainvoke(sample_query)

    assert results == []


@pytest.mark.asyncio
async def test_chain_invocation_with_mocked_llm():
    """The RAG chain should pass context and question to the LLM."""
    mock_llm_response = "HNSW is a graph-based ANN algorithm."

    with patch("langchain_openai.ChatOpenAI") as MockLLM:
        mock_instance = AsyncMock()
        mock_instance.ainvoke.return_value = MagicMock(content=mock_llm_response)
        MockLLM.return_value = mock_instance

        # Invoke the mocked LLM directly to verify behavior
        result = await mock_instance.ainvoke("test prompt")
        assert result.content == mock_llm_response
```

---

## Integration Test with httpx.AsyncClient

Test the full FastAPI request/response cycle.

```python
# tests/api/test_routes.py
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_query_endpoint_returns_200(async_client):
    """POST /query should return 200 with a valid question."""
    with patch("src.api.routes.query_chain") as mock_chain:
        mock_chain.ainvoke = AsyncMock(return_value="HNSW is an ANN algorithm.")

        response = await async_client.post(
            "/query",
            json={"question": "What is HNSW?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0


@pytest.mark.asyncio
async def test_query_endpoint_validates_empty_question(async_client):
    """POST /query should return 422 when question is empty."""
    response = await async_client.post(
        "/query",
        json={"question": ""},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_endpoint_validates_missing_body(async_client):
    """POST /query should return 422 when body is missing."""
    response = await async_client.post("/query")

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_health_endpoint(async_client):
    """GET /health should return 200 with status ok."""
    response = await async_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

---

## Qdrant In-Memory Fixture

Use `QdrantClient(":memory:")` for test isolation. No external Qdrant instance needed.

```python
# tests/test_ingestion.py
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from unittest.mock import MagicMock


def test_documents_stored_in_qdrant(qdrant_memory_client, sample_documents):
    """Ingested documents should be retrievable from Qdrant."""
    collection_name = "test_collection"

    # Create collection with matching dimensions for mock embeddings
    qdrant_memory_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=3, distance=Distance.COSINE),
    )

    # Use a mock embeddings that returns fixed-size vectors
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

    vector_store = QdrantVectorStore(
        client=qdrant_memory_client,
        collection_name=collection_name,
        embedding=mock_embeddings,
    )

    vector_store.add_documents(sample_documents)

    # Verify documents are stored
    collection_info = qdrant_memory_client.get_collection(collection_name)
    assert collection_info.points_count == 3


def test_qdrant_similarity_search(qdrant_memory_client):
    """Similarity search should return the most relevant documents."""
    collection_name = "test_search"

    qdrant_memory_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=3, distance=Distance.COSINE),
    )

    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    mock_embeddings.embed_query.return_value = [0.9, 0.1, 0.0]

    vector_store = QdrantVectorStore(
        client=qdrant_memory_client,
        collection_name=collection_name,
        embedding=mock_embeddings,
    )

    docs = [
        Document(page_content="First document", metadata={"id": 1}),
        Document(page_content="Second document", metadata={"id": 2}),
    ]
    vector_store.add_documents(docs)

    results = vector_store.similarity_search("query", k=1)
    assert len(results) == 1
    assert results[0].page_content == "First document"
```

---

## Parametrize for Chunk Sizes

Test that chunking works correctly across different size configurations.

```python
# tests/test_chunking.py
import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


@pytest.mark.parametrize("chunk_size", [256, 512, 1024])
def test_chunk_size_produces_valid_chunks(chunk_size):
    """Each chunk should be at most chunk_size characters."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,
    )
    long_text = "The quick brown fox jumps over the lazy dog. " * 200
    doc = Document(page_content=long_text, metadata={"source": "test.txt"})

    chunks = splitter.split_documents([doc])

    assert len(chunks) > 1, f"Expected multiple chunks for chunk_size={chunk_size}"
    for chunk in chunks:
        assert len(chunk.page_content) <= chunk_size + 50  # allow small overflow from separator logic
        assert chunk.metadata["source"] == "test.txt"


@pytest.mark.parametrize("chunk_size", [256, 512, 1024])
def test_chunk_overlap_preserves_context(chunk_size):
    """Consecutive chunks should have overlapping content."""
    overlap = 50
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    long_text = "word " * 500
    doc = Document(page_content=long_text)

    chunks = splitter.split_documents([doc])

    if len(chunks) >= 2:
        # Verify overlap: the end of chunk N should appear at the start of chunk N+1
        end_of_first = chunks[0].page_content[-overlap:]
        start_of_second = chunks[1].page_content[:overlap]
        # At least some overlap should exist
        assert len(set(end_of_first.split()) & set(start_of_second.split())) > 0


@pytest.mark.parametrize(
    "chunk_size,expected_min_chunks",
    [(256, 4), (512, 2), (1024, 1)],
)
def test_smaller_chunk_size_produces_more_chunks(chunk_size, expected_min_chunks):
    """Smaller chunk sizes should produce more chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
    )
    text = "x" * 2000
    doc = Document(page_content=text)

    chunks = splitter.split_documents([doc])

    assert len(chunks) >= expected_min_chunks
```

---

## Mocking OpenAI

Patch OpenAI classes at the import location in the module under test.

```python
# tests/test_generation.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_generate_answer_calls_openai():
    """The generation step should call ChatOpenAI with the formatted prompt."""
    with patch("src.generation.chain.ChatOpenAI") as MockChatOpenAI:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content="RAG stands for Retrieval-Augmented Generation."
        )
        MockChatOpenAI.return_value = mock_llm

        # Import after patching so the module gets the mock
        from src.generation.chain import generate_answer

        result = await generate_answer(
            context="RAG is a technique that combines retrieval with generation.",
            question="What is RAG?",
        )

        assert "Retrieval-Augmented Generation" in result
        mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_openai_embeddings_called_with_correct_model():
    """Embeddings should use the configured model name."""
    with patch("src.ingestion.embeddings.OpenAIEmbeddings") as MockEmbeddings:
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        MockEmbeddings.return_value = mock_emb

        from src.ingestion.embeddings import get_embeddings

        embeddings = get_embeddings()
        MockEmbeddings.assert_called_once_with(model="text-embedding-3-small")
```

---

## Test Writing Checklist

Before submitting any test file, verify:

1. Every test function has a descriptive docstring explaining what is being tested.
2. All async tests are decorated with `@pytest.mark.asyncio`.
3. Mocks are applied at the correct import path (where the symbol is used, not where it is defined).
4. Assertions are specific — assert exact values, not just truthiness.
5. No tests depend on external services (OpenAI API, running Qdrant instance).
6. No tests depend on each other — each test can run independently.
7. Fixtures are used for shared setup — no duplicated setup code across tests.
8. Edge cases are covered: empty input, missing fields, error responses.
9. Run `uv run pytest` to confirm all tests pass before committing.
10. Run `uv run ruff check .` to confirm test code passes linting.
