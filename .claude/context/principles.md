# Engineering Principles

These 10 principles guide all engineering decisions in the RAG Knowledge Assistant. Each principle is stated, explained, and illustrated with a concrete example from the Python/FastAPI/LangChain/Qdrant stack. When principles conflict, the order below reflects priority: correctness before performance, security before convenience.

---

## 1. Single Responsibility

Every module, class, and function should have exactly one reason to change. A LangChain chain should do one thing: retrieve, generate, or format — not all three in a single function. A FastAPI route handler should validate input and call a service function, not contain business logic.

**Example:** The retrieval chain retrieves relevant document chunks. It does not format them for display, score them, or generate answers. Formatting is a separate function. Generation is a separate chain. They compose together via LangChain's pipe operator, but each component has one job.

```python
# Single responsibility: each component does one thing
retrieval_chain = retriever | format_docs  # retrieves and formats
generation_chain = prompt | llm | StrOutputParser()  # generates answer
full_chain = {"context": retrieval_chain, "question": RunnablePassthrough()} | generation_chain
```

If you find a function that retrieves documents AND generates answers AND logs metrics, split it. The function name should describe one action: `retrieve_relevant_chunks`, not `retrieve_and_generate_and_log`.

---

## 2. Explicit Over Implicit

Configuration, dependencies, and behavior should be visible and explicit. No hidden magic, no global state mutation, no implicit type coercion. If a function needs a QdrantClient, it should receive it as a parameter (dependency injection), not import a global singleton.

**Example:** FastAPI's dependency injection system makes dependencies explicit. The route handler declares what it needs, and FastAPI provides it.

```python
# Explicit: dependency is declared and injected
@app.post("/query")
async def query(
    request: QueryRequest,
    chain: RunnableSequence = Depends(get_retrieval_chain),
):
    return await chain.ainvoke(request.question)

# NOT implicit: hidden global import
from src.globals import chain  # where does this come from? what state does it have?
```

Configuration values come from `Settings` (pydantic-settings), not from hardcoded strings scattered through the codebase. Environment variables are loaded once, validated by Pydantic, and accessed through a typed settings object.

---

## 3. Fail Fast, Fail Loud

When something goes wrong, raise a typed exception immediately. Do not return `None` to indicate failure. Do not catch exceptions and silently continue. Do not log an error and return a default value. The caller deserves to know what happened so it can decide how to respond.

**Example:** When Qdrant returns no results because the collection does not exist, raise a `RetrievalError` immediately. Do not return an empty list and let the LLM hallucinate an answer from empty context.

```python
async def retrieve_chunks(query: str) -> list[Document]:
    try:
        results = await retriever.ainvoke(query)
    except UnexpectedResponse as e:
        raise RetrievalError(f"Qdrant query failed: {e}") from e

    if not results:
        raise RetrievalError("No relevant documents found for this query.")

    return results
```

The FastAPI exception handler translates `RetrievalError` into a 404 response with a structured error body. The user sees a clear message, not a generic 500.

---

## 4. Composition Over Inheritance

Build complex behavior by composing simple, focused components — not by inheriting from deep class hierarchies. LangChain's LCEL (LangChain Expression Language) is designed around this principle: you compose chains with the pipe (`|`) operator, connecting small runnables into pipelines.

**Example:** Instead of creating a `BaseRAGChain` class with `RetrievalRAGChain` and `StreamingRAGChain` subclasses, compose the behavior from independent components.

```python
# Composition: build behavior by connecting components
base_chain = prompt | llm | StrOutputParser()
retrieval_chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | base_chain

# NOT inheritance: deep class hierarchy
class BaseRAGChain:  # don't do this
    def retrieve(self): ...
    def generate(self): ...
class StreamingRAGChain(BaseRAGChain):  # don't do this
    def generate(self): ...  # override for streaming
```

This applies beyond LangChain. FastAPI middleware, Pydantic validators, and Python decorators all favor composition. When you need to add caching to retrieval, wrap the retriever with a caching layer — do not modify the retriever class.

---

## 5. Immutable By Default

Data structures should be immutable unless mutation is specifically required. Immutable objects are easier to reason about, safer in concurrent contexts, and prevent a class of bugs where shared state is modified unexpectedly. Use frozen Pydantic models for configuration and data transfer objects.

**Example:** Application settings should never change after startup.

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_config = {"frozen": True, "env_file": ".env"}

    openai_api_key: str
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    chunk_size: int = 512
    chunk_overlap: int = 50
```

Attempting to modify `settings.chunk_size = 1024` after initialization raises a `ValidationError`. If you need different settings for a specific operation, create a new settings object — do not mutate the global one. This principle extends to LangChain Document objects: create new documents with modified metadata rather than mutating existing ones.

---

## 6. Test at the Boundaries

Focus testing effort at the boundaries between your code and external systems: the LLM, the vector database, the file system, the HTTP API. Mock the external system and verify that your code sends the right requests and handles the right responses. Internal logic tests are valuable, but boundary tests catch the bugs that actually break production.

**Example:** Mock the LLM and verify the chain sends the correctly formatted prompt. Mock Qdrant and verify the retriever sends the right query vector with the right parameters.

```python
@pytest.mark.asyncio
async def test_chain_sends_correct_prompt():
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(content="test answer")

    chain = prompt | mock_llm | StrOutputParser()
    await chain.ainvoke({"context": "test context", "question": "test question"})

    # Verify the prompt sent to the LLM contains the context and question
    call_args = mock_llm.ainvoke.call_args[0][0]
    assert "test context" in str(call_args)
    assert "test question" in str(call_args)
```

Use `QdrantClient(":memory:")` for integration tests that need a real vector database without network dependencies. Use `httpx.AsyncClient` with `ASGITransport` for API endpoint tests without starting a server.

---

## 7. Observe Everything

Every operation that crosses a system boundary should be logged, timed, and traced. When something goes wrong in production, you need to answer three questions: What happened? When? Where in the pipeline? Without observability, debugging a RAG pipeline is guesswork because failures can occur at embedding, retrieval, prompt formatting, or generation — and each looks like "wrong answer" from the user's perspective.

**Example:** Log at every pipeline step with structlog context.

```python
logger.info("query_received", question_length=len(question))
logger.info("retrieval_completed", chunk_count=len(chunks), latency_ms=retrieval_ms)
logger.info("generation_completed", answer_length=len(answer), latency_ms=generation_ms)
```

Add prometheus metrics for query latency, ingestion throughput, and error rates. Add OpenTelemetry spans for each pipeline step so traces show where time is spent. The observability stack (structlog + prometheus-client + opentelemetry-sdk) is part of the project's core dependencies, not an afterthought.

---

## 8. Secure By Default

Every input is untrusted until validated. Every output is unsanitized until proven otherwise. API keys live in environment variables, not source code. User questions go through Pydantic validation before touching the prompt template. File uploads are checked for size and type before processing. Prompt templates use LangChain's `{variable}` placeholders, never f-strings with user input.

**Example:** The `QueryRequest` model validates input before the route handler runs any logic.

```python
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question must not be blank")
        return v.strip()
```

Security is not a feature to add later — it is a constraint that shapes every design decision. See the security-check skill for the full audit checklist.

---

## 9. Automate the Boring Stuff

Formatting, linting, testing, and dependency auditing should happen automatically. Human reviewers should not spend time on import ordering or trailing whitespace. ruff handles linting and formatting. pytest runs on every change. pip-audit checks for vulnerable dependencies. These tools are configured once and enforced consistently.

**Commands that should be run regularly (and ideally in CI):**
- `uv run ruff check .` — lint for errors and style violations
- `uv run ruff format .` — auto-format all Python files
- `uv run pytest` — run all tests
- `uv run pip-audit` — check for known CVEs in dependencies

If a check can be automated, it should be. If it cannot be automated, it should be documented in a checklist (like the code-review skill). The goal is to free human attention for design, architecture, and the problems that machines cannot solve.

---

## 10. Simple Until Proven Otherwise

Do not add complexity until you have measured a need. Do not add caching until you have measured that the same queries are repeated. Do not add a message queue until you have measured that synchronous processing cannot handle the load. Do not add microservices until you have measured that a monolith cannot scale.

**Example:** Start with a synchronous ingestion endpoint. If ingestion takes too long and blocks the API, measure the bottleneck first. If it is embedding computation, try batching before adding a background task queue. If batching is sufficient, stop there. If it is not, then add Celery or a background worker — with the measurement data to justify the complexity.

```python
# Start simple
@app.post("/ingest")
async def ingest(file: UploadFile):
    documents = await load_document(file)
    chunks = split_documents(documents)
    await vector_store.aadd_documents(chunks)
    return {"chunks_ingested": len(chunks)}

# Only add complexity when measured need exists
# (background tasks, queues, etc.)
```

Every abstraction, every indirection, every additional service has a maintenance cost. The simplest solution that meets the measured requirements is the correct solution. Premature optimization is the root of much unnecessary complexity.
