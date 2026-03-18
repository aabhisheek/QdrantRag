---
name: code-review
description: Comprehensive code review protocol for Python/FastAPI/LangChain/Qdrant with a 10-point checklist
---

# Code Review Skill

Apply this 10-point checklist to every code change. Each point includes what to check, what "good" looks like, and common violations in the RAG Knowledge Assistant stack.

---

## 1. Correctness

**Check:** Does the code do what it claims to do? Are LangChain chains composed correctly? Are Qdrant queries using the right distance metric?

**What to verify:**
- Chain components are piped in the correct order (retriever -> formatter -> prompt -> llm -> parser).
- `RunnablePassthrough()` is used where the input should flow through unchanged.
- `StrOutputParser()` is at the end of chains that should return strings (not `AIMessage` objects).
- Qdrant collection uses the correct distance metric for the embedding model (COSINE for OpenAI embeddings).
- Embedding dimensions match the collection's vector configuration (1536 for `text-embedding-3-small`).
- Prompt template variables (`{context}`, `{question}`) match the keys provided by the chain.

**Common violations:**
```python
# WRONG — missing StrOutputParser, returns AIMessage instead of string
chain = prompt | llm  # returns AIMessage(content="...")

# WRONG — variable name mismatch
prompt = ChatPromptTemplate.from_template("Context: {docs}\nQuestion: {query}")
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt  # KeyError: 'docs'

# WRONG — wrong distance metric
VectorParams(size=1536, distance=Distance.EUCLID)  # should be COSINE for normalized embeddings
```

---

## 2. Error Handling

**Check:** All external calls (LLM, Qdrant, file I/O) are wrapped in try/except with typed exceptions. No bare `except:`.

**What to verify:**
- LLM calls handle `openai.APIError`, `openai.RateLimitError`, `openai.APITimeoutError`.
- Qdrant calls handle `qdrant_client.http.exceptions.UnexpectedResponse`.
- File I/O handles `FileNotFoundError`, `PermissionError`, `UnicodeDecodeError`.
- Custom exceptions are raised with clear messages and mapped to HTTP status codes via FastAPI exception handlers.
- No bare `except:` or `except Exception:` without re-raising or logging.

**Good pattern:**
```python
from src.exceptions import RetrievalError

async def retrieve_chunks(query: str) -> list[Document]:
    try:
        return await retriever.ainvoke(query)
    except UnexpectedResponse as e:
        if "not found" in str(e):
            raise RetrievalError("Collection not found. Please ingest documents first.") from e
        raise RetrievalError(f"Qdrant query failed: {e}") from e
```

**Bad pattern:**
```python
# BAD — bare except swallows all errors silently
try:
    result = await chain.ainvoke(question)
except:
    pass

# BAD — catches too broadly, including KeyboardInterrupt and SystemExit
try:
    result = await chain.ainvoke(question)
except Exception:
    return {"answer": "Something went wrong"}
```

---

## 3. Async Correctness

**Check:** No sync blocking in async handlers. Proper `await` on all coroutines. No fire-and-forget tasks without error handling.

**What to verify:**
- All LangChain calls in async handlers use `ainvoke()`, `astream()`, `aget_relevant_documents()`, not their sync counterparts.
- CPU-bound operations (local embeddings, PDF parsing) use `asyncio.to_thread()` or `run_in_executor()`.
- No `time.sleep()` in async code (use `asyncio.sleep()` instead).
- Background tasks created with `asyncio.create_task()` have error handling via `task.add_done_callback()` or try/except in the coroutine.
- No forgotten `await` (calling `chain.ainvoke()` without `await` returns a coroutine object, not the result).

**Detection:** Look for sync method calls inside `async def` functions.

```python
# BAD — sync invoke in async handler
async def handle_query(question: str):
    result = chain.invoke(question)  # blocks event loop

# BAD — forgotten await
async def handle_query(question: str):
    result = chain.ainvoke(question)  # result is a coroutine, not the answer
    return result  # returns coroutine object

# GOOD
async def handle_query(question: str):
    result = await chain.ainvoke(question)
    return result
```

---

## 4. Testing

**Check:** Every new function has tests. Mocks are minimal and targeted. Integration tests exist for API endpoints.

**What to verify:**
- New public functions have corresponding test functions.
- Tests use `@pytest.mark.asyncio` for async code.
- Mocks target the correct import path (where the symbol is used, not where it is defined).
- Tests assert specific values, not just truthiness.
- Edge cases are tested: empty input, missing data, error conditions.
- Integration tests use `httpx.AsyncClient` with `ASGITransport`.
- No tests are skipped (`@pytest.mark.skip`) without a documented reason.

---

## 5. Typing

**Check:** All function signatures are typed. Pydantic models for request/response. No `Any` without justification.

**What to verify:**
- All function parameters have type annotations.
- All function return types are specified.
- Pydantic `BaseModel` subclasses for all API request and response bodies.
- Generic types use modern syntax (`list[str]` not `List[str]`, `dict[str, int]` not `Dict[str, int]`).
- `Any` is only used when the type is genuinely unknown and documented with a comment explaining why.
- Type aliases are used for complex types to improve readability.

```python
# GOOD
async def retrieve_chunks(query: str, top_k: int = 4) -> list[Document]:
    ...

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)

class QueryResponse(BaseModel):
    answer: str
    sources: list[DocumentSource]

# BAD
async def retrieve_chunks(query, top_k=4):  # no types
    ...

def process(data: Any) -> Any:  # Any without justification
    ...
```

---

## 6. Security

**Check:** No hardcoded secrets. Input validated. Prompt injection considered.

**What to verify:**
- No API keys, passwords, or tokens in source code (only in `.env`).
- All user input goes through Pydantic validation before processing.
- Prompt templates use LangChain `{variable}` placeholders, not f-strings.
- File uploads are validated for size and type.
- URLs (if accepted) are validated against allowlists.
- Refer to the security-check skill for the full audit checklist.

---

## 7. Performance

**Check:** Batch operations used. No N+1 queries to Qdrant. Streaming where appropriate.

**What to verify:**
- Document ingestion uses `add_documents()` with batch processing, not individual inserts.
- No loops that make individual Qdrant queries when a batch query would work.
- LLM streaming is used for user-facing query endpoints.
- QdrantClient is shared (not created per request).
- Embedding results are cacheable for repeated queries.
- Refer to the perf-check skill for the full performance checklist.

---

## 8. Naming

**Check:** Functions describe what they do. Variables describe what they hold. No abbreviations except standard ones.

**What to verify:**
- Functions use verb phrases: `retrieve_relevant_chunks()`, `build_retrieval_chain()`, `ingest_document()`.
- Variables describe their content: `query_embedding`, `relevant_chunks`, `formatted_context`.
- Chain variables end with `_chain`: `retrieval_chain`, `generation_chain`.
- Prompt templates end with `_prompt` or `_template`: `rag_prompt`, `system_template`.
- Standard abbreviations are acceptable: `db`, `config`, `req`, `res`, `doc`, `docs`.
- No single-letter variables except in list comprehensions and lambda expressions.
- Module names are singular nouns: `retrieval`, `ingestion`, not `retrievals`, `ingestions`.

**Bad naming examples:**
```python
# BAD
def get_stuff(q):  # vague function name, single-letter parameter
    d = db.search(q)  # single-letter variable
    return d

# GOOD
def retrieve_relevant_chunks(query: str) -> list[Document]:
    matching_documents = vector_store.similarity_search(query)
    return matching_documents
```

---

## 9. Imports

**Check:** Organized by ruff (stdlib -> third-party -> local). No unused imports.

**What to verify:**
- Imports are in three groups separated by blank lines: stdlib, third-party, local.
- No unused imports (ruff catches these).
- Absolute imports only (`from src.retrieval.chain import ...`, not `from .chain import ...`).
- Import from module's `__init__.py` for public API, not from internal files.
- No circular imports (A imports B, B imports A).
- No wildcard imports (`from module import *`).

Run `uv run ruff check .` to verify import organization automatically.

---

## 10. Documentation

**Check:** Public API functions have docstrings. Complex logic has inline comments explaining WHY, not WHAT.

**What to verify:**
- All public functions (not prefixed with `_`) have docstrings.
- Docstrings describe what the function does, its parameters, return value, and exceptions raised.
- Complex algorithms or non-obvious logic have inline comments explaining the reasoning.
- Comments explain WHY, not WHAT (the code already shows what).
- No commented-out code (delete it, version control remembers).
- No TODO comments without a linked issue or ticket.

```python
# GOOD — docstring + WHY comment
async def retrieve_relevant_chunks(
    query: str, top_k: int = 4
) -> list[Document]:
    """Retrieve the most relevant document chunks for a query.

    Args:
        query: The user's natural language question.
        top_k: Number of chunks to retrieve. Defaults to 4.

    Returns:
        List of Document objects ranked by relevance.

    Raises:
        RetrievalError: If the Qdrant collection does not exist.
    """
    # Use MMR (Maximal Marginal Relevance) instead of pure similarity
    # to reduce redundancy in retrieved chunks. Without MMR, top-k
    # results often contain near-duplicate passages from the same section.
    return await retriever.ainvoke(query)

# BAD — WHAT comment (redundant with code)
# Get documents from vector store
docs = vector_store.similarity_search(query)  # search for documents
```

---

## Review Workflow

When asked to review code, execute these steps:

1. **Read all changed files** completely. Do not skim.
2. **Walk through the 10-point checklist** above for each file.
3. **Categorize findings** by severity:
   - **Must fix:** Bugs, security issues, async correctness problems, missing error handling.
   - **Should fix:** Missing types, poor naming, missing tests, performance issues.
   - **Consider:** Documentation improvements, minor style preferences.
4. **Report findings** with specific file paths, line numbers, and code snippets.
5. **Suggest fixes** with corrected code for each finding.
6. **Verify** by running `uv run ruff check .` and `uv run pytest` if possible.
