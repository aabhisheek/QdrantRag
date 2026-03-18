---
name: refactor
description: Apply structured refactoring patterns to Python/FastAPI/LangChain/Qdrant code with safety checks
---

# Refactoring Skill

Apply one of the five refactoring types below. Every refactoring MUST follow the safety rules at the bottom.

---

## 1. Extract — Pull function/class from a large module

When a module does too much, extract a focused function or class into its own module.

**When to use:** A single file has multiple responsibilities, a function exceeds ~50 lines, or the same logic appears in more than one place.

**Example — extracting a retrieval chain builder:**

Before (monolithic `query_handler.py`):
```python
# src/api/query_handler.py — does ingestion, retrieval, generation, formatting
async def handle_query(request: QueryRequest) -> QueryResponse:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name="docs", embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_template("Answer based on: {context}\n\nQuestion: {question}")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return QueryResponse(answer=await chain.ainvoke(request.question))
```

After (extracted `src/generation/retrieval_chain.py`):
```python
# src/generation/retrieval_chain.py
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def build_retrieval_chain(retriever, llm):
    """Build a RAG retrieval chain from a retriever and LLM."""
    prompt = ChatPromptTemplate.from_template(
        "Answer based on: {context}\n\nQuestion: {question}"
    )
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
```

```python
# src/api/query_handler.py — now only orchestrates
from src.generation.retrieval_chain import build_retrieval_chain

async def handle_query(request: QueryRequest) -> QueryResponse:
    chain = build_retrieval_chain(retriever, llm)
    return QueryResponse(answer=await chain.ainvoke(request.question))
```

---

## 2. Inline — Remove unnecessary abstraction

When a wrapper adds no value, inline it to reduce indirection.

**When to use:** A function is called from exactly one place, does no transformation, and its name adds no clarity beyond the code it wraps.

**Example — inlining a single-use vector store helper:**

Before:
```python
# src/ingestion/helpers.py
def create_vector_store(client, collection_name, embeddings):
    """Create a Qdrant vector store."""
    return QdrantVectorStore.from_documents(
        documents=[], client=client, collection_name=collection_name, embedding=embeddings
    )
```

After (inlined at the single call site):
```python
# src/ingestion/pipeline.py
vector_store = QdrantVectorStore(
    client=qdrant_client, collection_name="docs", embedding=embeddings
)
```

Delete the helper file if it contained only that function.

---

## 3. Rename — Improve naming for clarity

When a name is vague, abbreviated, or misleading, rename it across the entire codebase.

**When to use:** A function name does not describe what it does, a variable name does not describe what it holds, or a module name does not describe its responsibility.

**Example:**
```python
# Before
def get_stuff(q: str) -> list[Document]:
    ...

# After
def retrieve_relevant_chunks(query: str) -> list[Document]:
    ...
```

Steps:
1. Read every file that imports or calls the old name.
2. Rename the definition.
3. Rename every usage (imports, calls, tests, type hints).
4. Run `uv run ruff check .` to catch any missed references.
5. Run `uv run pytest` to confirm nothing broke.

---

## 4. Restructure — Move modules to better locations

When a module lives in the wrong directory, move it and update all imports.

**When to use:** A shared utility lives inside a feature-specific directory, or a module's location does not match its responsibility.

**Example — moving shared chains from api/ to generation/:**

Before:
```
src/
  api/
    routes.py
    rag_chain.py       # <-- shared chain logic buried in api/
  generation/
    __init__.py
```

After:
```
src/
  api/
    routes.py           # imports from src.generation.rag_chain
  generation/
    __init__.py
    rag_chain.py         # <-- moved here
```

Steps:
1. Identify all imports of the module being moved.
2. Move the file.
3. Update every import statement.
4. Update `__init__.py` exports in both old and new packages.
5. Run `uv run ruff check .` and `uv run pytest`.

---

## 5. Generalize — Extract interface for provider switching

When you have two implementations that should be interchangeable, extract a common protocol.

**When to use:** You need to swap between providers (OpenAI/Ollama), storage backends, or processing strategies without changing calling code.

**Example — creating a BaseLLMProvider protocol:**

```python
# src/providers/base.py
from typing import Protocol

class BaseLLMProvider(Protocol):
    """Protocol for LLM provider implementations."""

    def get_llm(self) -> BaseChatModel:
        """Return a configured LangChain chat model."""
        ...

    def get_embeddings(self) -> Embeddings:
        """Return a configured LangChain embeddings model."""
        ...
```

```python
# src/providers/openai_provider.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class OpenAIProvider:
    def get_llm(self) -> ChatOpenAI:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def get_embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(model="text-embedding-3-small")
```

```python
# src/providers/ollama_provider.py
from langchain_ollama import ChatOllama, OllamaEmbeddings

class OllamaProvider:
    def get_llm(self) -> ChatOllama:
        return ChatOllama(model="llama3.1:8b")

    def get_embeddings(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(model="llama3.1:8b")
```

Calling code depends only on the protocol, never on a concrete provider.

---

## Safety Rules

These rules are non-negotiable for every refactoring operation:

1. **Read all callers before changing.** Use Grep to find every import, every call, every reference to the symbol being refactored. Do not rely on memory.
2. **Run `uv run pytest` after each step.** Not just at the end — after each discrete change. If tests fail, revert the last step before proceeding.
3. **Never break imports.** When moving or renaming a module, update every import statement in the codebase. Run `uv run ruff check .` to verify.
4. **Make atomic commits.** Each refactoring step gets its own commit with a clear message (e.g., `refactor: extract build_retrieval_chain into generation module`).
5. **Do not change behavior.** Refactoring changes structure, not behavior. If a test needs to change, that is a sign you are changing behavior, not just structure. Stop and reassess.
