---
name: explain
description: Explain code, concepts, or architecture using a 5-layer framework from one-liner to edge cases
---

# Explanation Skill

Use this 5-layer framework to explain any code, concept, or architectural decision in the RAG Knowledge Assistant project. Start at the layer the user needs and go deeper on request. Each layer builds on the previous one.

---

## Layer 1 — One-Liner

State what the thing does in a single sentence. No jargon beyond what the audience already knows. This is the "elevator pitch" of explanations.

**Template:** `[Subject] [verb] [object] [purpose].`

**Examples:**
- "The retrieval chain fetches the most relevant document chunks from Qdrant and passes them to the LLM as context for answering the user's question."
- "The ingestion pipeline splits uploaded documents into chunks, generates embeddings, and stores them in the Qdrant vector database."
- "The `QdrantVectorStore` class wraps the qdrant-client to provide LangChain-compatible similarity search."
- "The `RecursiveCharacterTextSplitter` breaks documents into overlapping chunks that fit within the embedding model's token window."

---

## Layer 2 — Paragraph

Explain how it works in 3-5 sentences. Cover the inputs, the transformation, and the outputs. Mention the key dependencies.

**Template:**
1. What triggers it / what input it receives
2. What it does with that input (the main processing steps)
3. What it produces as output
4. What external systems it talks to
5. Where it lives in the codebase

**Example — Retrieval Chain:**
The retrieval chain receives a user question as a plain string. It first passes the question through the embedding model (OpenAI `text-embedding-3-small` or Ollama) to convert it into a vector, then performs a similarity search against the Qdrant collection to find the top-k most relevant document chunks. Those chunks are formatted and injected into a prompt template alongside the original question. The composed prompt is sent to the LLM (GPT-4o-mini or LLaMA 3.1 8B), which generates a natural-language answer. The chain returns the answer as a parsed string via LangChain's `StrOutputParser`.

---

## Layer 3 — Technical Deep-Dive

Walk through the implementation step by step. Reference specific classes, functions, and configuration values. Explain the chain composition using LCEL (LangChain Expression Language) pipe operators.

**Example — LCEL chain composition:**
```python
# The chain is built using LangChain's pipe operator (|)
# Each step transforms data and passes it to the next

chain = (
    # Step 1: RunnableParallel — runs context retrieval and question passthrough in parallel
    {
        "context": retriever | format_docs,  # retriever returns Documents, format_docs converts to string
        "question": RunnablePassthrough(),     # passes the input string through unchanged
    }
    # Step 2: ChatPromptTemplate — injects context and question into the prompt
    | ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n\n"
        "{context}\n\n"
        "Question: {question}"
    )
    # Step 3: LLM — sends the formatted prompt to the model
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Step 4: StrOutputParser — extracts the string content from the AIMessage
    | StrOutputParser()
)

# Invocation: chain.ainvoke("What is HNSW?") → "HNSW is a graph-based..."
```

**Qdrant query flow:**
1. `retriever.aget_relevant_documents(query)` is called.
2. LangChain's `QdrantVectorStore` calls `embeddings.aembed_query(query)` to get the query vector.
3. The vector is sent to Qdrant via `qdrant_client.search(collection_name, query_vector, limit=k)`.
4. Qdrant performs HNSW approximate nearest neighbor search across the indexed vectors.
5. Payload filtering (if configured) is applied to narrow results by metadata (e.g., `source`, `page`).
6. Qdrant returns `ScoredPoint` objects with scores and payloads.
7. LangChain converts them back to `Document` objects with `page_content` and `metadata`.

**FastAPI request flow:**
1. HTTP POST to `/query` with JSON body `{"question": "..."}`.
2. Pydantic `QueryRequest` model validates the input.
3. FastAPI dependency injection provides the configured chain.
4. `chain.ainvoke()` executes the full RAG pipeline asynchronously.
5. Response is returned as `QueryResponse` Pydantic model converted to JSON.

---

## Layer 4 — Analogy

Provide a real-world comparison that maps each component to something the audience already understands.

**RAG = Research Librarian**
- **User question** = patron walks up to the reference desk with a question
- **Embedding model** = the librarian's understanding of what the question is really about (semantic meaning, not just keywords)
- **Vector database (Qdrant)** = the library's catalog system, but instead of Dewey Decimal, it organizes books by meaning similarity
- **Retrieval (top-k search)** = the librarian goes to the shelves and pulls the 4 most relevant pages
- **Prompt template** = the librarian's mental framework: "Given these reference pages, answer this specific question"
- **LLM** = the librarian's ability to read the pages and synthesize a coherent answer
- **StrOutputParser** = the librarian writing the answer on a card and handing it to the patron

**HNSW Index = Airport Hub Network**
- Vectors are cities. Finding the nearest city by brute force means checking every city. HNSW builds a "flight network" with hub cities (higher layers) and local connections (lower layers). To find the nearest city, you start at a major hub, fly to progressively closer hubs, then walk to the exact destination. The `m` parameter controls how many flight routes each city has. The `ef_construct` parameter controls how many cities you consider when building routes.

**Chunking = Cutting a Book into Index Cards**
- You cannot put an entire book into the LLM's context window (too large, too expensive, too noisy). So you cut the book into overlapping index cards (chunks). Overlap ensures no sentence is split across cards without context. The chunk_size controls card size, chunk_overlap controls how much neighboring cards share.

---

## Layer 5 — Edge Cases

Identify what breaks, what is tricky, and what surprises people.

**Embedding dimension mismatch:**
If you switch from `text-embedding-3-small` (1536 dimensions) to a different model, the existing Qdrant collection has vectors of the old dimension. New queries will fail with a dimension mismatch error. Solution: recreate the collection and re-ingest all documents.

**Chunk boundary issues:**
A critical fact might be split across two chunks. The overlap mitigates this but does not eliminate it. If the answer requires information from the end of chunk N and the start of chunk N+1, and only one is retrieved, the LLM will give an incomplete answer. Increasing `chunk_overlap` or using semantic chunking (splitting at paragraph/section boundaries) helps.

**Prompt template variable errors:**
LangChain prompt templates use `{variable}` syntax. If your template references `{context}` but the chain provides `{documents}`, you get a `KeyError` at runtime — not at chain construction time. Always test chains end-to-end, not just individual components.

**Async/sync mismatch:**
LangChain has both sync (`invoke`) and async (`ainvoke`) methods. Calling `invoke` inside an `async def` FastAPI handler blocks the event loop. Always use `ainvoke`, `aget_relevant_documents`, and `aembed_query` in async contexts. If a component lacks an async method, wrap with `asyncio.to_thread()`.

**Qdrant collection not found:**
If the Qdrant collection does not exist when a query is made (e.g., first deployment before any ingestion), the qdrant-client raises `UnexpectedResponse`. Handle this with a clear error message telling the user to ingest documents first.

**Empty retrieval results:**
If no chunks meet the similarity threshold, the retriever returns an empty list. The LLM receives an empty context and may hallucinate. Always check for empty retrieval results and return a "no relevant documents found" response instead of passing empty context to the LLM.

**Rate limiting from OpenAI:**
OpenAI APIs have rate limits (TPM and RPM). Bulk ingestion with many embedding calls can hit these limits. Use exponential backoff, batch embedding calls, or switch to local Ollama embeddings for large ingestion jobs.

---

## Usage Guidelines

- **Default to Layer 2** unless the user specifies a depth.
- If the user says "explain briefly," use Layer 1.
- If the user says "explain in detail" or "how does this work exactly," use Layers 2-3.
- If the user says "explain like I'm new to this," use Layers 2 and 4.
- If the user says "what could go wrong," focus on Layer 5.
- Always ground explanations in the actual codebase. Read the relevant files first, then explain what IS there, not what should be there.
