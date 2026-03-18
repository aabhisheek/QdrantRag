---
name: perf-check
description: Identify and fix performance bottlenecks in the Python/FastAPI/LangChain/Qdrant RAG pipeline
---

# Performance Check Skill

Use this checklist to audit and optimize performance in the RAG Knowledge Assistant. Each item describes the problem, how to detect it, and how to fix it.

---

## 1. Embedding Batch Size

**Problem:** Processing documents one at a time through the embedding model. Each call has network overhead (for OpenAI) or GPU scheduling overhead (for Ollama). Ingesting 1000 documents with individual calls takes 10-50x longer than batched calls.

**Detection:** Look for loops calling `embeddings.embed_query()` or `embeddings.embed_documents([single_doc])` inside a for-loop.

```python
# BAD — one API call per document
for doc in documents:
    vector = embeddings.embed_documents([doc.page_content])
    # store vector...
```

**Fix:** Use `QdrantVectorStore.add_documents()` with a batch_size parameter, or batch embed calls manually.

```python
# GOOD — batch processing
vector_store.add_documents(documents, batch_size=200)

# Or manual batching for custom pipelines
BATCH_SIZE = 200
for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i : i + BATCH_SIZE]
    texts = [doc.page_content for doc in batch]
    vectors = embeddings.embed_documents(texts)  # single API call for batch
    # store vectors...
```

**Target:** 100-500 documents per batch. Larger batches use more memory. OpenAI has a token-per-minute limit that may require smaller batches.

---

## 2. Qdrant HNSW Tuning

**Problem:** Default HNSW parameters may not balance recall and speed for your dataset size and query patterns.

**Detection:** Check collection configuration for HNSW parameters. If using defaults, they may not be optimal.

```python
# Check current config
collection_info = qdrant_client.get_collection("documents")
print(collection_info.config.hnsw_config)
```

**Fix:** Configure HNSW parameters at collection creation time.

```python
from qdrant_client.models import HnswConfigDiff, VectorParams, Distance

qdrant_client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=1536,  # text-embedding-3-small dimension
        distance=Distance.COSINE,
    ),
    hnsw_config=HnswConfigDiff(
        m=16,              # Number of edges per node. Higher = better recall, more memory
        ef_construct=100,  # Search width during index build. Higher = better index, slower build
    ),
)

# At query time, increase ef for higher recall
results = qdrant_client.search(
    collection_name="documents",
    query_vector=query_vector,
    limit=4,
    search_params=SearchParams(hnsw_ef=128),  # Higher ef = better recall at query time
)
```

**Guidelines:**
- `m=16`: Good default. Increase to 32-64 for datasets over 1M vectors where recall matters more than memory.
- `ef_construct=100`: Good default. Increase to 200 for better index quality if build time is not critical.
- Query-time `hnsw_ef`: Should be >= `limit` (k). Set to 128-256 for high-recall use cases.

---

## 3. Async Event Loop Blocking

**Problem:** Calling synchronous, CPU-bound, or long-running I/O functions directly inside an `async def` FastAPI handler blocks the entire event loop. No other requests can be served while the blocking call completes.

**Detection:** Look for sync calls inside `async def` handlers:

```python
# BAD — blocks the event loop
@app.post("/ingest")
async def ingest(file: UploadFile):
    texts = [doc.page_content for doc in documents]
    vectors = embeddings.embed_documents(texts)  # SYNC call, blocks event loop
    qdrant_client.upsert(...)  # SYNC call, blocks event loop
```

**Fix:** Use `asyncio.to_thread()` for sync-only libraries, or use async methods when available.

```python
import asyncio

# GOOD — offload sync calls to thread pool
@app.post("/ingest")
async def ingest(file: UploadFile):
    texts = [doc.page_content for doc in documents]
    vectors = await asyncio.to_thread(embeddings.embed_documents, texts)
    await asyncio.to_thread(qdrant_client.upsert, collection_name="docs", points=points)

# BETTER — use async methods when available
@app.post("/query")
async def query(request: QueryRequest):
    # LangChain async methods don't block the event loop
    result = await chain.ainvoke(request.question)
    return QueryResponse(answer=result)
```

---

## 4. Python GIL and CPU-Bound Embedding

**Problem:** If using local sentence-transformers or Ollama embeddings with CPU inference, the Python GIL prevents true parallelism. All async handlers are blocked during CPU-bound embedding computation.

**Detection:** High latency on all endpoints during ingestion. CPU at 100% on a single core. Other requests queue up.

**Fix options:**
1. **Use OpenAI embeddings for production** — the work happens on OpenAI's servers (I/O-bound, not CPU-bound), so async works correctly.
2. **Offload to a separate process** for local embeddings:
   ```python
   from concurrent.futures import ProcessPoolExecutor

   executor = ProcessPoolExecutor(max_workers=2)

   async def embed_in_process(texts: list[str]) -> list[list[float]]:
       loop = asyncio.get_event_loop()
       return await loop.run_in_executor(executor, embeddings.embed_documents, texts)
   ```
3. **Use a dedicated embedding service** that runs in its own process/container.

---

## 5. Streaming LLM Responses

**Problem:** Without streaming, the user waits for the entire LLM generation to complete before seeing any response. For long answers, this can be 5-15 seconds of blank screen.

**Detection:** Look for `chain.ainvoke()` returning a complete string that is then sent as a JSON response.

```python
# BAD — user waits for full generation
@app.post("/query")
async def query(request: QueryRequest):
    result = await chain.ainvoke(request.question)  # waits 5-15 seconds
    return {"answer": result}
```

**Fix:** Use `astream()` with FastAPI's `StreamingResponse`.

```python
from fastapi.responses import StreamingResponse

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    async def generate():
        async for chunk in chain.astream(request.question):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")
```

**Impact:** Time-to-first-byte drops from 5-15 seconds to 200-500ms. Users see the answer being generated in real time.

---

## 6. Connection Pooling for QdrantClient

**Problem:** Creating a new `QdrantClient` instance per request wastes resources on TCP connection setup and teardown. Each new client opens a new connection to the Qdrant server.

**Detection:** Look for `QdrantClient()` instantiation inside route handlers or per-request functions.

```python
# BAD — new connection per request
@app.post("/query")
async def query(request: QueryRequest):
    client = QdrantClient(host="localhost", port=6333)  # new connection every time
    vector_store = QdrantVectorStore(client=client, ...)
    ...
```

**Fix:** Create a single client instance during FastAPI lifespan and share it via dependency injection or app state.

```python
from contextlib import asynccontextmanager
from qdrant_client import QdrantClient

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create shared client
    app.state.qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        prefer_grpc=True,  # gRPC is faster than REST for high-throughput
    )
    yield
    # Shutdown: close client
    app.state.qdrant_client.close()

app = FastAPI(lifespan=lifespan)

def get_qdrant_client(request: Request) -> QdrantClient:
    return request.app.state.qdrant_client
```

---

## 7. Caching Embedding Results

**Problem:** Repeated identical queries generate the same embedding vector every time. Each embedding call costs latency and (for OpenAI) money.

**Detection:** Check query logs for repeated questions. Common in development, testing, and production with popular queries.

**Fix:** Cache embeddings for repeated queries.

```python
from functools import lru_cache
import hashlib

# Simple in-memory cache for query embeddings
@lru_cache(maxsize=1000)
def cached_embed_query(query: str) -> tuple[float, ...]:
    """Cache embedding results for repeated queries."""
    vector = embeddings.embed_query(query)
    return tuple(vector)  # tuples are hashable, lists are not

# For production, use Redis:
# import redis
# r = redis.Redis()
# cache_key = f"emb:{hashlib.sha256(query.encode()).hexdigest()}"
# cached = r.get(cache_key)
# if cached: return json.loads(cached)
# vector = embeddings.embed_query(query)
# r.setex(cache_key, 3600, json.dumps(vector))
```

---

## 8. Document Chunking Configuration

**Problem:** Default or arbitrary chunk sizes lead to poor retrieval quality or wasted tokens. Chunks too small lose context. Chunks too large dilute relevance and waste LLM context window.

**Detection:** Check the `RecursiveCharacterTextSplitter` configuration. If chunk_size and chunk_overlap are not explicitly set, or are set without profiling, they may be suboptimal.

**Fix:** Start with a baseline and profile for your content type.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Baseline configuration — good starting point for most content
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,      # ~128 tokens for text-embedding-3-small
    chunk_overlap=50,    # ~10% overlap to preserve context at boundaries
    separators=["\n\n", "\n", ". ", " ", ""],  # split at natural boundaries first
)
```

**Tuning guidelines:**
- **Technical documentation:** chunk_size=512, chunk_overlap=50. Dense information, smaller chunks improve precision.
- **Narrative text (books, articles):** chunk_size=1024, chunk_overlap=100. Longer context needed for coherent passages.
- **Code:** chunk_size=1024, chunk_overlap=200. Functions and classes need more context.
- **FAQ/Q&A pairs:** chunk_size=256, chunk_overlap=0. Each Q&A is self-contained.

**Profiling approach:**
1. Ingest a representative sample with different chunk_size values (256, 512, 1024).
2. Run 20-50 representative queries against each configuration.
3. Measure retrieval precision (are the returned chunks relevant?) and answer quality.
4. Pick the chunk_size that gives the best retrieval precision for your content.

---

## Performance Audit Workflow

When asked to check performance, run through these steps in order:

1. **Read the codebase** — identify ingestion and query paths.
2. **Check each item** in this checklist against the actual code.
3. **Prioritize findings** by impact: event loop blocking > connection pooling > batching > HNSW tuning > caching > streaming > chunking.
4. **Report findings** with specific file paths and line numbers.
5. **Propose fixes** with code examples tailored to the project's actual implementation.
6. **Estimate impact** — quantify expected improvement (e.g., "batching embeddings should reduce ingestion time from ~60s to ~3s for 1000 documents").
