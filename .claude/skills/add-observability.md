---
name: add-observability
description: Add structured logging (structlog), metrics (prometheus-client), and tracing (OpenTelemetry) to the RAG Knowledge Assistant
---

# Add Observability Skill

Add the three pillars of observability to the RAG Knowledge Assistant: structured logging with structlog, metrics with prometheus-client, and distributed tracing with OpenTelemetry. All examples use Python and integrate with FastAPI/LangChain/Qdrant.

---

## Pillar 1: Structured Logging with structlog

### Setup

Configure structlog once at application startup, typically in `src/core/logging.py` or directly in `src/main.py`.

```python
# src/core/logging.py
import structlog
from src.core.config import settings


def setup_logging() -> None:
    """Configure structlog for the application."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _redact_sensitive_fields,
            structlog.dev.ConsoleRenderer()
            if settings.debug
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(settings.log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _redact_sensitive_fields(logger, method_name, event_dict):
    """Remove sensitive fields from log output."""
    sensitive_keys = {
        "api_key",
        "openai_api_key",
        "secret",
        "password",
        "token",
        "authorization",
    }
    for key in list(event_dict.keys()):
        if key.lower() in sensitive_keys:
            event_dict[key] = "***REDACTED***"
    return event_dict
```

### Usage in Application Code

```python
import structlog

logger = structlog.get_logger()

# In route handlers
@app.post("/query")
async def query(request: QueryRequest):
    logger.info("query_received", question_length=len(request.question))

    try:
        result = await chain.ainvoke(request.question)
        logger.info("query_completed", answer_length=len(result))
        return QueryResponse(answer=result)
    except RetrievalError as e:
        logger.error("retrieval_failed", error=str(e))
        raise

# In ingestion pipeline
async def ingest_document(file_path: str):
    logger.info("ingestion_started", file_path=file_path)
    chunks = splitter.split_documents(documents)
    logger.info("document_chunked", chunk_count=len(chunks), file_path=file_path)
    await vector_store.aadd_documents(chunks)
    logger.info("ingestion_completed", chunk_count=len(chunks), file_path=file_path)
```

### Request Context with Middleware

Bind request-scoped context (request ID, user agent) to all log entries within a request.

```python
# src/middleware/logging.py
import uuid
import structlog
from starlette.middleware.base import BaseHTTPMiddleware

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        logger = structlog.get_logger()
        logger.info("request_started")

        response = await call_next(request)

        logger.info("request_completed", status_code=response.status_code)
        response.headers["X-Request-ID"] = request_id
        return response
```

---

## Pillar 2: Metrics with prometheus-client

### Setup

Define metrics in a central module and expose them via a `/metrics` endpoint.

```python
# src/core/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info

# Query metrics
query_count = Counter(
    "rag_queries_total",
    "Total number of RAG queries processed",
    ["status"],  # labels: success, error
)
query_latency = Histogram(
    "rag_query_duration_seconds",
    "Time spent processing RAG queries",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

# Ingestion metrics
ingestion_count = Counter(
    "rag_documents_ingested_total",
    "Total number of documents ingested",
    ["doc_type"],  # labels: pdf, txt, md
)
ingestion_latency = Histogram(
    "rag_ingestion_duration_seconds",
    "Time spent ingesting documents",
)
chunks_created = Counter(
    "rag_chunks_created_total",
    "Total number of chunks created during ingestion",
)

# Retrieval metrics
retrieval_latency = Histogram(
    "rag_retrieval_duration_seconds",
    "Time spent on vector similarity search",
)
retrieval_result_count = Histogram(
    "rag_retrieval_results",
    "Number of chunks returned per retrieval",
    buckets=[0, 1, 2, 3, 4, 5, 10],
)

# System metrics
active_connections = Gauge(
    "rag_active_connections",
    "Number of active client connections",
)
qdrant_collection_size = Gauge(
    "rag_qdrant_collection_vectors",
    "Number of vectors in the Qdrant collection",
    ["collection"],
)

# Application info
app_info = Info(
    "rag_app",
    "RAG Knowledge Assistant application information",
)
```

### Expose Metrics Endpoint

```python
# In src/main.py
from prometheus_client import make_asgi_app

# Mount prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### Usage in Application Code

```python
import time
from src.core.metrics import query_count, query_latency, retrieval_latency

@app.post("/query")
async def query(request: QueryRequest):
    start_time = time.perf_counter()

    try:
        result = await chain.ainvoke(request.question)
        query_count.labels(status="success").inc()
        return QueryResponse(answer=result)
    except Exception:
        query_count.labels(status="error").inc()
        raise
    finally:
        duration = time.perf_counter() - start_time
        query_latency.observe(duration)
```

### Using Histogram as Context Manager

```python
# Cleaner syntax with context manager
with query_latency.time():
    result = await chain.ainvoke(request.question)
```

---

## Pillar 3: Distributed Tracing with OpenTelemetry

### Setup

Configure OpenTelemetry tracing at application startup. This integrates with FastAPI to automatically trace all HTTP requests.

```python
# src/core/tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from src.core.config import settings


def setup_tracing(app) -> None:
    """Configure OpenTelemetry tracing for the application."""
    if not settings.tracing_enabled:
        return

    resource = Resource.create(
        {
            "service.name": "rag-knowledge-assistant",
            "service.version": settings.app_version,
            "deployment.environment": settings.environment,
        }
    )

    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Auto-instrument FastAPI (traces all requests automatically)
    FastAPIInstrumentor.instrument_app(app)

    # Auto-instrument httpx (traces outgoing HTTP calls to OpenAI, etc.)
    HTTPXClientInstrumentor().instrument()


# Module-level tracer for manual spans
tracer = trace.get_tracer("rag-knowledge-assistant")
```

### Usage — Manual Spans for RAG Pipeline Steps

```python
from src.core.tracing import tracer

async def retrieve_and_generate(question: str) -> str:
    """Execute the full RAG pipeline with tracing."""

    with tracer.start_as_current_span("rag_pipeline") as span:
        span.set_attribute("question.length", len(question))

        # Trace retrieval step
        with tracer.start_as_current_span("retrieval") as retrieval_span:
            chunks = await retriever.ainvoke(question)
            retrieval_span.set_attribute("chunks.count", len(chunks))
            retrieval_span.set_attribute("chunks.sources", [
                c.metadata.get("source", "unknown") for c in chunks
            ])

        # Trace generation step
        with tracer.start_as_current_span("generation") as gen_span:
            context = format_docs(chunks)
            gen_span.set_attribute("context.length", len(context))

            answer = await chain.ainvoke({
                "context": context,
                "question": question,
            })
            gen_span.set_attribute("answer.length", len(answer))

        span.set_attribute("status", "success")
        return answer
```

### Trace Context for Ingestion

```python
async def ingest_document(file_path: str) -> int:
    """Ingest a document with tracing for each pipeline step."""

    with tracer.start_as_current_span("ingest_document") as span:
        span.set_attribute("file.path", file_path)

        with tracer.start_as_current_span("load_document"):
            documents = loader.load()

        with tracer.start_as_current_span("split_documents") as split_span:
            chunks = splitter.split_documents(documents)
            split_span.set_attribute("chunks.count", len(chunks))

        with tracer.start_as_current_span("store_embeddings") as store_span:
            await vector_store.aadd_documents(chunks)
            store_span.set_attribute("vectors.stored", len(chunks))

        span.set_attribute("total_chunks", len(chunks))
        return len(chunks)
```

---

## Integration in main.py

Wire all three pillars together at application startup.

```python
# src/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from prometheus_client import make_asgi_app

from src.core.logging import setup_logging
from src.core.tracing import setup_tracing
from src.core.metrics import app_info
from src.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging()
    app_info.info({
        "version": settings.app_version,
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
    })
    yield
    # Shutdown (cleanup if needed)


app = FastAPI(title="RAG Knowledge Assistant", lifespan=lifespan)

# Tracing (must be after app creation)
setup_tracing(app)

# Metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

---

## Docker Compose for Observability Stack

For local development, add Prometheus, Grafana, and Jaeger to docker-compose.

```yaml
# docker-compose.observability.yml
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"   # Jaeger UI
      - "4317:4317"     # OTLP gRPC
```

---

## Observability Checklist

When adding observability to a new feature or endpoint:

1. **Logging:** Add structlog calls at entry, exit, and error points. Include relevant context (IDs, counts, durations). Ensure no sensitive data is logged.
2. **Metrics:** Add counters for operations (success/error), histograms for latencies, gauges for current state. Choose labels carefully (low cardinality only).
3. **Tracing:** Add spans for each distinct step in the pipeline. Set attributes for debugging (input sizes, result counts, source files).
4. **Verify:** Check `/metrics` endpoint returns expected metrics. Check logs include request_id context. Check traces appear in Jaeger UI.
