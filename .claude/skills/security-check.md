---
name: security-check
description: Audit the RAG Knowledge Assistant for security vulnerabilities across prompt injection, SSRF, secrets, dependencies, and input validation
---

# Security Check Skill

Systematic security audit for the Python/FastAPI/LangChain/Qdrant RAG application. Work through each category below, checking the actual codebase for each vulnerability pattern.

---

## 1. Prompt Injection

**Threat:** User input is concatenated directly into prompt templates, allowing the user to override system instructions and extract sensitive information or change LLM behavior.

**Detection:** Grep for f-strings, string concatenation, or `.format()` used to build prompts.

```python
# BAD — user input directly in f-string, vulnerable to prompt injection
prompt = f"Answer this question: {user_input}\nContext: {context}"

# BAD — string concatenation
prompt = "Answer this: " + user_input + "\nContext: " + context

# BAD — .format() with user input
prompt = "Answer this: {question}".format(question=user_input)
```

**Fix:** Always use LangChain's `ChatPromptTemplate` with `{variable}` placeholders. LangChain handles proper escaping and template composition.

```python
# GOOD — LangChain template with variable placeholders
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer based only on the provided context."),
    ("human", "Context: {context}\n\nQuestion: {question}"),
])

# The chain handles variable injection safely
chain = prompt | llm | StrOutputParser()
result = await chain.ainvoke({"context": context, "question": user_input})
```

**Grep patterns to run:**
```
f".*{.*input.*}.*prompt
f".*{.*query.*}.*prompt
f".*{.*question.*}.*answer
.format(.*user
.format(.*input
.format(.*query
prompt.*\+.*input
prompt.*\+.*query
```

---

## 2. SSRF (Server-Side Request Forgery)

**Threat:** If the application loads documents from user-provided URLs, an attacker can make the server fetch internal resources (cloud metadata endpoints, internal services, private network hosts).

**Detection:** Look for URL-based document loading endpoints. Check if URLs are validated against an allowlist or if private IP ranges are blocked.

```python
# BAD — no URL validation
@app.post("/ingest/url")
async def ingest_from_url(url: str):
    loader = WebBaseLoader(url)  # fetches any URL, including internal ones
    docs = loader.load()
```

**Fix:** Validate URLs against an allowlist and block private IP ranges.

```python
import ipaddress
from urllib.parse import urlparse

BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # AWS metadata
    ipaddress.ip_network("fd00::/8"),  # IPv6 private
]

def validate_url(url: str) -> bool:
    """Validate that URL is not targeting internal resources."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    try:
        import socket
        resolved_ip = socket.gethostbyname(parsed.hostname)
        ip = ipaddress.ip_address(resolved_ip)
        for network in BLOCKED_NETWORKS:
            if ip in network:
                return False
    except (socket.gaierror, ValueError):
        return False
    return True
```

---

## 3. API Key Exposure

**Threat:** API keys (OpenAI, Qdrant, etc.) hardcoded in source code, logged in plaintext, or committed to version control.

**Detection:** Grep the entire codebase for key patterns.

**Grep patterns to run:**
```
OPENAI_API_KEY
api_key\s*=\s*["']
secret\s*=\s*["']
password\s*=\s*["']
sk-[a-zA-Z0-9]{20,}
token\s*=\s*["']
```

**Also check:**
- Log output for API keys (structlog processors should filter sensitive fields).
- `docker-compose.yml` for hardcoded environment variables.
- `.env` file is listed in `.gitignore`.
- No `.env` files in git history (`git log --all --full-history -- "*.env"`).

**Fix:** Use environment variables loaded via pydantic-settings or python-dotenv. Add structlog processor to redact sensitive fields.

```python
# Settings via pydantic-settings (reads from .env automatically)
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

# structlog processor to redact sensitive fields
def redact_sensitive(logger, method_name, event_dict):
    """Remove sensitive fields from log output."""
    sensitive_keys = {"api_key", "openai_api_key", "secret", "password", "token", "authorization"}
    for key in sensitive_keys:
        if key in event_dict:
            event_dict[key] = "***REDACTED***"
    return event_dict
```

---

## 4. Dependency Audit

**Threat:** Known CVEs in installed Python packages. Supply chain attacks through compromised packages.

**Detection:** Run `uv run pip-audit` to check all dependencies against the OSV database.

```bash
# Check for known vulnerabilities
uv run pip-audit

# Check specific package
uv run pip-audit --require-hashes

# Generate report
uv run pip-audit --format=json --output=audit-report.json
```

**Fix:** Update vulnerable packages. If a fix is not available, evaluate the risk and document the decision. Pin exact versions in `pyproject.toml` for reproducible builds.

**Also check:**
- Are dependencies pinned with exact versions or version ranges?
- Is `uv.lock` committed to version control?
- Are there any unnecessary dependencies that increase attack surface?

---

## 5. Input Validation

**Threat:** Unvalidated input leads to crashes, resource exhaustion, or injection attacks.

**Detection:** Check all API endpoints for Pydantic model validation. Look for raw `str` or `dict` parameters without validation.

```python
# BAD — no validation on question length, type, or content
@app.post("/query")
async def query(body: dict):
    question = body.get("question")
    ...

# BAD — no file size or type validation
@app.post("/ingest")
async def ingest(file: UploadFile):
    content = await file.read()  # could be a 10GB file
    ...
```

**Fix:** Use Pydantic models with validators for all API inputs. Enforce file size limits and extension allowlists.

```python
from pydantic import BaseModel, Field, field_validator

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question must not be blank")
        return v.strip()

class IngestRequest(BaseModel):
    max_file_size: int = 50 * 1024 * 1024  # 50 MB

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}

@app.post("/ingest")
async def ingest(file: UploadFile):
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type {ext} not allowed. Allowed: {ALLOWED_EXTENSIONS}")

    # Validate file size (read in chunks to avoid memory exhaustion)
    content = bytearray()
    max_size = 50 * 1024 * 1024  # 50 MB
    while chunk := await file.read(8192):
        content.extend(chunk)
        if len(content) > max_size:
            raise HTTPException(413, "File too large. Maximum size is 50 MB.")
```

---

## 6. Qdrant Metadata Filtering Injection

**Threat:** Although Qdrant does not use SQL, metadata filtering via user-controlled parameters could lead to unexpected query behavior if filters are constructed from raw user input.

**Detection:** Look for Qdrant filter construction using unvalidated user input.

```python
# BAD — user controls filter field and value directly
@app.post("/query")
async def query(request: QueryRequest):
    filter_condition = models.Filter(
        must=[models.FieldCondition(
            key=request.filter_field,    # user controls the field name
            match=models.MatchValue(value=request.filter_value),
        )]
    )
```

**Fix:** Whitelist allowed filter fields. Validate filter values against expected types.

```python
ALLOWED_FILTER_FIELDS = {"source", "doc_type", "page"}

def build_filter(field: str, value: str) -> models.Filter:
    if field not in ALLOWED_FILTER_FIELDS:
        raise ValueError(f"Filter field '{field}' not allowed. Allowed: {ALLOWED_FILTER_FIELDS}")
    return models.Filter(
        must=[models.FieldCondition(
            key=field,
            match=models.MatchValue(value=value),
        )]
    )
```

---

## 7. .env Safety

**Threat:** Secrets committed to version control, exposed in Docker images, or missing from .gitignore.

**Detection checklist:**
- [ ] `.env` is in `.gitignore`
- [ ] `.env.example` exists with placeholder values (no real secrets)
- [ ] `docker-compose.yml` uses `env_file: .env`, not hardcoded `environment:` values
- [ ] Docker build does not `COPY .env` into the image
- [ ] No `.env` files in git history

**Grep patterns:**
```
# Check .gitignore
grep ".env" .gitignore

# Check docker-compose for hardcoded secrets
grep -i "api_key\|secret\|password" docker-compose.yml

# Check Dockerfile for .env copy
grep "COPY.*\.env" Dockerfile

# Check git history for .env commits
git log --all --full-history -- "*.env"
```

---

## 8. Rate Limiting

**Threat:** Abuse of `/query` and `/ingest` endpoints via excessive requests. LLM calls are expensive. Qdrant writes can overwhelm the database.

**Detection:** Check if rate limiting middleware is configured. Look for slowapi, fastapi-limiter, or custom middleware.

```python
# Check for rate limiting imports and configuration
# Look for: slowapi, fastapi_limiter, RateLimitMiddleware
```

**Fix:** Add rate limiting with slowapi.

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/query")
@limiter.limit("20/minute")
async def query(request: Request, body: QueryRequest):
    ...

@app.post("/ingest")
@limiter.limit("5/minute")
async def ingest(request: Request, file: UploadFile):
    ...
```

---

## Security Audit Workflow

When asked to perform a security audit, execute these steps in order:

1. **Read the codebase** — identify all API endpoints, configuration files, and external integrations.
2. **Run each check** in this file against the actual code. Use Grep to search for vulnerable patterns.
3. **Run `uv run pip-audit`** to check dependencies.
4. **Categorize findings** by severity: Critical (secrets exposed, no input validation) > High (prompt injection, SSRF) > Medium (no rate limiting, missing .gitignore) > Low (logging improvements).
5. **Report findings** with specific file paths, line numbers, and code snippets showing the vulnerability.
6. **Propose fixes** with code examples tailored to the project.
7. **Verify fixes** after implementation by re-running the relevant checks.
