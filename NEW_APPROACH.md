# New Approach - Following LlamaCloud Pipeline

## What Changed

**OLD APPROACH (What we built):**
- Custom hash-based vectorization
- Pattern matching for skills
- Custom matching algorithm
- FastAPI + SQLAlchemy
- Manual skill extraction

**NEW APPROACH (LlamaCloud/LlamaIndex):**
- LlamaParse for PDF → Markdown
- LLM-based metadata extraction (OpenAI)
- LlamaCloud for indexing + embeddings
- Metadata filtering for search
- Structured output with Pydantic
- LLM for candidate analysis

---

## Pipeline Overview

```
1. PDF Resume
   ↓
2. LlamaParse → Markdown
   ↓
3. LLM Metadata Extraction (Pydantic)
   - Skills: ["python", "django", ...]
   - Domain: "Software Engineering"
   - Education Countries: ["USA", "India"]
   ↓
4. Upload to LlamaCloud
   - Indexed with metadata
   - Vector embeddings created
   ↓
5. Query Processing
   - Natural language query
   - LLM extracts metadata filters
   ↓
6. Retrieval
   - Semantic search + Metadata filtering
   - Returns matching candidates
   ↓
7. Analysis
   - LLM analyzes fit
   - Generates explanation
```

---

## Key Components

### 1. LlamaParse (PDF → Markdown)
```python
from llama_parse import LlamaParse

parser = LlamaParse(result_type="markdown")
documents = parser.load_data("resume.pdf")
```

### 2. Metadata Extraction (Pydantic + LLM)
```python
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI

class ResumeMetadata(BaseModel):
    skills: list[str]
    domain: str
    education_countries: list[str]

llm = OpenAI(model="gpt-4o-mini")
metadata = llm.structured_predict(
    ResumeMetadata,
    prompt=f"Extract metadata from: {resume_text}"
)
```

### 3. LlamaCloud Indexing
```python
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex

index = LlamaCloudIndex(
    name="resumes",
    project_name="talent-matching",
    api_key=os.environ["LLAMA_CLOUD_API_KEY"]
)

# Upload with metadata
index.insert_documents(documents_with_metadata)
```

### 4. Query with Metadata Filtering
```python
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

# Extract metadata from query
query_metadata = get_query_metadata("Find Python developers with 5+ years")

# Build filters
filters = MetadataFilters(filters=[
    MetadataFilter(key="skills", value="python", operator="contains"),
    MetadataFilter(key="domain", value="Software Engineering")
])

# Retrieve
retriever = index.as_retriever(filters=filters)
candidates = retriever.retrieve(query)
```

### 5. LLM Analysis
```python
analysis_prompt = f"""
Job Requirements: {job_description}
Candidate: {candidate_text}

Analyze:
1. Skills match
2. Experience level
3. Overall fit
4. Strengths and gaps
"""

analysis = llm.complete(analysis_prompt)
```

---

## Required Setup

### Environment Variables
```bash
export LLAMA_CLOUD_API_KEY="llx-..."
export OPENAI_API_KEY="sk-..."
```

### Dependencies
```txt
llama-index
llama-index-core
llama-parse
llama-cloud
llama-index-llms-openai
llama-index-embeddings-openai
llama-index-indices-managed-llama-cloud
pydantic
nest-asyncio
```

---

## Advantages of This Approach

### 1. **Better Accuracy**
- LLM understands context, not just keywords
- Structured extraction with Pydantic validation
- Semantic search vs simple keyword matching

### 2. **Less Code**
- No manual skill extraction logic
- No custom vectorization
- No database setup
- LlamaCloud handles indexing

### 3. **Scalability**
- Cloud-based indexing
- Async processing
- Production-ready infrastructure

### 4. **Flexibility**
- Natural language queries
- Metadata filtering
- Easy to extend metadata schema

---

## Implementation Plan

### Step 1: Environment Setup
```bash
pip install llama-index llama-parse llama-cloud
export LLAMA_CLOUD_API_KEY="..."
export OPENAI_API_KEY="..."
```

### Step 2: Parse Resumes
```python
from llama_parse import LlamaParse

parser = LlamaParse(result_type="markdown")
docs = parser.load_data(["resume1.pdf", "resume2.pdf"])
```

### Step 3: Extract Metadata
```python
async def get_metadata(text: str) -> ResumeMetadata:
    return await llm.astructured_predict(
        ResumeMetadata,
        prompt=f"Extract: {text}"
    )
```

### Step 4: Upload to LlamaCloud
```python
from llama_cloud import CloudDocumentCreate

cloud_docs = [
    CloudDocumentCreate(
        text=doc.text,
        metadata=metadata.dict(),
        id=doc.doc_id
    )
    for doc, metadata in zip(docs, metadatas)
]

index.insert_cloud_documents(cloud_docs)
```

### Step 5: Query & Retrieve
```python
query = "Find senior Python developers"
filters = get_query_metadata(query)
candidates = index.as_retriever(filters=filters).retrieve(query)
```

### Step 6: Analyze
```python
for candidate in candidates:
    analysis = llm.complete(
        f"Analyze fit: {candidate.text} for job: {job_desc}"
    )
    print(analysis)
```

---

## Demo Flow

### Old Demo (Custom System):
1. Upload PDF → Custom parser
2. Pattern match skills → Manual extraction
3. Hash vectorize → Custom similarity
4. Match → Custom algorithm
5. Show results

### New Demo (LlamaCloud):
1. Upload PDF → **LlamaParse** (better parsing)
2. Extract metadata → **LLM** (smarter extraction)
3. Index → **LlamaCloud** (managed service)
4. Query in plain English → **LLM** understands intent
5. Retrieve → **Semantic search + metadata filters**
6. Analyze → **LLM** explains fit

---

## Trade-offs

### Pros ✅
- Much smarter (LLM-powered)
- Less code to maintain
- Better parsing (LlamaParse)
- Metadata filtering
- Cloud-based (no database setup)
- Production-ready infrastructure

### Cons ⚠️
- Requires API keys (LlamaCloud + OpenAI)
- Costs money (API calls)
- Network dependency
- Less control over algorithms

---

## Cost Considerations

### LlamaCloud
- Free tier: 1,000 pages/month
- Paid: $0.003 per page

### OpenAI
- GPT-4o-mini: $0.15 per 1M input tokens
- text-embedding-ada-002: $0.10 per 1M tokens

**For demo with 100 resumes:**
- Parsing: ~$0.30 (100 pages)
- Metadata extraction: ~$0.05 (100 resumes)
- Embeddings: ~$0.01
- Queries: ~$0.01 per query
- **Total: ~$0.40 for demo**

---

## Next Steps

1. Get API keys (LlamaCloud + OpenAI)
2. Install new dependencies
3. Rewrite using LlamaIndex pipeline
4. Create Jupyter notebook for demo
5. Test with Resume.csv data

---

**This is the modern, LLM-powered approach vs our custom system.**
