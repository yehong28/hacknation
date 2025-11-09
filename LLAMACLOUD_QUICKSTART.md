# LlamaCloud Resume Matching - Quick Start

## What This Does

**Modern LLM-powered resume matching** using LlamaIndex and LlamaCloud.

Instead of manual pattern matching, we use:
- **LlamaParse** for smart PDF parsing
- **OpenAI LLM** for intelligent metadata extraction
- **LlamaCloud** for semantic search and indexing
- **Natural language queries** instead of structured filters

---

## Setup (5 Minutes)

### Step 1: Get API Keys

**LlamaCloud:**
1. Go to https://cloud.llamaindex.ai/
2. Sign up (free tier: 1,000 pages/month)
3. Get your API key: `llx-...`

**OpenAI:**
1. Go to https://platform.openai.com/
2. Create account and add payment method
3. Get your API key: `sk-...`

### Step 2: Install Dependencies

```bash
pip install -r requirements_llamacloud.txt
```

### Step 3: Set Environment Variables

```bash
export LLAMA_CLOUD_API_KEY="llx-..."
export OPENAI_API_KEY="sk-..."
```

Or create `.env` file:
```
LLAMA_CLOUD_API_KEY=llx-...
OPENAI_API_KEY=sk-...
```

---

## Usage

### Basic Example

```python
import asyncio
from llamacloud_setup import (
    setup_llama_index,
    parse_resume_pdfs,
    extract_metadata_batch,
    create_llamacloud_index,
    upload_documents_to_cloud,
    retrieve_candidates,
    analyze_candidate
)

async def main():
    # 1. Setup
    setup_llama_index()

    # 2. Parse PDFs
    documents = parse_resume_pdfs(["resume1.pdf", "resume2.pdf"])

    # 3. Extract metadata using LLM
    metadatas = await extract_metadata_batch(documents)

    # 4. Create LlamaCloud index
    index = create_llamacloud_index()

    # 5. Upload documents
    upload_documents_to_cloud(index, documents, metadatas)

    # 6. Search with natural language
    candidates = retrieve_candidates(
        index,
        query="Find Python developers with 5+ years experience"
    )

    # 7. Analyze top candidate
    if candidates:
        analysis = await analyze_candidate(
            candidates[0].text,
            "Senior Python Developer needed"
        )
        print(analysis)

asyncio.run(main())
```

### With Resume.csv

```python
import pandas as pd
from llamacloud_setup import *

async def process_resume_csv():
    # Load Resume.csv
    df = pd.read_csv("Resume.csv")

    # Convert to documents
    documents = []
    for idx, row in df.iterrows():
        documents.append({
            "text": row["Resume_str"],
            "metadata": {"category": row["Category"]},
            "doc_id": str(row["ID"])
        })

    # Extract metadata with LLM
    metadatas = await extract_metadata_batch(documents[:100])  # First 100

    # Upload to LlamaCloud
    index = create_llamacloud_index()
    upload_documents_to_cloud(index, documents[:100], metadatas)

    # Query
    candidates = retrieve_candidates(
        index,
        query="Find senior software engineers with Python and AWS"
    )

    return candidates

asyncio.run(process_resume_csv())
```

---

## Pipeline Explanation

### 1. PDF Parsing (LlamaParse)
```python
parser = LlamaParse(result_type="markdown")
documents = parser.load_data(["resume.pdf"])
```
**What it does:** Converts PDF → structured markdown (better than PyPDF)

### 2. Metadata Extraction (LLM + Pydantic)
```python
class ResumeMetadata(BaseModel):
    skills: List[str]
    domain: str
    education_countries: List[str]
    years_of_experience: int

metadata = llm.structured_predict(ResumeMetadata, prompt=text)
```
**What it does:** LLM extracts structured data, validated by Pydantic

### 3. Cloud Indexing
```python
index = LlamaCloudIndex(name="resumes")
index.insert_cloud_documents(documents_with_metadata)
```
**What it does:** Uploads to LlamaCloud, creates embeddings, enables search

### 4. Natural Language Query
```python
candidates = retrieve_candidates(
    index,
    query="Find Python developers with 5+ years"
)
```
**What it does:** Semantic search + metadata filtering

### 5. LLM Analysis
```python
analysis = llm.complete(f"""
    Job: {job_desc}
    Candidate: {resume}
    Analyze fit...
""")
```
**What it does:** Explains why candidate matches

---

## Advantages Over Custom System

| Feature | Custom (what we built) | LlamaCloud |
|---------|----------------------|------------|
| PDF Parsing | PyPDF (basic) | LlamaParse (smart) |
| Skill Extraction | Pattern matching | LLM-based |
| Search | Hash vectors | Semantic + filters |
| Queries | Structured filters | Natural language |
| Code | 6,500 lines | ~400 lines |
| Infrastructure | Self-hosted DB | Managed cloud |
| Accuracy | Keyword matching | Context-aware LLM |

---

## Cost Estimate

**For 100 resumes demo:**
- LlamaParse: ~$0.30 (100 pages)
- Metadata extraction: ~$0.05 (100 LLM calls)
- Embeddings: ~$0.01
- Queries: ~$0.01 each
- **Total: ~$0.40** for full demo

**Free tier includes:**
- LlamaCloud: 1,000 pages/month free
- OpenAI: $5 credit for new accounts

---

## Demo Script (5 Minutes)

### Minute 1: Upload
```python
# Upload a resume
documents = parse_resume_pdfs(["sarah_chen_resume.pdf"])
metadatas = await extract_metadata_batch(documents)
print(f"Extracted skills: {metadatas[0].skills}")
```
**Show:** LLM automatically extracted 15+ skills

### Minute 2-3: Query
```python
# Natural language search
candidates = retrieve_candidates(
    index,
    "Find senior Python developers with Django and AWS experience"
)
print(f"Found {len(candidates)} matches")
```
**Show:** Search in plain English, get ranked results in <2 seconds

### Minute 4: Analyze
```python
# LLM explains the match
analysis = await analyze_candidate(
    candidates[0].text,
    "Senior Python Developer - Django, AWS, 5+ years"
)
print(analysis)
```
**Show:** Detailed explanation of why candidate matches

### Minute 5: Value
- **Smarter:** LLM understands context, not just keywords
- **Faster:** Managed infrastructure, no DB setup
- **Scalable:** Cloud-based, handles millions of resumes
- **Natural:** Search in plain English

---

## Files You Need

### Must Have:
- `llamacloud_setup.py` - Complete implementation ✅
- `requirements_llamacloud.txt` - Dependencies ✅
- `NEW_APPROACH.md` - Architecture explanation ✅
- `LLAMACLOUD_QUICKSTART.md` - This guide ✅

### Data:
- `Resume.csv` - Your existing dataset ✅
- Or any PDF resumes

### Optional:
- `resume_matching_notebook.ipynb` - Jupyter notebook for demo
- `.env` - API keys (don't commit!)

---

## Troubleshooting

### "LLAMA_CLOUD_API_KEY not set"
```bash
export LLAMA_CLOUD_API_KEY="llx-..."
# Or add to .env file
```

### "OpenAI API key not set"
```bash
export OPENAI_API_KEY="sk-..."
```

### "Module not found"
```bash
pip install -r requirements_llamacloud.txt
```

### "Rate limit exceeded"
- Use free tier carefully (1,000 pages/month)
- Add delays between API calls
- Start with small dataset (10-20 resumes)

---

## Next Steps

1. **Get API keys** (5 min)
2. **Install dependencies** (2 min)
3. **Run example** (`python llamacloud_setup.py`)
4. **Process Resume.csv** (Use the code above)
5. **Demo it!** (Follow 5-minute script)

---

**This is the modern, LLM-powered approach. Much smarter than keyword matching!** 🚀
