#!/usr/bin/env python
"""
LlamaCloud Resume Matching - Setup and Configuration

Following the pipeline from:
https://github.com/run-llama/llamacloud-demo/blob/main/examples/resume_matching/

Pipeline:
1. PDF → LlamaParse → Markdown
2. Markdown → LLM → Structured Metadata (Pydantic)
3. Documents + Metadata → LlamaCloud Index
4. Query → LLM → Metadata Filters
5. Retrieve with filters → Semantic Search
6. Candidates → LLM → Analysis
"""

import os
import asyncio
from typing import List, Optional
from pathlib import Path

from pydantic import BaseModel, Field
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_parse import LlamaParse
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_cloud.types import CloudDocumentCreate


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for LlamaCloud Resume Matching"""

    # API Keys (set via environment variables)
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Models
    LLM_MODEL = "gpt-4o-mini"  # For metadata extraction and analysis
    EMBEDDING_MODEL = "text-embedding-ada-002"  # For vector embeddings

    # LlamaCloud
    PROJECT_NAME = "talent-matching"
    INDEX_NAME = "resumes"

    # Parsing
    PARSE_RESULT_TYPE = "markdown"

    # Retrieval
    TOP_K = 10  # Number of candidates to retrieve

    @classmethod
    def validate(cls):
        """Validate required API keys are set"""
        if not cls.LLAMA_CLOUD_API_KEY:
            raise ValueError(
                "LLAMA_CLOUD_API_KEY not set. Get it from: https://cloud.llamaindex.ai/"
            )
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not set. Get it from: https://platform.openai.com/"
            )


# ============================================================================
# PYDANTIC MODELS (Structured Output)
# ============================================================================

class ResumeMetadata(BaseModel):
    """
    Structured metadata extracted from resume using LLM

    This replaces manual pattern matching with LLM-based extraction
    """
    skills: List[str] = Field(
        description="List of technical and professional skills mentioned in resume"
    )
    domain: str = Field(
        description="Primary professional domain (e.g., 'Software Engineering', 'Data Science')"
    )
    education_countries: List[str] = Field(
        description="Countries where candidate received education"
    )
    years_of_experience: Optional[int] = Field(
        default=None,
        description="Total years of professional experience"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "skills": ["python", "django", "aws", "postgresql"],
                "domain": "Software Engineering",
                "education_countries": ["USA", "India"],
                "years_of_experience": 7
            }
        }


class QueryMetadata(BaseModel):
    """
    Metadata filters extracted from user query/job description using LLM

    This enables natural language queries to be converted to structured filters
    """
    skills: Optional[List[str]] = Field(
        default=None,
        description="Required or preferred skills"
    )
    domain: Optional[str] = Field(
        default=None,
        description="Target professional domain"
    )
    countries: Optional[List[str]] = Field(
        default=None,
        description="Preferred education countries"
    )
    min_years_experience: Optional[int] = Field(
        default=None,
        description="Minimum years of experience required"
    )


# ============================================================================
# GLOBAL SETUP
# ============================================================================

def setup_llama_index():
    """
    Initialize LlamaIndex with OpenAI models

    This sets up the global configuration for LLM and embeddings
    """
    Config.validate()

    # Set up LLM
    Settings.llm = OpenAI(
        model=Config.LLM_MODEL,
        api_key=Config.OPENAI_API_KEY,
        temperature=0  # Deterministic for metadata extraction
    )

    # Set up embeddings
    Settings.embed_model = OpenAIEmbedding(
        model=Config.EMBEDDING_MODEL,
        api_key=Config.OPENAI_API_KEY
    )

    print(f"✅ LlamaIndex configured:")
    print(f"   LLM: {Config.LLM_MODEL}")
    print(f"   Embeddings: {Config.EMBEDDING_MODEL}")


# ============================================================================
# STEP 1: PDF PARSING
# ============================================================================

def parse_resume_pdfs(pdf_paths: List[str]) -> List[dict]:
    """
    Parse PDF resumes into markdown using LlamaParse

    Args:
        pdf_paths: List of paths to PDF files

    Returns:
        List of documents with text and metadata
    """
    print(f"\n📄 Parsing {len(pdf_paths)} PDF resumes with LlamaParse...")

    parser = LlamaParse(
        api_key=Config.LLAMA_CLOUD_API_KEY,
        result_type=Config.PARSE_RESULT_TYPE,
        verbose=True
    )

    documents = parser.load_data(pdf_paths)

    print(f"✅ Parsed {len(documents)} documents")

    return [
        {
            "text": doc.text,
            "metadata": doc.metadata,
            "doc_id": doc.doc_id
        }
        for doc in documents
    ]


# ============================================================================
# STEP 2: METADATA EXTRACTION
# ============================================================================

async def extract_metadata(text: str) -> ResumeMetadata:
    """
    Extract structured metadata from resume text using LLM

    This uses OpenAI's structured output (function calling) to ensure
    the response conforms to the Pydantic model

    Args:
        text: Resume text (markdown from LlamaParse)

    Returns:
        ResumeMetadata object with extracted information
    """
    llm = Settings.llm

    prompt = f"""
    Extract structured information from this resume:

    {text}

    Return:
    - skills: List of all technical and professional skills
    - domain: Primary professional domain/field
    - education_countries: Countries where education was received
    - years_of_experience: Total years of work experience (estimate if not explicit)
    """

    metadata = await llm.astructured_predict(
        ResumeMetadata,
        prompt=prompt
    )

    return metadata


async def extract_metadata_batch(documents: List[dict]) -> List[ResumeMetadata]:
    """
    Extract metadata from multiple documents in parallel

    Args:
        documents: List of document dicts with 'text' key

    Returns:
        List of ResumeMetadata objects
    """
    print(f"\n🧠 Extracting metadata from {len(documents)} resumes using LLM...")

    tasks = [extract_metadata(doc["text"]) for doc in documents]
    metadatas = await asyncio.gather(*tasks)

    print(f"✅ Extracted metadata for {len(metadatas)} resumes")

    return metadatas


# ============================================================================
# STEP 3: LLAMACLOUD INDEXING
# ============================================================================

def create_llamacloud_index() -> LlamaCloudIndex:
    """
    Create or connect to LlamaCloud index

    The index handles:
    - Vector embeddings
    - Metadata storage
    - Semantic search
    - Filtering

    Returns:
        LlamaCloudIndex instance
    """
    print(f"\n☁️ Creating LlamaCloud index: {Config.INDEX_NAME}...")

    index = LlamaCloudIndex(
        name=Config.INDEX_NAME,
        project_name=Config.PROJECT_NAME,
        api_key=Config.LLAMA_CLOUD_API_KEY
    )

    print(f"✅ Connected to LlamaCloud index")

    return index


def upload_documents_to_cloud(
    index: LlamaCloudIndex,
    documents: List[dict],
    metadatas: List[ResumeMetadata]
):
    """
    Upload documents with metadata to LlamaCloud

    Args:
        index: LlamaCloudIndex instance
        documents: List of document dicts
        metadatas: List of ResumeMetadata objects
    """
    print(f"\n⬆️ Uploading {len(documents)} documents to LlamaCloud...")

    cloud_documents = [
        CloudDocumentCreate(
            text=doc["text"],
            metadata={
                "skills": meta.skills,
                "domain": meta.domain,
                "education_countries": meta.education_countries,
                "years_of_experience": meta.years_of_experience,
                "doc_id": doc["doc_id"]
            },
            id=doc["doc_id"]
        )
        for doc, meta in zip(documents, metadatas)
    ]

    index.insert_cloud_documents(cloud_documents)

    print(f"✅ Uploaded {len(cloud_documents)} documents")


# ============================================================================
# STEP 4: QUERY METADATA EXTRACTION
# ============================================================================

async def extract_query_metadata(query: str, all_skills: List[str]) -> QueryMetadata:
    """
    Extract metadata filters from natural language query

    This enables queries like:
    "Find Python developers with 5+ years experience"
    → {skills: ["python"], min_years_experience: 5}

    Args:
        query: Natural language query or job description
        all_skills: Global list of all skills in the database

    Returns:
        QueryMetadata with extracted filters
    """
    llm = Settings.llm

    prompt = f"""
    Analyze this job requirement or search query:

    "{query}"

    Extract filters for searching resumes:
    - skills: Required skills (must be from this list: {all_skills[:50]})
    - domain: Professional domain if specified
    - countries: Education countries if specified
    - min_years_experience: Minimum years if specified

    Only include filters that are explicitly mentioned or clearly implied.
    """

    metadata = await llm.astructured_predict(
        QueryMetadata,
        prompt=prompt
    )

    return metadata


# ============================================================================
# STEP 5: RETRIEVAL
# ============================================================================

def retrieve_candidates(
    index: LlamaCloudIndex,
    query: str,
    filters: Optional[QueryMetadata] = None,
    top_k: int = Config.TOP_K
):
    """
    Retrieve matching candidates using semantic search + metadata filters

    Args:
        index: LlamaCloudIndex instance
        query: Search query
        filters: Optional metadata filters
        top_k: Number of results to return

    Returns:
        List of candidate nodes
    """
    from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator

    print(f"\n🔍 Retrieving candidates for: '{query}'")

    # Build metadata filters
    metadata_filters = []
    if filters:
        if filters.skills:
            for skill in filters.skills:
                metadata_filters.append(
                    MetadataFilter(
                        key="skills",
                        value=skill,
                        operator=FilterOperator.CONTAINS
                    )
                )
        if filters.domain:
            metadata_filters.append(
                MetadataFilter(
                    key="domain",
                    value=filters.domain,
                    operator=FilterOperator.EQ
                )
            )
        if filters.min_years_experience:
            metadata_filters.append(
                MetadataFilter(
                    key="years_of_experience",
                    value=filters.min_years_experience,
                    operator=FilterOperator.GTE
                )
            )

    filters_obj = MetadataFilters(filters=metadata_filters) if metadata_filters else None

    # Retrieve
    retriever = index.as_retriever(
        similarity_top_k=top_k,
        filters=filters_obj
    )

    nodes = retriever.retrieve(query)

    print(f"✅ Retrieved {len(nodes)} candidates")

    return nodes


# ============================================================================
# STEP 6: ANALYSIS
# ============================================================================

async def analyze_candidate(candidate_text: str, job_description: str) -> str:
    """
    Generate LLM analysis of candidate fit for job

    Args:
        candidate_text: Resume text
        job_description: Job requirements

    Returns:
        Analysis text explaining fit
    """
    llm = Settings.llm

    prompt = f"""
    Job Requirements:
    {job_description}

    Candidate Resume:
    {candidate_text}

    Analyze this candidate's suitability:
    1. Skills match (which skills match, which are missing)
    2. Experience level (years and relevance)
    3. Overall fit (rating 1-10 and explanation)
    4. Strengths (top 3)
    5. Gaps or concerns (if any)

    Be specific and concise.
    """

    response = await llm.acomplete(prompt)

    return response.text


# ============================================================================
# MAIN DEMO FUNCTION
# ============================================================================

async def demo_resume_matching():
    """
    Complete demo of the LlamaCloud resume matching pipeline
    """
    print("="*80)
    print(" LlamaCloud Resume Matching Demo")
    print("="*80)

    # Step 0: Setup
    setup_llama_index()

    # Step 1: Parse PDFs (example)
    # documents = parse_resume_pdfs(["resume1.pdf", "resume2.pdf"])

    # For demo, use text instead
    documents = [
        {
            "text": "John Doe - Software Engineer with 7 years experience in Python, Django, AWS",
            "metadata": {},
            "doc_id": "doc1"
        }
    ]

    # Step 2: Extract metadata
    metadatas = await extract_metadata_batch(documents)

    # Step 3: Create index
    index = create_llamacloud_index()

    # Step 4: Upload
    upload_documents_to_cloud(index, documents, metadatas)

    # Step 5: Query
    query = "Find senior Python developers with AWS experience"
    all_skills = ["python", "django", "aws", "react", "javascript"]
    query_filters = await extract_query_metadata(query, all_skills)

    # Step 6: Retrieve
    candidates = retrieve_candidates(index, query, query_filters)

    # Step 7: Analyze
    if candidates:
        analysis = await analyze_candidate(
            candidates[0].text,
            "Senior Python Developer with 5+ years"
        )
        print(f"\n📊 Analysis:\n{analysis}")

    print("\n" + "="*80)
    print(" Demo Complete!")
    print("="*80)


if __name__ == "__main__":
    # Check environment
    if not os.getenv("LLAMA_CLOUD_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Missing API keys!")
        print("\nSet these environment variables:")
        print("  export LLAMA_CLOUD_API_KEY='llx-...'")
        print("  export OPENAI_API_KEY='sk-...'")
        print("\nGet keys from:")
        print("  LlamaCloud: https://cloud.llamaindex.ai/")
        print("  OpenAI: https://platform.openai.com/")
    else:
        # Run demo
        asyncio.run(demo_resume_matching())
