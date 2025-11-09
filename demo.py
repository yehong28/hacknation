#!/usr/bin/env python
"""
AI-Powered Resume Matching Demo

This script demonstrates the complete LlamaCloud pipeline:
1. Load resumes from Resume.csv
2. Extract metadata using LLM
3. Upload to LlamaCloud
4. Search with natural language queries
5. Analyze top candidates

Usage:
    python demo.py
"""

import asyncio
import os
import sys
from collections import Counter
import pandas as pd
from llamacloud_setup import (
    setup_llama_index,
    extract_metadata_batch,
    create_llamacloud_index,
    upload_documents_to_cloud,
    retrieve_candidates,
    analyze_candidate,
    extract_query_metadata
)


# Demo configuration
DEMO_SIZE = 20  # Number of resumes to process for demo
DEMO_QUERIES = [
    "Find senior Python developers with Django and AWS experience",
    "Data scientists with machine learning and Python",
    "Java developer with Spring Boot and microservices",
    "Frontend developer with React and JavaScript",
    "DevOps engineer with Kubernetes and AWS"
]


def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")


def print_section(title):
    """Print a formatted section"""
    print(f"\n{'─'*80}")
    print(f"  {title}")
    print(f"{'─'*80}\n")


async def main():
    """Main demo function"""

    print_header("AI-Powered Resume Matching Demo")
    print("This demo uses:")
    print("  • LlamaCloud for semantic search and indexing")
    print("  • OpenAI GPT-4o-mini for metadata extraction and analysis")
    print("  • Natural language queries instead of structured filters")

    # Step 1: Initialize
    print_section("Step 1: Initialize LlamaIndex")
    try:
        setup_llama_index()
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease set your API keys:")
        print("  export LLAMA_CLOUD_API_KEY='llx-...'")
        print("  export OPENAI_API_KEY='sk-...'")
        print("\nGet keys from:")
        print("  • LlamaCloud: https://cloud.llamaindex.ai/")
        print("  • OpenAI: https://platform.openai.com/")
        sys.exit(1)

    # Step 2: Load data
    print_section("Step 2: Load Resume Data")

    if not os.path.exists("Resume.csv"):
        print("❌ Error: Resume.csv not found")
        sys.exit(1)

    df = pd.read_csv("Resume.csv")
    print(f"📊 Total resumes in dataset: {len(df):,}")
    print(f"\nCategories: {len(df['Category'].unique())} unique")
    print("\nTop 10 categories:")
    for cat, count in df['Category'].value_counts().head(10).items():
        print(f"  {cat:30s} : {count:3d} resumes")

    # Use subset for demo
    demo_df = df.head(DEMO_SIZE)
    print(f"\n📝 Using {len(demo_df)} resumes for demo (to stay within free tier)")

    # Convert to document format
    documents = []
    for idx, row in demo_df.iterrows():
        documents.append({
            "text": row["Resume_str"],
            "metadata": {
                "category": row["Category"],
                "original_id": str(row["ID"])
            },
            "doc_id": f"resume_{row['ID']}"
        })

    print(f"✅ Prepared {len(documents)} documents")

    # Step 3: Extract metadata
    print_section("Step 3: Extract Metadata with LLM")
    print(f"🧠 Extracting skills, domain, experience from {len(documents)} resumes...")
    print("   (This takes ~30-60 seconds)")

    metadatas = await extract_metadata_batch(documents)

    print(f"\n✅ Extracted metadata from {len(metadatas)} resumes")

    # Show sample metadata
    print("\n📊 Sample Extracted Metadata:\n")
    for i, (doc, meta) in enumerate(zip(documents[:3], metadatas[:3])):
        print(f"Resume {i+1}: {doc['metadata']['category']}")
        print(f"  Skills: {', '.join(meta.skills[:10])}{'...' if len(meta.skills) > 10 else ''}")
        print(f"  Domain: {meta.domain}")
        print(f"  Experience: {meta.years_of_experience} years")
        print()

    # Analyze skills
    all_skills = []
    for meta in metadatas:
        all_skills.extend(meta.skills)

    skill_counts = Counter(all_skills)
    print(f"📊 Top 15 Skills in Demo Dataset:\n")
    for skill, count in skill_counts.most_common(15):
        bar = "█" * count
        print(f"  {skill:25s} : {bar} ({count})")

    # Step 4: Create index and upload
    print_section("Step 4: Create LlamaCloud Index & Upload")

    index = create_llamacloud_index()
    upload_documents_to_cloud(index, documents, metadatas)

    print("\n✅ Resume database ready for queries!")

    # Step 5: Natural language search
    print_section("Step 5: Natural Language Search")

    print("Testing multiple queries...\n")

    all_unique_skills = list(skill_counts.keys())

    for i, query in enumerate(DEMO_QUERIES[:3], 1):  # First 3 queries
        print(f"\n{i}. Query: \"{query}\"")

        # Extract filters from query
        query_filters = await extract_query_metadata(query, all_unique_skills)

        # Search
        candidates = retrieve_candidates(index, query, filters=query_filters, top_k=3)

        if candidates:
            print(f"   Found {len(candidates)} matches:\n")
            for j, candidate in enumerate(candidates, 1):
                meta = candidate.metadata
                print(f"   {j}. Score: {candidate.score:.3f} | Domain: {meta.get('domain', 'N/A')}")
                print(f"      Skills: {', '.join(meta.get('skills', [])[:8])}")
                print(f"      Experience: {meta.get('years_of_experience', 'N/A')} years")
        else:
            print("   No matches found")

    # Step 6: Detailed analysis
    print_section("Step 6: AI-Powered Candidate Analysis")

    # Use first query for detailed analysis
    query = DEMO_QUERIES[0]
    job_description = """
    Senior Python Developer

    Requirements:
    - 5+ years of Python development experience
    - Expert in Django or Flask web frameworks
    - Experience with AWS cloud services (EC2, S3, Lambda)
    - Strong understanding of REST APIs and microservices
    - PostgreSQL or MySQL database experience
    - Experience with Docker and Kubernetes
    - Excellent problem-solving and communication skills
    """

    print(f"Query: \"{query}\"")
    print("\nJob Description:")
    print(job_description)

    # Get top candidate
    query_filters = await extract_query_metadata(query, all_unique_skills)
    candidates = retrieve_candidates(index, query, filters=query_filters, top_k=1)

    if candidates:
        top_candidate = candidates[0]

        print(f"\n🤖 Analyzing top candidate (Score: {top_candidate.score:.3f})...\n")

        analysis = await analyze_candidate(
            top_candidate.text,
            job_description
        )

        print("─"*80)
        print("AI ANALYSIS")
        print("─"*80)
        print(analysis)

    # Summary
    print_header("Demo Complete!")

    print("✅ What we demonstrated:")
    print("  1. Smart metadata extraction (LLM reads resumes)")
    print("  2. Natural language queries (no structured filters)")
    print("  3. Semantic search (understands context)")
    print("  4. AI-powered analysis (explains matches)")

    print(f"\n📊 Demo Stats:")
    print(f"  • Processed: {len(documents)} resumes")
    print(f"  • Extracted: {len(all_skills)} total skill mentions")
    print(f"  • Unique skills: {len(skill_counts)} different skills")
    print(f"  • Queries tested: {min(3, len(DEMO_QUERIES))}")

    print(f"\n💰 Estimated Cost:")
    print(f"  • Metadata extraction: ~$0.02")
    print(f"  • Queries: ~$0.03")
    print(f"  • Total: <$0.10")

    print("\n🚀 Next Steps:")
    print("  • Scale to more resumes (100-1000)")
    print("  • Try custom queries")
    print("  • Build UI (Streamlit, Flask, etc.)")
    print("  • Add PDF upload support")

    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
