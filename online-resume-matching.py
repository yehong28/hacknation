from __future__ import annotations

import re
import zlib
from collections import Counter
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RESUME_CSV = BASE_DIR / "archive" / "Resume" / "Resume.csv"
HASH_DIM = 4096
SAMPLES_PER_CATEGORY = 25
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "with", "would", "you", "your", "yours", "yourself",
    "yourselves"
}
TOKEN_PATTERN = re.compile(r"[a-z]+")


def preprocess_text(text: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall((text or "").lower())
    return [tok for tok in tokens if tok not in STOP_WORDS and len(tok) > 1]


def vectorize_tokens(tokens: list[str], dim: int = HASH_DIM) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    if not tokens:
        return vector
    counts = Counter(tokens)
    total = float(sum(counts.values()))
    for token, count in counts.items():
        idx = zlib.crc32(token.encode("utf-8")) % dim
        vector[idx] += count / total
    norm = np.linalg.norm(vector)
    return vector if norm == 0.0 else vector / norm


def load_resume_data() -> pd.DataFrame:
    resume_df = pd.read_csv(RESUME_CSV)
    required_cols = {"ID", "Category", "Resume_str"}
    missing = required_cols - set(resume_df.columns)
    if missing:
        raise ValueError(f"Missing columns in resume data: {missing}")
    resume_df = resume_df[["ID", "Category", "Resume_str"]].rename(
        columns={"Resume_str": "text"}
    )
    resume_df["text"] = resume_df["text"].fillna("")
    resume_df["tokens"] = resume_df["text"].apply(preprocess_text)
    resume_df["vector"] = resume_df["tokens"].apply(vectorize_tokens)
    return resume_df


def build_job_profiles(resume_df: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    category_counts = (
        resume_df.groupby("Category")
        .size()
        .sort_values(ascending=False)
    )
    rows = []
    for position, category in enumerate(category_counts.index[:limit]):
        category_rows = resume_df[resume_df["Category"] == category].head(
            SAMPLES_PER_CATEGORY
        )
        token_lists = category_rows["tokens"].tolist()
        combined_tokens = list(chain.from_iterable(token_lists))
        if not combined_tokens:
            combined_tokens = preprocess_text(" ".join(category_rows["text"]))
        rows.append(
            {
                "jobId": position,
                "position_title": category,
                "job_description": " ".join(category_rows["text"]),
                "vector": vectorize_tokens(combined_tokens),
            }
        )
    return pd.DataFrame(rows)


def match_resumes(job_profiles: pd.DataFrame, resumes: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    if job_profiles.empty or resumes.empty:
        return pd.DataFrame(
            columns=["jobId", "resumeId", "similarity", "domainResume", "domainDesc"]
        )
    resume_matrix = np.vstack(resumes["vector"].to_list())
    results = []
    for _, job_row in job_profiles.iterrows():
        job_vec = job_row["vector"]
        similarities = resume_matrix @ job_vec
        top_indices = np.argsort(similaries)[::-1][:top_k]
        for idx in top_indices:
            resume_row = resumes.iloc[idx]
            results.append(
                {
                    "jobId": job_row["jobId"],
                    "resumeId": int(resume_row["ID"]),
                    "similarity": float(similarities[idx]),
                    "domainResume": resume_row["Category"],
                    "domainDesc": job_row["position_title"],
                }
            )
    return pd.DataFrame(results)


def print_top_matching_resumes(result_df: pd.DataFrame) -> None:
    if result_df.empty:
        print("No matches were generated.")
        return
    for job_id, group in result_df.groupby("jobId"):
        group_sorted = group.sort_values(by="similarity", ascending=False)
        title = group_sorted["domainDesc"].iloc[0]
        print(f"\nJob ID: {job_id} ({title})")
        print("Cosine Similarity | Resume ID | Resume Domain")
        for _, row in group_sorted.iterrows():
            print(
                f"{row['similarity']:.4f} | {row['resumeId']} | {row['domainResume']}"
            )


def main() -> None:
    resumes = load_resume_data()
    job_profiles = build_job_profiles(resumes)
    result_df = match_resumes(job_profiles, resumes)
    print_top_matching_resumes(result_df)


if __name__ == "__main__":
    main()
