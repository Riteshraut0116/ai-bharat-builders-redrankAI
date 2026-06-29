"""
precompute.py — Offline embedding pipeline for RedrankAI.

Encodes the job description and all 100K candidate summaries once using
all-MiniLM-L6-v2 (or a user-supplied local model), saving three .npy files
that rank.py loads at runtime.

Usage:
    python precompute.py --candidates ./candidates.jsonl.gz \
                         --model ./model \
                         --out ./
"""

import argparse
import gzip
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JD text — hardcoded Senior AI Engineer @ Redrob AI
# ---------------------------------------------------------------------------
JD_TEXT = (
    "Senior AI Engineer — Redrob AI, India (Bangalore / Hyderabad / Pune / Noida preferred). "
    "5 to 9 years of experience at product companies. "
    "Must have: production embeddings and vector search deployed to real users, "
    "hybrid retrieval using FAISS QDRANT Pinecone OpenSearch Elasticsearch, "
    "semantic search re-ranking RAG pipelines, "
    "evaluation frameworks NDCG MRR MAP offline-to-online correlation, "
    "Python NLP LLM text embeddings sentence-transformers BGE E5 models, "
    "information retrieval ranking recommendation systems. "
    "Nice to have: LLM fine-tuning LoRA QLoRA PEFT, "
    "learning-to-rank XGBoost LightGBM, PyTorch transformers BERT, "
    "A/B testing feature engineering MLflow Docker Kubernetes Spark AWS GCP. "
    "Avoid: entire career at IT services firms TCS Infosys Wipro Cognizant Accenture, "
    "research-only background with zero production deployment."
)

BATCH_SIZE = 256


def _build_candidate_summary(c: dict) -> str:
    """Build a short text summary of the candidate for embedding."""
    profile = c.get("profile", {}) or {}
    skills = c.get("skills", []) or []
    history = c.get("career_history", []) or []
    education = c.get("education", []) or []

    headline = profile.get("headline") or ""
    summary_text = (profile.get("summary") or "")[:300]
    current_title = profile.get("current_title") or ""
    current_company = profile.get("current_company") or ""

    # Top 8 skills by endorsements
    sorted_skills = sorted(skills, key=lambda s: s.get("endorsements", 0) or 0, reverse=True)
    top_skills = sorted_skills[:8]
    skills_str = ", ".join(
        f"{s.get('name', '')} ({s.get('proficiency', '')})" for s in top_skills
    )

    # First 3 career history entries
    career_parts = []
    for job in history[:3]:
        title = job.get("title") or ""
        company = job.get("company") or ""
        duration = job.get("duration_months", 0) or 0
        career_parts.append(f"{title} at {company} ({duration}m)")
    career_str = "; ".join(career_parts)

    # Education
    edu_parts = []
    for edu in education[:2]:
        degree = edu.get("degree") or ""
        field = edu.get("field_of_study") or ""
        institution = edu.get("institution") or ""
        tier = edu.get("tier") or ""
        edu_parts.append(f"{degree} {field} {institution} {tier}".strip())
    edu_str = " | ".join(edu_parts)

    return (
        f"{headline}. {summary_text}. "
        f"Title: {current_title} at {current_company}. "
        f"Skills: {skills_str}. "
        f"Career: {career_str}. "
        f"Education: {edu_str}."
    )


def _open_candidates(path: Path):
    """Open candidates file (supports .jsonl and .jsonl.gz)."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def main():
    """Entry point: encode JD and all candidate summaries, save .npy files."""
    parser = argparse.ArgumentParser(description="RedrankAI offline precompute")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--model", required=True, help="Path to local SentenceTransformer model directory")
    parser.add_argument("--out", default="./", help="Output directory for .npy files (default: ./)")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    model_path = args.model
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not candidates_path.exists():
        log.error("Candidates file not found: %s", candidates_path)
        sys.exit(1)

    t0 = time.time()

    # Load model
    log.info("Loading model from %s", model_path)
    model = SentenceTransformer(model_path)

    # Encode JD
    log.info("Encoding job description ...")
    jd_vec = model.encode(JD_TEXT, normalize_embeddings=True)
    jd_path = out_dir / "jd_embedding.npy"
    np.save(jd_path, jd_vec)
    log.info("Saved %s (dim=%d)", jd_path, jd_vec.shape[0])

    # Stream candidates and build summaries
    log.info("Streaming candidates from %s ...", candidates_path)
    summaries = []
    candidate_ids = []
    parse_errors = 0

    with _open_candidates(candidates_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Line %d: JSON parse error — %s", line_num, exc)
                parse_errors += 1
                continue

            cid = c.get("candidate_id")
            if not cid:
                log.warning("Line %d: missing candidate_id, skipping", line_num)
                parse_errors += 1
                continue

            candidate_ids.append(cid)
            summaries.append(_build_candidate_summary(c))

            if line_num % 10_000 == 0:
                log.info("  Streamed %d candidates ...", line_num)

    log.info("Loaded %d candidates (%d parse errors)", len(candidate_ids), parse_errors)

    # Encode in batches
    log.info("Encoding %d candidate summaries (batch_size=%d) ...", len(summaries), BATCH_SIZE)
    embeddings = model.encode(
        summaries,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    ids_array = np.array(candidate_ids)

    emb_path = out_dir / "embeddings.npy"
    ids_path = out_dir / "candidate_ids.npy"
    np.save(emb_path, embeddings)
    np.save(ids_path, ids_array)

    elapsed = time.time() - t0
    log.info("Saved embeddings %s shape=%s", emb_path, embeddings.shape)
    log.info("Saved candidate_ids %s count=%d", ids_path, len(ids_array))
    log.info("Total time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
