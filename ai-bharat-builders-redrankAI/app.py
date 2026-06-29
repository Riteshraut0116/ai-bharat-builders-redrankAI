"""
app.py — Streamlit sandbox for RedrankAI.

Allows uploading a JSON array of up to 100 candidate objects, runs the full
scoring pipeline locally (loads model from ./model if present, otherwise
downloads all-MiniLM-L6-v2), and displays ranked results with a download button.

Run with:
    streamlit run app.py
"""

import io
import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RedrankAI — Candidate Ranker",
    page_icon="🎯",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("ℹ️ About RedrankAI")
    st.markdown(
        """
        **Approach:** Two-phase pipeline combining semantic embeddings
        (all-MiniLM-L6-v2) with a 5-component rule-based scorer.

        **Scoring Formula:**
        ```
        final = 0.60 × semantic
              + 0.40 × rule
        ```
        **Rule Components:**
        | Component | Weight |
        |-----------|--------|
        | Skills    | 35 %   |
        | Career    | 25 %   |
        | Experience| 15 %   |
        | Location  | 10 %   |
        | Behavioral| 15 %   |

        **Tech Stack:**
        - sentence-transformers 2.7.0
        - all-MiniLM-L6-v2 (80 MB, CPU-only)
        - numpy, pandas, streamlit
        - Pure Python rule engine

        **Honeypot detection** filters fabricated profiles before scoring.
        """
    )

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("🎯 RedrankAI — Candidate Ranker")
st.caption(
    "Upload a JSON array of up to 100 candidates, run the full RedrankAI pipeline, "
    "and download the ranked CSV — all offline, no GPU, no API calls."
)

uploaded_file = st.file_uploader(
    "Upload candidates JSON (array of candidate objects, max 100)",
    type=["json"],
    help="The file must be a JSON array where each element matches candidate_schema.json",
)

run_btn = st.button("🚀 Run Ranker", type="primary", disabled=(uploaded_file is None))

if run_btn and uploaded_file is not None:
    # -----------------------------------------------------------------------
    # Load candidates
    # -----------------------------------------------------------------------
    try:
        raw_bytes = uploaded_file.read()
        candidates = json.loads(raw_bytes)
        if not isinstance(candidates, list):
            st.error("❌ Uploaded file must be a JSON **array** of candidate objects.")
            st.stop()
        if len(candidates) > 100:
            st.warning(f"⚠️ Only the first 100 of {len(candidates)} candidates will be ranked.")
            candidates = candidates[:100]
    except json.JSONDecodeError as exc:
        st.error(f"❌ JSON parse error: {exc}")
        st.stop()

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    local_model_dir = Path("./model")
    model_name = str(local_model_dir) if local_model_dir.exists() else "all-MiniLM-L6-v2"

    with st.spinner(f"Loading model from `{model_name}` …"):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
        except Exception as exc:
            st.error(f"❌ Failed to load model: {exc}")
            st.stop()

    # -----------------------------------------------------------------------
    # Import scoring modules
    # -----------------------------------------------------------------------
    try:
        from honeypot import is_honeypot
        from scorer import (
            behavioral_score,
            career_score,
            experience_score,
            location_score,
            skill_score,
        )
    except ImportError as exc:
        st.error(f"❌ Could not import scorer modules: {exc}. Make sure scorer.py and honeypot.py are in the same directory.")
        st.stop()

    # JD text (same as precompute.py)
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
        "A/B testing feature engineering MLflow Docker Kubernetes Spark AWS GCP."
    )

    # -----------------------------------------------------------------------
    # Encode JD + candidates
    # -----------------------------------------------------------------------
    import numpy as np

    def _build_summary(c: dict) -> str:
        profile = c.get("profile", {}) or {}
        skills = sorted(c.get("skills", []) or [], key=lambda s: s.get("endorsements", 0) or 0, reverse=True)
        history = c.get("career_history", []) or []
        headline = profile.get("headline") or ""
        summary_snip = (profile.get("summary") or "")[:300]
        top_skills_str = ", ".join(f"{s.get('name','')} ({s.get('proficiency','')})" for s in skills[:8])
        career_str = "; ".join(
            f"{j.get('title','')} at {j.get('company','')} ({j.get('duration_months',0)}m)"
            for j in history[:3]
        )
        return f"{headline}. {summary_snip}. Title: {profile.get('current_title','')} at {profile.get('current_company','')}. Skills: {top_skills_str}. Career: {career_str}."

    with st.spinner("Encoding candidates …"):
        try:
            jd_vec = model.encode(JD_TEXT, normalize_embeddings=True)
            summaries = [_build_summary(c) for c in candidates]
            embs = model.encode(summaries, normalize_embeddings=True, show_progress_bar=False)
            sims = embs @ jd_vec
        except Exception as exc:
            st.error(f"❌ Encoding failed: {exc}")
            st.stop()

    # -----------------------------------------------------------------------
    # Score each candidate
    # -----------------------------------------------------------------------
    W_SEMANTIC, W_RULE = 0.60, 0.40
    W_SKILL, W_CAREER, W_EXP, W_LOC, W_BEH = 0.35, 0.25, 0.15, 0.10, 0.15

    results = []
    honeypot_count = 0

    progress = st.progress(0, text="Scoring candidates …")
    for i, c in enumerate(candidates):
        if is_honeypot(c):
            honeypot_count += 1
            progress.progress((i + 1) / len(candidates))
            continue

        try:
            rule = (
                W_SKILL * skill_score(c)
                + W_CAREER * career_score(c)
                + W_EXP * experience_score(c)
                + W_LOC * location_score(c)
                + W_BEH * behavioral_score(c)
            )
            final = W_SEMANTIC * float(sims[i]) + W_RULE * rule
        except Exception as exc:
            log.warning("Scoring error for candidate %s: %s", c.get("candidate_id"), exc)
            progress.progress((i + 1) / len(candidates))
            continue

        profile = c.get("profile", {}) or {}
        results.append({
            "candidate_id": c.get("candidate_id", ""),
            "_score": final,
            "_candidate": c,
            "current_title": profile.get("current_title", ""),
            "location": profile.get("location", ""),
            "years_of_experience": profile.get("years_of_experience", 0),
        })
        progress.progress((i + 1) / len(candidates))

    progress.empty()

    if not results:
        st.warning("⚠️ No valid candidates to rank after filtering.")
        st.stop()

    if honeypot_count:
        st.info(f"ℹ️ {honeypot_count} honeypot profile(s) removed before ranking.")

    # Sort and normalise
    results.sort(key=lambda x: (-x["_score"], x["candidate_id"]))

    raw_scores = [r["_score"] for r in results]
    lo, hi = min(raw_scores), max(raw_scores)
    SCORE_MIN, SCORE_MAX = 0.30, 0.99

    def _norm(s):
        if hi == lo:
            return SCORE_MAX
        return round(SCORE_MIN + (s - lo) / (hi - lo) * (SCORE_MAX - SCORE_MIN), 4)

    from rank import _get_top_relevant_skills, _build_reasoning
    rows = []
    for rank_idx, r in enumerate(results, 1):
        c = r["_candidate"]
        top_skills = _get_top_relevant_skills(c)
        reasoning = _build_reasoning(c, top_skills)
        rows.append({
            "rank": rank_idx,
            "candidate_id": r["candidate_id"],
            "score": _norm(r["_score"]),
            "current_title": r["current_title"],
            "location": r["location"],
            "years_of_experience": r["years_of_experience"],
            "reasoning": reasoning,
        })

    # -----------------------------------------------------------------------
    # Display results
    # -----------------------------------------------------------------------
    st.success(f"✅ Ranked {len(rows)} candidates successfully.")
    df = pd.DataFrame(rows)

    st.dataframe(
        df[["rank", "candidate_id", "score", "current_title", "location", "years_of_experience", "reasoning"]],
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------------------------
    # Download CSV
    # -----------------------------------------------------------------------
    csv_buf = io.StringIO()
    df[["candidate_id", "rank", "score", "reasoning"]].to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️ Download submission.csv",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name="submission.csv",
        mime="text/csv",
    )
