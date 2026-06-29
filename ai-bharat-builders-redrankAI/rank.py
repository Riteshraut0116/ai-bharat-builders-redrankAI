"""
rank.py — Timed candidate ranking pipeline for RedrankAI.

Loads precomputed embeddings, streams candidate data, blends semantic + rule
scores, takes the top 100, and writes a validated submission.csv.

sentence_transformers is NOT imported here — all embeddings are on disk.

Usage:
    python rank.py --candidates ./candidates.jsonl.gz \
                   --embeddings ./embeddings.npy \
                   --ids ./candidate_ids.npy \
                   --jd ./jd_embedding.npy \
                   --out ./submission.csv
"""

import argparse
import csv
import gzip
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from honeypot import is_honeypot
from scorer import (
    behavioral_score,
    career_score,
    experience_score,
    location_score,
    skill_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Rule component weights (sum to 1.0)
W_SKILL = 0.35
W_CAREER = 0.25
W_EXP = 0.15
W_LOC = 0.10
W_BEHAVIORAL = 0.15

# Blend: semantic vs rule
W_SEMANTIC = 0.60
W_RULE = 0.40

# Output constraints
TOP_N = 100
SCORE_MIN = 0.30
SCORE_MAX = 0.99


def _open_candidates(path: Path):
    """Open candidates.jsonl or .jsonl.gz transparently."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _build_reasoning(c: dict, top_skills: list) -> str:
    """
    Generate a one-sentence reasoning string using only fields present in the profile.
    Never mentions skills absent from the candidate's actual skill list.
    """
    profile = c.get("profile", {}) or {}
    signals = c.get("redrob_signals", {}) or {}

    title = profile.get("current_title") or "Unknown Title"
    company = profile.get("current_company") or "Unknown Company"
    city = (profile.get("location") or "Unknown Location").split(",")[0].strip()
    yoe = profile.get("years_of_experience") or 0
    notice = signals.get("notice_period_days", 90) or 90

    skill_str = ", ".join(top_skills[:2]) if top_skills else "relevant AI skills"

    parts = [
        f"{title} at {company} ({city})",
        f"{yoe:.1f} yrs exp",
        f"skills: {skill_str}",
        f"notice: {notice}d",
    ]

    # Flag availability concerns
    concerns = []
    from datetime import date, datetime
    last_active_str = signals.get("last_active_date") or ""
    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
        days_inactive = (date.today() - last_active).days
        if days_inactive > 90:
            concerns.append(f"inactive {days_inactive}d")
    except (ValueError, TypeError):
        pass

    if notice > 60:
        concerns.append(f"notice >{notice}d")

    if concerns:
        parts.append("(" + "; ".join(concerns) + ")")

    return " | ".join(parts) + "."


def _get_top_relevant_skills(c: dict) -> list:
    """Return top 2 relevant skill names sorted by endorsements."""
    from scorer import MUST_HAVE_SKILLS, GOOD_SKILLS
    skills = c.get("skills", []) or []
    relevant = [
        s for s in skills
        if any(kw in (s.get("name") or "").lower() or (s.get("name") or "").lower() in kw
               for kw in MUST_HAVE_SKILLS | GOOD_SKILLS)
    ]
    relevant.sort(key=lambda s: s.get("endorsements", 0) or 0, reverse=True)
    return [s.get("name", "") for s in relevant[:2] if s.get("name")]


def _normalise_scores(scores: list) -> list:
    """
    Linear-normalise a list of float scores to [SCORE_MIN, SCORE_MAX].
    Preserves relative order (non-increasing). Returns a new list.
    """
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [SCORE_MAX] + [SCORE_MIN] * (len(scores) - 1)
    result = []
    for s in scores:
        normed = SCORE_MIN + (s - lo) / (hi - lo) * (SCORE_MAX - SCORE_MIN)
        result.append(round(normed, 4))
    return result


def main():
    """Entry point: load embeddings, score candidates, write submission.csv."""
    parser = argparse.ArgumentParser(description="RedrankAI candidate ranker")
    parser.add_argument("--candidates", required=True, help="candidates.jsonl or .jsonl.gz")
    parser.add_argument("--embeddings", required=True, help="embeddings.npy from precompute.py")
    parser.add_argument("--ids", required=True, help="candidate_ids.npy from precompute.py")
    parser.add_argument("--jd", required=True, help="jd_embedding.npy from precompute.py")
    parser.add_argument("--out", default="./submission.csv", help="Output CSV path")
    args = parser.parse_args()

    for p in (args.candidates, args.embeddings, args.ids, args.jd):
        if not Path(p).exists():
            log.error("File not found: %s", p)
            sys.exit(1)

    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load precomputed arrays
    # ------------------------------------------------------------------
    log.info("Loading embeddings from %s ...", args.embeddings)
    embeddings = np.load(args.embeddings)          # (N, dim)  float32
    candidate_ids_arr = np.load(args.ids, allow_pickle=True)  # (N,)
    jd_vec = np.load(args.jd)                      # (dim,)

    log.info("Loaded embeddings shape=%s", embeddings.shape)

    # ------------------------------------------------------------------
    # 2. Vectorised cosine similarity (embeddings already normalised)
    # ------------------------------------------------------------------
    log.info("Computing cosine similarities ...")
    jd_norm = np.linalg.norm(jd_vec)
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norms = np.where(emb_norms == 0, 1e-9, emb_norms)  # avoid div-by-zero
    normalised_embs = embeddings / emb_norms
    jd_unit = jd_vec / (jd_norm if jd_norm > 0 else 1e-9)
    sims = normalised_embs @ jd_unit  # (N,)

    # Build a lookup: candidate_id → semantic_score
    id_to_sim = {cid: float(sims[i]) for i, cid in enumerate(candidate_ids_arr)}

    # ------------------------------------------------------------------
    # 3. Stream candidates, filter honeypots, score each
    # ------------------------------------------------------------------
    log.info("Scoring candidates ...")
    candidates_path = Path(args.candidates)
    results = []
    honeypot_count = 0
    miss_count = 0
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

            if is_honeypot(c):
                honeypot_count += 1
                continue

            # Rule-based score
            s_skill = skill_score(c)
            s_career = career_score(c)
            s_exp = experience_score(c)
            s_loc = location_score(c)
            s_beh = behavioral_score(c)
            rule = (
                W_SKILL * s_skill
                + W_CAREER * s_career
                + W_EXP * s_exp
                + W_LOC * s_loc
                + W_BEHAVIORAL * s_beh
            )

            # Semantic score from precomputed array
            semantic = id_to_sim.get(cid)
            if semantic is None:
                miss_count += 1
                semantic = 0.0  # fallback for candidates not in embeddings

            final = W_SEMANTIC * semantic + W_RULE * rule
            results.append((cid, final, c))

            if line_num % 10_000 == 0:
                log.info("  Processed %d candidates ...", line_num)

    log.info(
        "Scoring done: %d scored, %d honeypots filtered, %d embedding misses, %d parse errors",
        len(results), honeypot_count, miss_count, parse_errors,
    )

    # ------------------------------------------------------------------
    # 4. Sort descending, apply tie-break (candidate_id ascending)
    # ------------------------------------------------------------------
    results.sort(key=lambda x: (-x[1], x[0]))

    # ------------------------------------------------------------------
    # 5. Take top 100 and normalise scores to [SCORE_MIN, SCORE_MAX]
    # ------------------------------------------------------------------
    top = results[:TOP_N]
    raw_scores = [r[1] for r in top]
    normed_scores = _normalise_scores(raw_scores)

    # Verify non-increasing after normalisation (should always hold)
    for i in range(len(normed_scores) - 1):
        if normed_scores[i] < normed_scores[i + 1]:
            log.error(
                "Non-increasing score violation at ranks %d/%d: %.4f < %.4f",
                i + 1, i + 2, normed_scores[i], normed_scores[i + 1],
            )

    # ------------------------------------------------------------------
    # 6. Generate reasoning and write CSV
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Writing %s ...", out_path)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank_idx, ((cid, raw_score, c), normed_score) in enumerate(zip(top, normed_scores), 1):
            top_skills = _get_top_relevant_skills(c)
            reasoning = _build_reasoning(c, top_skills)
            writer.writerow([cid, rank_idx, f"{normed_score:.4f}", reasoning])

    elapsed = time.time() - t0
    log.info("Done. Total time: %.1f seconds", elapsed)
    log.info("Output: %s", out_path.resolve())


if __name__ == "__main__":
    main()
