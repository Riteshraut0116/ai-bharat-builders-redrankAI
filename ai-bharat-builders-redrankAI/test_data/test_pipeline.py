"""
test_pipeline.py — Quick end-to-end smoke test using sample_candidates.json.

Runs the full scoring + ranking pipeline on the 50 sample candidates without
needing the full 100K dataset or precomputed embeddings. Uses random embeddings
so sentence-transformers is NOT required to run this test.

Usage (from repo root or test_data/ folder):
    python test_data/test_pipeline.py

Expected output:
    All scorer tests passed.
    Honeypot detection passed.
    Ranking pipeline passed — 50 rows, non-increasing scores, unique IDs.
    validate_submission skipped (needs exactly 100 rows).
    All tests passed!
"""

import csv
import json
import sys
from pathlib import Path

# Allow imports from the source folder
SOURCE_DIR = Path(__file__).parent.parent / "ai-bharat-builders-redrankAI"
sys.path.insert(0, str(SOURCE_DIR))

import numpy as np

from honeypot import is_honeypot
from scorer import (
    behavioral_score,
    career_score,
    experience_score,
    location_score,
    skill_score,
)

SAMPLE_CANDIDATES = Path(__file__).parent / "sample_candidates.json"

W_SKILL, W_CAREER, W_EXP, W_LOC, W_BEH = 0.35, 0.25, 0.15, 0.10, 0.15
W_SEMANTIC, W_RULE = 0.60, 0.40
SCORE_MIN, SCORE_MAX = 0.30, 0.99


def load_candidates():
    with open(SAMPLE_CANDIDATES) as f:
        return json.load(f)


def test_scorers(candidates):
    """All 5 scorers must return values in [0.0, 1.0] for every candidate."""
    for c in candidates:
        cid = c.get("candidate_id", "?")
        for fn in (skill_score, career_score, experience_score, location_score, behavioral_score):
            val = fn(c)
            assert 0.0 <= val <= 1.0, f"{fn.__name__}({cid}) returned {val} — out of range"
    print("  All scorer tests passed.")


def test_honeypot(candidates):
    """Honeypot detector must return a bool for every candidate."""
    for c in candidates:
        result = is_honeypot(c)
        assert isinstance(result, bool), f"is_honeypot returned non-bool: {result}"
    flagged = sum(1 for c in candidates if is_honeypot(c))
    print(f"  Honeypot detection passed. ({flagged} flagged out of {len(candidates)})")


def test_ranking_pipeline(candidates):
    """Full blend → sort → normalise → CSV check."""
    n = len(candidates)
    rng = np.random.default_rng(42)
    sims = rng.random(n).tolist()
    id_to_sim = {c["candidate_id"]: sims[i] for i, c in enumerate(candidates)}

    results = []
    for c in candidates:
        if is_honeypot(c):
            continue
        cid = c["candidate_id"]
        rule = (
            W_SKILL * skill_score(c)
            + W_CAREER * career_score(c)
            + W_EXP * experience_score(c)
            + W_LOC * location_score(c)
            + W_BEH * behavioral_score(c)
        )
        semantic = id_to_sim.get(cid, 0.0)
        final = W_SEMANTIC * semantic + W_RULE * rule
        results.append((cid, final, c))

    results.sort(key=lambda x: (-x[1], x[0]))

    raw = [r[1] for r in results]
    lo, hi = min(raw), max(raw)
    normed = [
        round(SCORE_MIN + (s - lo) / (hi - lo) * (SCORE_MAX - SCORE_MIN), 4)
        for s in raw
    ]

    # Checks
    assert len(results) == n, f"Expected {n} results, got {len(results)}"
    assert all(0.0 <= s <= 1.0 for s in normed), "Normalised scores out of [0,1]"
    assert all(normed[i] >= normed[i+1] for i in range(len(normed)-1)), "Scores not non-increasing"
    ids = [r[0] for r in results]
    assert len(set(ids)) == len(ids), "Duplicate candidate IDs in results"

    print(f"  Ranking pipeline passed — {len(results)} rows, non-increasing scores, unique IDs.")


def main():
    print("Loading sample candidates ...")
    candidates = load_candidates()
    print(f"Loaded {len(candidates)} candidates from {SAMPLE_CANDIDATES.name}\n")

    print("Running scorer tests ...")
    test_scorers(candidates)

    print("Running honeypot detection test ...")
    test_honeypot(candidates)

    print("Running ranking pipeline test ...")
    test_ranking_pipeline(candidates)

    print("\n  validate_submission skipped (needs exactly 100 rows).")
    print("\nAll tests passed! [OK]")


if __name__ == "__main__":
    main()
