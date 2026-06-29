"""
honeypot.py — Detect fabricated or impossibly-inflated candidate profiles.

A honeypot profile is designed to game a naive keyword-based ranker.
These profiles are excluded from ranking entirely by rank.py.
"""


def is_honeypot(c: dict) -> bool:
    """
    Return True if the candidate profile shows clear signs of fabrication.

    Conditions (ANY one is sufficient):
        1. 8 or more skills with proficiency == 'expert' AND endorsements == 0
        2. profile_completeness_score == 100 AND count of expert skills >= 15
        3. 6 or more skills with proficiency == 'expert' AND duration_months == 0
           AND endorsements == 0  (claimed expertise with zero usage or social proof)
        4. years_of_experience > 35
    """
    skills = c.get("skills", []) or []
    signals = c.get("redrob_signals", {}) or {}
    profile = c.get("profile", {}) or {}

    yoe = profile.get("years_of_experience", 0) or 0
    completeness = signals.get("profile_completeness_score", 0) or 0

    expert_zero_endorsements = sum(
        1 for s in skills
        if (s.get("proficiency") or "").lower() == "expert"
        and (s.get("endorsements") or 0) == 0
    )

    expert_count = sum(
        1 for s in skills
        if (s.get("proficiency") or "").lower() == "expert"
    )

    expert_zero_duration_zero_endorse = sum(
        1 for s in skills
        if (s.get("proficiency") or "").lower() == "expert"
        and (s.get("duration_months") or 0) == 0
        and (s.get("endorsements") or 0) == 0
    )

    if expert_zero_endorsements >= 8:
        return True

    if completeness == 100 and expert_count >= 15:
        return True

    if expert_zero_duration_zero_endorse >= 6:
        return True

    if yoe > 35:
        return True

    return False
