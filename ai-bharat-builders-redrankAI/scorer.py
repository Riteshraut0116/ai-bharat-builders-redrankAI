"""
scorer.py — Five pure scoring functions for RedrankAI.

Each function accepts a candidate dict (matching candidate_schema.json) and returns
a float in [0.0, 1.0]. All JD-specific constants are hardcoded at module level.
"""

import math
from datetime import date, datetime

# ---------------------------------------------------------------------------
# JD Constants — Senior AI Engineer @ Redrob AI
# ---------------------------------------------------------------------------
JD_YOE_MIN = 5
JD_YOE_MAX = 9

JD_LOCATIONS = {
    "pune", "noida", "bangalore", "bengaluru", "hyderabad",
    "mumbai", "delhi", "gurgaon", "gurugram", "chennai",
}

SERVICES_FIRMS = {
    "tcs", "infosys", "wipro", "cognizant", "accenture",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware",
}

MUST_HAVE_SKILLS = {
    "embeddings", "vector search", "faiss", "pinecone", "qdrant", "milvus",
    "weaviate", "opensearch", "elasticsearch", "sentence-transformers",
    "semantic search", "hybrid search", "retrieval", "information retrieval",
    "ranking", "re-ranking", "rag", "ndcg", "mrr", "map", "python", "nlp",
    "llm", "recommendation", "ann search", "bge", "e5", "text embeddings",
}

GOOD_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "learning to rank", "xgboost",
    "lightgbm", "pytorch", "transformers", "bert", "a/b testing",
    "feature engineering", "mlflow", "docker", "kubernetes", "spark",
    "aws", "gcp",
}

RESEARCH_TITLES = {"research scientist", "researcher", "postdoc", "phd researcher"}

PROD_KEYWORDS = {
    "production", "deployed", "real users", "shipped", "scale",
    "a/b test", "latency", "inference", "ranking", "retrieval", "embedding", "vector",
}

_PROFICIENCY_MULT = {
    "expert": 1.0,
    "advanced": 0.8,
    "intermediate": 0.5,
    "beginner": 0.2,
}

TODAY = date.today()


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float to [lo, hi]."""
    return max(lo, min(hi, val))


def _is_services_firm(company_name: str) -> bool:
    """Return True if the company name contains a known IT-services firm name."""
    name_lower = company_name.lower()
    return any(firm in name_lower for firm in SERVICES_FIRMS)


# ---------------------------------------------------------------------------
# 1. skill_score
# ---------------------------------------------------------------------------

def skill_score(c: dict) -> float:
    """
    Score based on how well the candidate's skills match the JD.

    Each matching skill contributes:
        proficiency_mult × endorsement_trust × duration_trust × base_weight

    where:
        base_weight = 1.0 for MUST_HAVE_SKILLS, 0.5 for GOOD_SKILLS (substring match)
        endorsement_trust = min(endorsements / 50, 1.0)
        duration_trust    = min(duration_months / 60, 1.0)

    Bonus from redrob_signals.skill_assessment_scores: up to 0.1 added.
    Returns a float in [0.0, 1.0].
    """
    skills = c.get("skills", []) or []
    signals = c.get("redrob_signals", {}) or {}
    assessment_scores = signals.get("skill_assessment_scores", {}) or {}

    total = 0.0
    max_possible = 0.0

    for skill in skills:
        name = (skill.get("name") or "").lower()
        proficiency = (skill.get("proficiency") or "beginner").lower()
        endorsements = skill.get("endorsements", 0) or 0
        duration_months = skill.get("duration_months", 0) or 0

        prof_mult = _PROFICIENCY_MULT.get(proficiency, 0.2)
        endorse_trust = min(endorsements / 50.0, 1.0)
        duration_trust = min(duration_months / 60.0, 1.0)
        # When both are 0, give partial credit for the proficiency alone
        signal_trust = max(0.3, (endorse_trust + duration_trust) / 2.0)

        # Substring matching against MUST_HAVE and GOOD skill sets
        is_must = any(kw in name or name in kw for kw in MUST_HAVE_SKILLS)
        is_good = (not is_must) and any(kw in name or name in kw for kw in GOOD_SKILLS)

        if is_must:
            base = 1.0
            max_possible += 1.0
            total += prof_mult * signal_trust * base
        elif is_good:
            base = 0.5
            max_possible += 0.5
            total += prof_mult * signal_trust * base

    # Normalise against a cap of 10 must-have skills worth of credit
    norm_cap = 10.0
    raw_score = total / norm_cap if total > 0 else 0.0

    # Bonus: average assessment score on relevant skills (up to 0.10)
    if assessment_scores:
        relevant_assessments = []
        for skill_name, score in assessment_scores.items():
            sn = skill_name.lower()
            if any(kw in sn or sn in kw for kw in MUST_HAVE_SKILLS | GOOD_SKILLS):
                relevant_assessments.append(score)
        if relevant_assessments:
            avg_assessment = sum(relevant_assessments) / len(relevant_assessments)
            raw_score += (avg_assessment / 100.0) * 0.10

    return _clamp(raw_score)


# ---------------------------------------------------------------------------
# 2. career_score
# ---------------------------------------------------------------------------

def career_score(c: dict) -> float:
    """
    Score based on career trajectory favouring product-company tenure and
    production-deployment evidence.

    Penalties:
        - services firm ratio > 90% of career: -0.4
        - services firm ratio > 50% of career: -0.2
        - research-only titles > 80% of career months: -0.3
        - more than 3 jobs under 12 months (job-hopping): -0.15
        - each extra job under 12 months beyond 3: -0.05 each

    Bonus:
        - current title contains senior/lead/staff/principal/architect: +0.15

    Returns a float in [0.0, 1.0].
    """
    history = c.get("career_history", []) or []
    if not history:
        return 0.2  # default for empty history

    total_months = 0
    services_months = 0
    research_months = 0
    product_months = 0
    prod_keyword_score = 0.0
    short_stints = 0

    for job in history:
        company = (job.get("company") or "").lower()
        title = (job.get("title") or "").lower()
        duration = job.get("duration_months", 0) or 0
        description = (job.get("description") or "").lower()

        total_months += duration

        if _is_services_firm(company):
            services_months += duration
        else:
            product_months += duration

        # Research-only detection
        if any(rt in title for rt in RESEARCH_TITLES):
            research_months += duration

        # Production keyword bonus — weighted by duration
        hit_count = sum(1 for kw in PROD_KEYWORDS if kw in description)
        if duration > 0 and hit_count > 0:
            # Normalise: up to 5 keyword hits considered rich
            prod_keyword_score += min(hit_count / 5.0, 1.0) * min(duration / 24.0, 1.0)

        if duration < 12:
            short_stints += 1

    # Base score from product tenure ratio
    if total_months > 0:
        product_ratio = product_months / total_months
        services_ratio = services_months / total_months
        research_ratio = research_months / total_months
    else:
        product_ratio = services_ratio = research_ratio = 0.0

    base = 0.4 + product_ratio * 0.4

    # Production keyword contribution (up to 0.2)
    base += min(prod_keyword_score / max(len(history), 1), 0.2)

    # Penalties
    if services_ratio > 0.9:
        base -= 0.4
    elif services_ratio > 0.5:
        base -= 0.2

    if research_ratio > 0.8:
        base -= 0.3

    if short_stints > 3:
        base -= 0.15
        base -= 0.05 * (short_stints - 3)

    # Seniority bonus
    current_title = (c.get("profile", {}).get("current_title") or "").lower()
    if any(kw in current_title for kw in ("senior", "lead", "staff", "principal", "architect")):
        base += 0.15

    return _clamp(base)


# ---------------------------------------------------------------------------
# 3. experience_score
# ---------------------------------------------------------------------------

def experience_score(c: dict) -> float:
    """
    Score based on years of experience relative to the JD's 5–9 year sweet spot.

    Bands:
        < 3 yrs       → 0.10
        3 to YOE_MIN  → linear scale 0.40–0.80
        YOE_MIN to YOE_MAX → 0.80–1.00 (peak at midpoint)
        > YOE_MAX     → decay 0.04 per year over, floor 0.40

    Returns a float in [0.0, 1.0].
    """
    yoe = c.get("profile", {}).get("years_of_experience", 0) or 0

    if yoe < 3:
        return 0.10

    if yoe < JD_YOE_MIN:
        # Linear interpolation 0.40 → 0.80 over [3, JD_YOE_MIN]
        t = (yoe - 3) / max(JD_YOE_MIN - 3, 1)
        return _clamp(0.40 + t * 0.40)

    if yoe <= JD_YOE_MAX:
        # Bell: peak 1.0 at midpoint, 0.80 at edges
        midpoint = (JD_YOE_MIN + JD_YOE_MAX) / 2.0
        half_width = (JD_YOE_MAX - JD_YOE_MIN) / 2.0
        dist = abs(yoe - midpoint) / max(half_width, 1)
        return _clamp(1.0 - dist * 0.20)

    # Over JD_YOE_MAX: decay 0.04 per year
    extra = yoe - JD_YOE_MAX
    return _clamp(0.80 - extra * 0.04, lo=0.40)


# ---------------------------------------------------------------------------
# 4. location_score
# ---------------------------------------------------------------------------

def location_score(c: dict) -> float:
    """
    Score based on candidate location vs. preferred JD cities, adjusted for
    notice period.

    Preferred cities (JD_LOCATIONS):  1.0
    India but not preferred city:     0.5 if willing_to_relocate else 0.3
    Outside India:                    0.1

    Notice modifier:
        ≤ 30 days → 1.0
        ≤ 60 days → 0.9
        ≤ 90 days → 0.8
        > 90 days → 0.6

    Returns a float in [0.0, 1.0].
    """
    profile = c.get("profile", {}) or {}
    signals = c.get("redrob_signals", {}) or {}

    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)
    notice_period = signals.get("notice_period_days", 90) or 90

    # Location base score
    in_preferred = any(city in location for city in JD_LOCATIONS)
    in_india = "india" in country or any(
        city in location for city in JD_LOCATIONS
    ) or country in ("in", "ind")

    if in_preferred:
        loc_base = 1.0
    elif in_india:
        loc_base = 0.5 if willing_to_relocate else 0.3
    else:
        loc_base = 0.1

    # Notice modifier
    if notice_period <= 30:
        notice_mult = 1.0
    elif notice_period <= 60:
        notice_mult = 0.9
    elif notice_period <= 90:
        notice_mult = 0.8
    else:
        notice_mult = 0.6

    return _clamp(loc_base * notice_mult)


# ---------------------------------------------------------------------------
# 5. behavioral_score
# ---------------------------------------------------------------------------

def behavioral_score(c: dict) -> float:
    """
    Score based on 23 Redrob platform behavioural signals.

    Blend:
        0.25 × activity_score       (recency of last_active_date)
        0.10 × open_to_work
        0.20 × recruiter_response_rate
        0.15 × interview_completion_rate
        0.10 × offer_acceptance_rate  (-1 → neutral 0.5)
        0.10 × github_activity / 100  (-1 → 0)
        0.05 × profile_completeness / 100
        0.05 × verified_signals       (email + phone + linkedin, 0–1)

    Returns a float in [0.0, 1.0].
    """
    signals = c.get("redrob_signals", {}) or {}

    # Activity: days since last_active_date
    last_active_str = signals.get("last_active_date") or ""
    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
        days_inactive = (TODAY - last_active).days
    except (ValueError, TypeError):
        days_inactive = 9999

    if days_inactive < 30:
        activity_score = 1.0
    elif days_inactive < 90:
        activity_score = 0.7
    elif days_inactive < 180:
        activity_score = 0.4
    else:
        activity_score = 0.1

    open_to_work = 1.0 if signals.get("open_to_work_flag", False) else 0.0

    recruiter_response = _clamp(signals.get("recruiter_response_rate", 0.0) or 0.0)

    interview_completion = _clamp(signals.get("interview_completion_rate", 0.0) or 0.0)

    raw_offer = signals.get("offer_acceptance_rate", -1)
    offer_acceptance = 0.5 if (raw_offer is None or raw_offer == -1) else _clamp(raw_offer)

    raw_github = signals.get("github_activity_score", -1)
    github_norm = 0.0 if (raw_github is None or raw_github == -1) else _clamp(raw_github / 100.0)

    completeness = _clamp((signals.get("profile_completeness_score", 0) or 0) / 100.0)

    verified_email = 1.0 if signals.get("verified_email", False) else 0.0
    verified_phone = 1.0 if signals.get("verified_phone", False) else 0.0
    linkedin = 1.0 if signals.get("linkedin_connected", False) else 0.0
    verified_signals = (verified_email + verified_phone + linkedin) / 3.0

    score = (
        0.25 * activity_score
        + 0.10 * open_to_work
        + 0.20 * recruiter_response
        + 0.15 * interview_completion
        + 0.10 * offer_acceptance
        + 0.10 * github_norm
        + 0.05 * completeness
        + 0.05 * verified_signals
    )

    return _clamp(score)
