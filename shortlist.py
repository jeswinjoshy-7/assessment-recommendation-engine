"""
Context-aware final shortlist selection pass.

Takes retrieved candidates + full conversation constraints, selects 1-10
items maximizing relevance + diversity + constraint coverage.
"""

MAX_SHORTLIST = 10


def _coverage_score(candidate: dict, constraints: dict) -> int:
    score = 0
    skills = set(s.lower() for s in constraints.get("skills", []))
    seniority = constraints.get("seniority", "").lower()
    role = constraints.get("role", "").lower()
    test_type = constraints.get("test_type", "").lower()
    language = constraints.get("language", "").lower()

    name_desc = (candidate.get("name", "") + " " + candidate.get("description", "")).lower()
    for s in skills:
        if s in name_desc:
            score += 2

    job_levels = [j.lower() for j in candidate.get("job_levels", [])]
    if seniority and any(seniority in jl for jl in job_levels):
        score += 2

    if role and role in name_desc:
        score += 1

    if test_type and test_type == candidate.get("type", "").lower():
        score += 1

    if language:
        langs = [l.lower() for l in candidate.get("languages", [])]
        if language in langs or any(lang.startswith(language[:2]) for lang in langs):
            score += 1

    return score


def _is_diverse(candidate: dict, selected: list[dict]) -> bool:
    if not selected:
        return True
    c_levels = set(candidate.get("job_levels", []))
    c_types = {candidate.get("type", "")}
    c_industries = set(candidate.get("industries", []))
    for s in selected:
        s_levels = set(s.get("job_levels", []))
        s_types = {s.get("type", "")}
        s_industries = set(s.get("industries", []))
        if c_levels == s_levels and c_types == s_types and c_industries == s_industries:
            return False
    return True


def select(results: list[dict], constraints: dict) -> list[dict]:
    if not results:
        return []

    scored = []
    for r in results:
        r["_cov_score"] = _coverage_score(r, constraints)
        blend = r.get("_ce_score", 0) * 0.6 + r["_cov_score"] * 0.4
        r["_blend"] = blend
        scored.append(r)

    scored.sort(key=lambda r: r["_blend"], reverse=True)
    selected = []
    for r in scored:
        if len(selected) >= MAX_SHORTLIST:
            break
        if _is_diverse(r, selected):
            selected.append(r)

    if not selected:
        selected = scored[:1]

    return selected
