import time
from threading import Lock

CE_THRESHOLD = 1.0
_pending: dict[str, dict] = {}
_lock = Lock()


def is_context_sufficient(constraints: dict) -> bool:
    has_skill = bool(constraints.get("skills"))
    has_role = bool(constraints.get("role"))
    has_test_name = bool(constraints.get("test_names"))
    has_seniority = bool(constraints.get("seniority"))
    return has_skill or has_role or has_test_name or has_seniority


def needs_review(results: list[dict]) -> bool:
    if not results:
        return True
    scores = [r.get("_ce_score", 0) for r in results]
    return max(scores) < CE_THRESHOLD


def create_review(session_id: str, query: str, results: list[dict]):
    with _lock:
        _pending[session_id] = {
            "query": query,
            "results": results,
            "created_at": time.time(),
            "status": "pending",
        }


def get_review(session_id: str) -> dict | None:
    with _lock:
        return _pending.get(session_id)


def approve_review(session_id: str, approved_indices: list[int]) -> list[dict] | None:
    with _lock:
        entry = _pending.get(session_id)
        if entry is None or entry["status"] != "pending":
            return None
        entry["status"] = "approved"
        return [entry["results"][i] for i in approved_indices if i < len(entry["results"])]


def reject_review(session_id: str):
    with _lock:
        entry = _pending.get(session_id)
        if entry:
            entry["status"] = "rejected"


def validate_recommendations(parsed: dict, allowed_names: set[str]) -> dict:
    validated = []
    hallucinated = []
    for rec in parsed.get("recommendations", []):
        if rec.get("name") in allowed_names:
            validated.append(rec)
        else:
            hallucinated.append(rec.get("name"))
    parsed["recommendations"] = validated
    return parsed, hallucinated
