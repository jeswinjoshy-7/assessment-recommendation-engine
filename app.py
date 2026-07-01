import os
import re
import json
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq

load_dotenv()
from retriever import search, lookup_by_names, detect_test_names, build_cumulative_query, grounding_verify
from shortlist import select as select_shortlist
from guardrails import needs_review, create_review, get_review, approve_review, reject_review, is_context_sufficient, validate_recommendations

with open("system_prompt.md") as f:
    SYSTEM_PROMPT = f.read()

app = FastAPI(title="SHL Assessment Recommender")
MAX_TURNS = 8
CHAT_TIMEOUT = 28


def _get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


SKILL_MAP = {
    "java": "Java", "python": "Python", "javascript": "JavaScript", "js": "JavaScript",
    "typescript": "TypeScript", "c++": "C++", "c#": "C#", "dotnet": ".NET",
    "sql": "SQL", "database": "Database", "sales": "Sales", "marketing": "Marketing",
    "leadership": "Leadership", "management": "Management", "coding": "Coding",
    "programming": "Programming", "communication": "Communication", "analytical": "Analytical",
    "problem.solving": "Problem Solving", "teamwork": "Teamwork", "customer.service": "Customer Service",
    "accounting": "Accounting", "finance": "Finance", "engineering": "Engineering",
    "technical": "Technical",
    "testing": "Testing", "quality.assurance": "Quality Assurance", "data": "Data Analysis",
    "digital": "Digital", "hr": "HR", "human.resources": "Human Resources",
}

SENIORITY_MAP = {
    "entry.level": "Entry-Level", "entry": "Entry-Level", "junior": "Entry-Level",
    "fresh": "Entry-Level", "graduate": "Graduate", "new.grad": "Graduate",
    "mid.level": "Professional", "mid": "Professional", "professional": "Professional",
    "experienced": "Professional", "senior": "Senior", "lead": "Senior",
    "manager": "Front Line Manager", "director": "Director", "executive": "Executive",
    "vp": "Executive", "head": "Director", "chief": "Executive",
}

LANG_MAP = {
    "english": "en-US", "spanish": "es-ES", "french": "fr-FR",
    "german": "de-DE", "chinese": "zh-CN", "japanese": "ja-JP",
    "arabic": "ar-SA", "dutch": "nl-NL", "portuguese": "pt-BR",
    "italian": "it-IT", "russian": "ru-RU", "korean": "ko-KR",
}

TYPE_KEYWORDS = {"assessment": "Assessment", "report": "Report", "test": "Assessment"}

INDUSTRY_MAP = {
    "safety": "Safety/Public", "public": "Safety/Public", "retail": "Retail",
    "manufacturing": "Manufacturing", "healthcare": "Healthcare", "health": "Healthcare",
    "medical": "Healthcare", "customer.service": "Customer Service",
    "telecom": "Telecommunications", "telecommunications": "Telecommunications",
    "banking": "Banking/Finance", "finance": "Banking/Finance", "financial": "Banking/Finance",
    "insurance": "Insurance", "hospitality": "Hospitality", "hotel": "Hospitality",
}


def _extract_constraints(text: str, existing: dict) -> dict:
    c = dict(existing)
    tl = text.lower()
    skills = c.get("skills", [])
    for pattern, skill in SKILL_MAP.items():
        if re.search(r'\b' + pattern.replace('.', r'[\s.]') + r'\b', tl):
            if skill not in skills:
                skills.append(skill)
    if re.search(r'\bIT\b', text) and 'IT' not in skills:
        skills.append('IT')
    if skills:
        c["skills"] = skills
    for pattern, level in SENIORITY_MAP.items():
        if re.search(r'\b' + pattern.replace('.', r'[\s.]') + r'\b', tl):
            c["seniority"] = level
            break
    for keyword, lang in LANG_MAP.items():
        if re.search(r'\b' + keyword + r'\b', tl):
            c["language"] = lang
            break
    for keyword, ind in INDUSTRY_MAP.items():
        if re.search(r'\b' + keyword.replace('.', r'[\s.]') + r'\b', tl):
            c["industry"] = ind
            break
    for keyword, tt in TYPE_KEYWORDS.items():
        if re.search(r'\b' + keyword + r'\b', tl):
            c["test_type"] = tt
            break
    test_names = detect_test_names(text)
    if test_names:
        c["test_names"] = test_names
    role_patterns = [
        r'\b(sales\s*(person|rep|representative|associate)s?)\b',
        r'\b(software\s*engineer(s)?|developer(s)?|programmer(s)?)\b',
        r'\b(engineer(s)?)\b',
        r'\b(customer\s*service\s*(rep|representative)s?)\b',
        r'\b(account\s*manager(s)?)\b',
        r'\b(project\s*manager(s)?)\b',
        r'\b(product\s*manager(s)?)\b',
        r'\b(data\s*(scientist(s)?|analyst(s)?))\b',
        r'\b(financial\s*analyst(s)?|accountant(s)?)\b',
        r'\b(hr\s*manager(s)?|recruiter(s)?)\b',
        r'\b(executive(s)?|director(s)?|vp|vice\s*president(s)?)\b',
        r'\b(manager(s)?|supervisor(s)?|team\s*lead(s)?)\b',
    ]
    for pat in role_patterns:
        m = re.search(pat, tl)
        if m:
            c["role"] = m.group(1).strip()
            break
    return c


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] | None = None
    message: str | None = None
    session_id: str | None = None
    filters: dict | None = None


class ChatResponse(BaseModel):
    reply: str
    recommendations: list
    end_of_conversation: bool
    needs_review: bool = False
    context_ready: bool = False


class ReviewRequest(BaseModel):
    session_id: str
    action: str
    approved_indices: list[int] = []


class SessionState(BaseModel):
    constraints: dict = {}
    shortlist: list = []
    turn: int = 0
    context_ready: bool = False
    history: list = []


sessions: dict[str, SessionState] = {}


@app.exception_handler(Exception)
async def _fallback_error(request: Request, exc: Exception):
    return JSONResponse(
        status_code=200,
        content={
            "reply": "An unexpected error occurred. Please try again.",
            "recommendations": [],
            "end_of_conversation": True,
            "needs_review": False,
            "context_ready": False,
        },
    )


def _clean_results(results):
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]


def _call_llm(query: str, results: list[dict], state: SessionState) -> dict:
    allowed_names = {r["name"] for r in results}
    context = (
        f"You may ONLY recommend assessments from the following list.\n"
        f"Do NOT add any assessment not in this list.\n\n"
        f"Retrieved assessments:\n{json.dumps(_clean_results(results), indent=2)}\n\n"
        f"User message: {query}"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in state.history:
        role = "assistant" if h["role"] == "model" else h["role"]
        messages.append({"role": role, "content": h["text"]})
    messages.append({"role": "user", "content": context})
    response = _get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    parsed = json.loads(response.choices[0].message.content)
    parsed, hallucinated = validate_recommendations(parsed, allowed_names)
    if hallucinated and not parsed["recommendations"]:
        retry_messages = messages + [
            {"role": "assistant", "content": response.choices[0].message.content},
            {"role": "user", "content": f"The previous response contained names not in the allowed list: {hallucinated}. Respond again using ONLY assessments from the provided list."},
        ]
        retry = _get_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=retry_messages,
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        parsed = json.loads(retry.choices[0].message.content)
        parsed, _ = validate_recommendations(parsed, allowed_names)
    state.history.append({"role": "user", "text": query})
    state.history.append({"role": "assistant", "text": json.dumps(parsed)})
    return parsed


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        return await asyncio.wait_for(_chat(req), timeout=CHAT_TIMEOUT)
    except asyncio.TimeoutError:
        return ChatResponse(reply="Request timed out. Please try again.", recommendations=[], end_of_conversation=True)


async def _chat(req: ChatRequest):
    if req.messages:
        last_user_msg = ""
        prior_history = []
        for m in req.messages:
            if m.role == "user":
                last_user_msg = m.content
            elif m.role in ("assistant", "model"):
                prior_history.append({"role": "model", "text": m.content})
        query = last_user_msg
    else:
        query = req.message or ""
        prior_history = []

    sid = req.session_id or "default"
    if sid not in sessions:
        sessions[sid] = SessionState()
    state = sessions[sid]

    if prior_history and not state.history:
        state.history = prior_history

    state.turn += 1
    if state.turn > MAX_TURNS:
        return ChatResponse(reply="Conversation limit reached. Please start a new session.", recommendations=[], end_of_conversation=True)

    constraints = _extract_constraints(query, state.constraints)
    state.constraints = constraints
    context_ready = is_context_sufficient(constraints)
    state.context_ready = context_ready
    test_names = constraints.get("test_names", [])

    if test_names:
        results = lookup_by_names(test_names)
        results = grounding_verify(results)
    elif context_ready:
        cumulative_query = build_cumulative_query(constraints)
        results = search(cumulative_query or query, k=10, filters=req.filters)
        results = select_shortlist(results, constraints)
        state.shortlist = results
    else:
        results = []

    if needs_review(results) and results:
        create_review(sid, query, results)
        return ChatResponse(
            reply="I need a human to verify these results before proceeding.",
            recommendations=_clean_results(results),
            end_of_conversation=False,
            needs_review=True,
            context_ready=context_ready,
        )

    if not context_ready:
        return ChatResponse(reply="", recommendations=[], end_of_conversation=False, context_ready=False)

    parsed = _call_llm(query, results, state)

    if parsed.get("end_of_conversation"):
        sessions.pop(sid, None)

    return ChatResponse(**parsed, context_ready=True)


@app.post("/chat/review", response_model=ChatResponse)
async def review(req: ReviewRequest):
    if req.action == "approve":
        approved = approve_review(req.session_id, req.approved_indices)
        if approved is None:
            return ChatResponse(reply="No pending review found.", recommendations=[], end_of_conversation=False)
        review_data = get_review(req.session_id)
        if req.session_id not in sessions:
            sessions[req.session_id] = SessionState()
        state = sessions[req.session_id]
        state.shortlist = approved
        state.context_ready = True
        parsed = _call_llm(review_data["query"], approved, state)
        if parsed.get("end_of_conversation"):
            sessions.pop(req.session_id, None)
        return ChatResponse(**parsed, context_ready=True)
    elif req.action == "reject":
        reject_review(req.session_id)
        return ChatResponse(
            reply="The human reviewer rejected the results. Please try rephrasing your query.",
            recommendations=[], end_of_conversation=False,
        )
    return ChatResponse(reply="Invalid action.", recommendations=[], end_of_conversation=False)
