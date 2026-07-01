import json
import os
import math
import re
import weaviate
from weaviate.classes.query import Filter as WFilter
from collections import Counter
from sentence_transformers import SentenceTransformer, CrossEncoder

MODEL_NAME = "jinaai/jina-embeddings-v2-small-en"
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CATALOG_PATH = os.path.join("data", "catalog_clean.json")

_model = None
_client = None
_ce_model = None
_bm25_scorer = None
_catalog = None
_catalog_names = None


def _load_catalog():
    global _catalog, _catalog_names
    if _catalog is not None:
        return
    with open(CATALOG_PATH) as f:
        _catalog = json.load(f)
    _catalog_names = {p["name"]: p for p in _catalog}


class BM25Scorer:
    def __init__(self, corpus_path):
        with open(corpus_path) as f:
            self.products = json.load(f)
        self.N = len(self.products)
        self.k1, self.b = 1.5, 0.75
        self.df = {}
        total_dl = 0
        for p in self.products:
            text = self._text(p)
            terms = set(text.lower().split())
            for t in terms:
                self.df[t] = self.df.get(t, 0) + 1
            total_dl += len(text.split())
        self.avg_dl = total_dl / self.N if self.N else 1
        self._name_map = {p["name"]: i for i, p in enumerate(self.products)}

    def _text(self, p):
        return f"{p['name']} {p['description']}"

    def score(self, query, product_name):
        idx = self._name_map.get(product_name)
        if idx is None:
            return 0.0
        p = self.products[idx]
        doc = self._text(p).lower().split()
        dl = len(doc)
        tf = Counter(doc)
        query_terms = query.lower().split()
        s = 0.0
        for t in set(query_terms):
            term_tf = tf.get(t, 0)
            if term_tf == 0:
                continue
            idf = math.log(
                (self.N - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5) + 1
            )
            s += idf * (term_tf * (self.k1 + 1)) / (
                term_tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            )
        return float(s)


def _get_bm25():
    global _bm25_scorer
    if _bm25_scorer is None:
        _bm25_scorer = BM25Scorer(CATALOG_PATH)
    return _bm25_scorer


def _get_ce():
    global _ce_model
    if _ce_model is None:
        _ce_model = CrossEncoder(CE_MODEL_NAME)
    return _ce_model


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    return _model


def _get_client():
    global _client
    if _client is None:
        host = os.getenv("WEAVIATE_HOST", "localhost")
        _client = weaviate.connect_to_local(host=host)
    return _client


def _build_filters(filters: dict) -> WFilter | None:
    ops = []
    if filters.get("job_levels"):
        ops.append(WFilter.by_property("job_levels").contains_any(filters["job_levels"]))
    if filters.get("industries"):
        ops.append(WFilter.by_property("industries").contains_any(filters["industries"]))
    if filters.get("languages"):
        ops.append(WFilter.by_property("languages").contains_any(filters["languages"]))
    if filters.get("type"):
        ops.append(WFilter.by_property("type").equal(filters["type"]))
    if filters.get("propositions"):
        ops.append(WFilter.by_property("propositions").contains_any(filters["propositions"]))
    if filters.get("training_levels"):
        ops.append(WFilter.by_property("training_levels").contains_any(filters["training_levels"]))
    if not ops:
        return None
    return ops[0] if len(ops) == 1 else WFilter.all_of(ops)


def smart_alpha(query: str) -> float:
    q = query.lower()
    acronym_count = len(re.findall(r'\b[A-Z]{2,}\b', query))
    word_count = len(q.split())
    if acronym_count >= 2 and word_count < 5:
        return 0.2
    if acronym_count >= 1:
        return 0.3
    if word_count > 8:
        return 0.7
    return 0.5


def build_cumulative_query(constraints: dict) -> str:
    parts = []
    if constraints.get("skills"):
        parts.append(" ".join(constraints["skills"]))
    if constraints.get("role"):
        parts.append(constraints["role"])
    if constraints.get("seniority"):
        parts.append(constraints["seniority"])
    if constraints.get("test_type"):
        parts.append(constraints["test_type"])
    return " ".join(parts) if parts else ""


def grounding_verify(results: list[dict]) -> list[dict]:
    _load_catalog()
    return [r for r in results if r["name"] in _catalog_names]


def lookup_by_names(names: list[str]) -> list[dict]:
    _load_catalog()
    return [_catalog_names[n] for n in names if n in _catalog_names]


def detect_test_names(query: str) -> list[str]:
    _load_catalog()
    ql = query.lower()
    q_words = set(ql.split())
    acronyms = set(re.findall(r'\b[A-Z]{2,}\b', query))
    is_short_query = len(q_words) <= 4
    matched = set()
    for name in _catalog_names:
        nl = name.lower()
        if nl in ql:
            matched.add(name)
            continue
        if is_short_query:
            first_word = nl.split()[0] if nl.split() else ""
            if first_word in q_words and len(first_word) > 1:
                matched.add(name)
                continue
        name_acronyms = set(re.findall(r'\b[A-Z]{2,}\b', name))
        if name_acronyms and name_acronyms.issubset(acronyms):
            matched.add(name)
    return list(matched)


def search(query: str, k: int = 10, filters: dict | None = None, alpha: float | None = None) -> list[dict]:
    model = _get_model()
    client = _get_client()
    vec = model.encode(query).tolist()
    fields = ["name", "type", "url", "description", "languages", "job_levels", "industries", "propositions", "training_levels"]
    collection = client.collections.get("Assessment")
    wf = _build_filters(filters or {})
    alpha = alpha if alpha is not None else smart_alpha(query)

    response = collection.query.hybrid(
        query=query,
        vector=vec,
        alpha=alpha,
        limit=k * 2,
        filters=wf,
        return_properties=fields,
    )

    results = [{**o.properties, "_vector_score": o.metadata.score} for o in response.objects]
    if not results:
        return []

    bm25 = _get_bm25()
    for r in results:
        r["_bm25"] = bm25.score(query, r["name"])
    results.sort(key=lambda r: r["_bm25"], reverse=True)
    results = results[:k]
    if not results:
        return []

    ce = _get_ce()
    pairs = [(query, f"{r.get('name','')} {r.get('description','')}") for r in results]
    ce_scores = ce.predict(pairs, convert_to_tensor=False)
    ce_scores = [float(s) if hasattr(s, '__float__') else float(s[0]) for s in ce_scores]

    for r, s in zip(results, ce_scores):
        r["_ce_score"] = s

    results.sort(key=lambda r: r["_ce_score"], reverse=True)

    return grounding_verify(results)
