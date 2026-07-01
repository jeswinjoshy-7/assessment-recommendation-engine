"""
Validate embedding/retrieval quality against known persona-to-assessment mappings.

Run:  python validate.py            (requires Weaviate running)
      python validate.py --reindex  (re-runs embedder first)
"""

import sys

TEST_CASES = [
    {
        "query": "Java coding test for mid-level developer",
        "skills": ["Java"],
        "seniority": "Professional",
        "expected_keywords": ["Java", "Coding", "Development"],
    },
    {
        "query": "Sales assessment for managers",
        "skills": ["Sales"],
        "seniority": "Front Line Manager",
        "expected_keywords": ["Sales", "Account Manager"],
    },
    {
        "query": "OPQ personality test",
        "expected_keywords": ["OPQ", "Personality"],
        "acronym_match": True,
    },
    {
        "query": "Customer service test for entry level",
        "skills": ["Customer Service"],
        "seniority": "Entry-Level",
        "expected_keywords": ["Customer Service", "Contact Center", "Contact Centre"],
    },
]


def test_retrieval():
    from retriever import search, detect_test_names, lookup_by_names, smart_alpha

    passed, total = 0, 0

    for tc in TEST_CASES:
        total += 1
        q = tc["query"]
        alpha = smart_alpha(q)
        results = search(q, k=10, alpha=alpha)

        if not results:
            print(f"  FAIL [{q[:40]:40s}] no results returned")
            continue

        top_names = [r["name"] for r in results]
        keywords = tc.get("expected_keywords", [])
        found_kw = any(any(kw.lower() in n.lower() for kw in keywords) for n in top_names)

        if tc.get("acronym_match") and detect_test_names(q):
            print(f"  PASS [{q[:40]:40s}] acronym detected, {len(results)} results")
        elif found_kw:
            print(f"  PASS [{q[:40]:40s}] keywords {keywords} found in top-10")
        else:
            print(f"  WARN [{q[:40]:40s}] no expected keywords surfaced: {top_names[:3]}")

        passed += 1 if found_kw or tc.get("acronym_match") else 0

    print(f"\nResults: {passed}/{total} passed")
    return passed == total


def test_lookup():
    from retriever import lookup_by_names, detect_test_names

    for q in ["OPQ", "Verify Calculation", "Occupational Personality"]:
        names = detect_test_names(q)
        if names:
            results = lookup_by_names(names)
            status = "OK" if results else "EMPTY"
            print(f"  {status} [{q:40s}] → {len(names)} names, {len(results)} results")
        else:
            print(f"  SKIP [{q:40s}] no names detected")


def test_grounding():
    from retriever import grounding_verify

    fake = [{"name": "Fake Test", "url": "https://fake.com"}]
    verified = grounding_verify(fake)
    assert len(verified) == 0, "Grounding should filter fake results"
    print("  PASS Grounding filters non-catalog results")


def test_cumulative():
    from app import _extract_constraints, is_context_sufficient
    from retriever import build_cumulative_query

    c1 = _extract_constraints("I need a test", {})
    assert not is_context_sufficient(c1), "Vague should not be ready"
    print("  PASS Gated: vague query withheld")

    c2 = _extract_constraints("Java coding test for mid-level developer", c1)
    assert is_context_sufficient(c2), "Specific should be ready"
    q = build_cumulative_query(c2)
    assert "Java" in q, f"Cumulative query should include Java: {q}"
    print("  PASS Cumulative: query built from constraints")

    c3 = _extract_constraints("Also need it in Spanish", c2)
    assert c3.get("language") == "es-ES", "Language not extracted"
    print("  PASS Refinement: constraints updated")


def main():
    print("=" * 60)
    print("VALIDATION: Embedding & Retrieval Quality")
    print("=" * 60)

    if "--reindex" in sys.argv:
        print("\nRe-indexing embeddings...")
        from embedder import main as embed
        embed()

    print("\n--- Retrieval Tests ---")
    test_retrieval()

    print("\n--- Targeted Lookup Tests ---")
    test_lookup()

    print("\n--- Grounding Tests ---")
    test_grounding()

    print("\n--- Conversation Flow Tests ---")
    test_cumulative()

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")


if __name__ == "__main__":
    main()
