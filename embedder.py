import json
import os
import re
import weaviate
from weaviate.classes.config import Property, DataType, Configure

DATA_PATH = os.path.join("data", "catalog_clean.json")


def extract_acronyms(name: str) -> list[str]:
    seen = set()
    result = []
    for m in re.findall(r'\(([^)]+)\)', name):
        for part in re.split(r'[\s\-–—/]+', m):
            part = part.strip().rstrip(".").rstrip(",")
            if len(part) >= 2 and part not in seen:
                seen.add(part)
                result.append(part)
    for word in name.split():
        word_clean = word.strip("().,-")
        if len(word_clean) >= 2 and word_clean.isupper() and word_clean not in seen:
            seen.add(word_clean)
            result.append(word_clean)
    return result


LEVEL_ENRICHMENT = {
    "Entry-Level": ["entry level", "junior", "fresh", "early career", "new hire"],
    "Graduate": ["graduate", "university", "college", "recent grad"],
    "Professional": ["professional", "experienced", "mid-level", "individual contributor"],
    "Senior": ["senior", "lead", "expert", "seasoned"],
    "Front Line Manager": ["manager", "supervisor", "team lead", "frontline"],
    "Director": ["director", "head", "senior leadership"],
    "Executive": ["executive", "vp", "c-suite", "senior executive"],
    "Talent Audit": ["talent audit", "assessment", "benchmarking"],
    "Succession Planning": ["succession planning", "leadership pipeline", "high potential"],
    "Development": ["development", "growth", "coaching", "learning"],
}


def enrich_text(product: dict) -> str:
    name = product["name"]
    desc = product["description"]
    test_type = product["type"]
    parts = [name, desc, f"Type: {test_type}"]

    if product.get("job_levels"):
        levels_str = ", ".join(product["job_levels"])
        parts.append(f"Job Levels: {levels_str}")
        enrichment_terms = []
        for level in product["job_levels"]:
            enrichment_terms.extend(LEVEL_ENRICHMENT.get(level, []))
        if enrichment_terms:
            parts.append("Keywords: " + ", ".join(enrichment_terms))

    if product.get("industries"):
        parts.append(f"Industries: {', '.join(product['industries'])}")

    if product.get("languages"):
        parts.append(f"Languages: {', '.join(product['languages'])}")

    if product.get("propositions"):
        parts.append(f"Propositions: {', '.join(product['propositions'])}")

    if product.get("training_levels"):
        parts.append(f"Training Levels: {', '.join(product['training_levels'])}")

    acronyms = extract_acronyms(name)
    if acronyms:
        parts.append(f"Acronyms: {', '.join(acronyms)}")

    return "\n".join(parts)


def get_client():
    host = os.getenv("WEAVIATE_HOST", "localhost")
    return weaviate.connect_to_local(host=host)


def ensure_schema(client):
    if client.collections.exists("Assessment"):
        client.collections.delete("Assessment")
    client.collections.create(
        name="Assessment",
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="name", data_type=DataType.TEXT),
            Property(name="type", data_type=DataType.TEXT),
            Property(name="url", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
            Property(name="languages", data_type=DataType.TEXT_ARRAY),
            Property(name="job_levels", data_type=DataType.TEXT_ARRAY),
            Property(name="industries", data_type=DataType.TEXT_ARRAY),
            Property(name="acronyms", data_type=DataType.TEXT_ARRAY),
            Property(name="propositions", data_type=DataType.TEXT_ARRAY),
            Property(name="training_levels", data_type=DataType.TEXT_ARRAY),
        ],
    )
    print("Schema 'Assessment' created")


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        products = json.load(f)
    print(f"Loaded {len(products)} products")

    docs = [enrich_text(p) for p in products]
    print(f"Built {len(docs)} enriched documents")

    print("Loading Jina embedding model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Embedding dim: {dim}")

    print("Generating embeddings...")
    embeddings = model.encode(docs, show_progress_bar=True, batch_size=32)
    print(f"Generated {len(embeddings)} embeddings, shape: {embeddings.shape}")

    print("Connecting to Weaviate...")
    client = get_client()
    ensure_schema(client)

    print("Indexing into Weaviate...")
    collection = client.collections.get("Assessment")
    with collection.batch.dynamic() as batch:
        for i, (p, vec) in enumerate(zip(products, embeddings)):
            batch.add_object(
                properties={
                    "name": p["name"],
                    "type": p["type"],
                    "url": p["url"],
                    "description": p["description"],
                    "languages": p.get("languages", []),
                    "job_levels": p.get("job_levels", []),
                    "industries": p.get("industries", []),
                    "acronyms": extract_acronyms(p["name"]),
                    "propositions": p.get("propositions", []),
                    "training_levels": p.get("training_levels", []),
                },
                vector=vec.tolist(),
            )
            if (i + 1) % 50 == 0:
                print(f"  Indexed {i + 1}/{len(products)}")
    print(f"Indexed {len(products)} products into Weaviate")

    sample_idx = 0
    acr = extract_acronyms(products[sample_idx]["name"])
    first = docs[0].split("\n")[0]
    print(f"\nSample: {first}")
    print(f"Acronyms: {acr}")

    print(f"\n=== INDEXING COMPLETE ===")
    print(f"Products: {len(products)}")
    print(f"Dimension: {dim}")


if __name__ == "__main__":
    main()
