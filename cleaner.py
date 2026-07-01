import json
import re
import os
from collections import Counter

INPUT_PATH = os.path.join("data", "catalog.json")
OUTPUT_PATH = os.path.join("data", "catalog_clean.json")


def slugify(name):
    slug = re.sub(r'[^a-zA-Z0-9\s-]', '', name).strip().lower()
    slug = re.sub(r'[\s-]+', '-', slug)
    return slug


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} products")

    before = len(data)
    data = [p for p in data if not p['job_levels'] or 'NULL' not in ','.join(p['job_levels'])]
    removed = before - len(data)
    print(f"Removed {removed} report/profile/pack variants (NULL job_levels)")

    for p in data:
        p['job_levels'] = [j for j in p['job_levels'] if j != 'NULL']

    slugs = [slugify(p['name']) for p in data]
    slug_counts = Counter(slugs)
    collisions = {s for s, c in slug_counts.items() if c > 1}

    collision_counters = {}
    for p in data:
        base = slugify(p['name'])
        if base in collisions:
            collision_counters[base] = collision_counters.get(base, 0) + 1
            idx = collision_counters[base]
            p['url'] = f"https://www.shl.com/products/assessments/{base}-{idx}"
        else:
            p['url'] = f"https://www.shl.com/products/assessments/{base}"

    print(f"Fixed {len(collisions)} collision groups: {collisions}")

    urls = [p['url'] for p in data]
    url_dupes = {u for u in urls if urls.count(u) > 1}
    names = [p['name'] for p in data]
    name_dupes = {n for n in names if names.count(n) > 1}

    assert not name_dupes, f"Duplicate names remain: {name_dupes}"
    assert not url_dupes, f"Duplicate URLs remain: {url_dupes}"

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n=== CLEAN SUMMARY ===")
    print(f"Final count: {len(data)} products")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Duplicate names: {len(name_dupes)}")
    print(f"Duplicate URLs: {len(url_dupes)}")

    has_industries = sum(1 for p in data if p['industries'])
    has_propositions = sum(1 for p in data if p['propositions'])
    has_training = sum(1 for p in data if p['training_levels'])
    print(f"Products with industry data: {has_industries}/{len(data)}")
    print(f"Products with propositions: {has_propositions}/{len(data)}")
    print(f"Products with training levels: {has_training}/{len(data)}")


if __name__ == "__main__":
    main()
