import cloudscraper
from bs4 import BeautifulSoup
import json
import os
import re

CATALOG_URL = "https://online.shl.com/gb/en-us/products?producttypes=1"
SCRAPER = cloudscraper.create_scraper()

INDUSTRY_MAP = {
    "1": "Safety/Public",
    "2": "Retail",
    "3": "Manufacturing",
    "4": "Healthcare",
    "5": "Customer Service",
    "6": "Telecommunications",
    "7": "Banking/Finance",
    "8": "Insurance",
    "9": "Hospitality",
}

PROPOSITION_MAP = {
    "1": "Entry-Level",
    "2": "Graduate",
    "3": "Professional",
    "4": "Talent Audit",
    "5": "Succession Planning",
    "6": "Development",
    "7": "Director",
    "8": "Executive",
    "9": "Front Line Manager",
}

TRAINING_MAP = {
    "1": "Basic",
    "2": "Intermediate",
    "3": "Advanced",
    "4": "Expert",
}


def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()


def parse_languages(text):
    codes = [c.strip() for c in text.split(",") if c.strip()]
    seen = set()
    unique = []
    for code in codes:
        norm = code.lower()
        if norm not in seen:
            seen.add(norm)
            unique.append(code)
    return unique


def parse_ids(text):
    return [i.strip() for i in text.split(",") if i.strip() and i.strip() != "NULL"]


def scrape_catalog():
    print(f"Fetching {CATALOG_URL}...")
    resp = SCRAPER.get(CATALOG_URL, timeout=60)
    resp.raise_for_status()
    print(f"Got {len(resp.text)} bytes")

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", id="myTable")
    if not table:
        raise RuntimeError("Could not find table#myTable in page")

    rows = table.find_all("tr")
    print(f"Found {len(rows)} table rows (including header)")

    products = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 9:
            continue

        img = cols[0].find("img")
        product_type = "Assessment"
        if img:
            alt = clean_text(img.get("alt", ""))
            if "report" in alt.lower():
                product_type = "Report"

        name_tag = cols[1].find("b")
        if not name_tag:
            continue
        name = clean_text(name_tag.get_text())

        desc_td = cols[2]
        description = clean_text(desc_td.get_text()) if desc_td else ""

        languages_raw = clean_text(cols[3].get_text())
        languages = parse_languages(languages_raw)

        job_level_ids = parse_ids(cols[4].get_text())
        job_level_names = [PROPOSITION_MAP.get(j, j) for j in job_level_ids]

        industry_ids = parse_ids(cols[5].get_text())
        industry_names = [INDUSTRY_MAP.get(i, i) for i in industry_ids]

        proposition_ids = parse_ids(cols[6].get_text())
        proposition_names = [PROPOSITION_MAP.get(p, p) for p in proposition_ids]

        product_type_id = cols[7].get_text(strip=True)

        training_ids = parse_ids(cols[8].get_text())
        training_names = [TRAINING_MAP.get(t, t) for t in training_ids]

        url_slug = re.sub(r'[^a-zA-Z0-9\s-]', '', name).strip().lower()
        url_slug = re.sub(r'[\s-]+', '-', url_slug)
        product_url = f"https://www.shl.com/products/assessments/{url_slug}"

        products.append({
            "name": name,
            "type": product_type,
            "url": product_url,
            "catalog_url": CATALOG_URL,
            "description": description,
            "languages": languages,
            "job_levels": job_level_names,
            "industries": industry_names,
            "propositions": proposition_names,
            "training_levels": training_names,
        })

    print(f"Scraped {len(products)} products")
    return products


def main():
    os.makedirs("data", exist_ok=True)
    products = scrape_catalog()
    path = os.path.join("data", "catalog.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(products)} products to {path}")

    for p in products[:3]:
        print(f"  - {p['name']} ({p['type']})")
        print(f"    propositions={p['propositions']} training={p['training_levels']}")


if __name__ == "__main__":
    main()
