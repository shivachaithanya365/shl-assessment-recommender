"""Build a compact local SHL catalog for the recommender.

The source CSV is a public scrape of SHL Individual Test Solutions containing
name, catalog URL, type, duration, remote testing, and adaptive flags.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen


SOURCE_URL = (
    "https://raw.githubusercontent.com/singhsourav0/SHL_Recommendation/main/"
    "rag_recommender/data/assessments.csv"
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "catalog.json"

TYPE_CODES = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgement": "B",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

DISPLAY_OVERRIDES = {
    "Entry Level Customer Serv-Retail & Contact Center": "Entry Level Customer Serv - Retail & Contact Center",
    "Verify Interactive - Numerical Reasoning": "SHL Verify Interactive - Numerical Reasoning",
    "Verify Interactive - Verbal Reasoning": "SHL Verify Interactive - Verbal Reasoning",
    "Verify Interactive - Inductive Reasoning": "SHL Verify Interactive - Inductive Reasoning",
    "Verify Interactive - Deductive Reasoning": "SHL Verify Interactive - Deductive Reasoning",
    "SVAR - Spoken English (US)  (New)": "SVAR Spoken English (US) (New)",
    "SVAR - Spoken English (Indian Accent)  (New)": "SVAR Spoken English (Indian Accent) (New)",
    "SVAR - Spoken English (U.K.)": "SVAR Spoken English (UK)",
    "SVAR - Spoken English (AUS)": "SVAR Spoken English (AUS)",
    "SVAR - Spoken French (Canadian) (New)": "SVAR Spoken French (Canadian) (New)",
    "SVAR - Spoken French (European) (New)": "SVAR Spoken French (European) (New)",
    "SVAR - Spoken Spanish (Castilian) (New)": "SVAR Spoken Spanish (Castilian) (New)",
    "SVAR - Spoken Spanish (North American) (New)": "SVAR Spoken Spanish (North American) (New)",
}

TYPE_OVERRIDES = {
    "Global Skills Development Report": ["D"],
    "OPQ Leadership Report": ["P"],
    "OPQ Universal Competency Report 1.0": ["P"],
    "OPQ Universal Competency Report 2.0": ["P"],
    "OPQ MQ Sales Report": ["P"],
    "Sales Transformation 1.0 - Individual Contributor": ["P"],
    "Sales Transformation 2.0 - Individual Contributor": ["P"],
    "Sales Transformation Report 1.0 - Sales Manager": ["P"],
    "Sales Transformation Report 2.0 - Sales Manager": ["P"],
    "SHL Verify Interactive - Inductive Reasoning": ["A"],
    "Verify Interactive - Numerical Reasoning": ["A", "S"],
    "Microsoft Excel 365 (New)": ["K", "S"],
    "Microsoft Excel 365 - Essentials (New)": ["K", "S"],
    "Microsoft Word 365 (New)": ["K", "S"],
    "Microsoft Word 365 - Essentials (New)": ["K", "S"],
    "Customer Service Phone Simulation": ["B", "S"],
    "Sales & Service Phone Simulation": ["B", "S"],
    "Entry Level Customer Serv-Retail & Contact Center": ["P", "C"],
    "Entry Level Customer Serv - Retail & Contact Center": ["P", "C"],
}

URL_OVERRIDES = {
    "Verify Interactive - Numerical Reasoning": "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-numerical-reasoning/",
}


def fetch_csv() -> str:
    request = Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8-sig")


def codes(value: str, name: str) -> list[str]:
    if name in TYPE_OVERRIDES:
        return TYPE_OVERRIDES[name]
    name_l = name.lower()
    if "verify interactive" in name_l and "numerical reasoning" in name_l:
        return ["A", "S"]
    if "verify interactive" in name_l and ("deductive reasoning" in name_l or "inductive reasoning" in name_l):
        return ["A"]
    found: list[str] = []
    for label, code in TYPE_CODES.items():
        if label.lower() in value.lower() and code not in found:
            found.append(code)
    if not found and "simulation" in name.lower():
        found.append("S")
    return found


def normalize_url(url: str) -> str:
    url = url.strip().replace("/solutions/products/product-catalog/", "/products/product-catalog/")
    if not url.endswith("/"):
        url += "/"
    return url


def duration(value: str) -> str:
    value = value.replace("Approximate Completion Time in minutes =", "").strip()
    if not value:
        return ""
    return value


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9+#.]+", text.lower())
    return [w for w in words if len(w) > 1]


def keywords(name: str, type_codes: list[str]) -> list[str]:
    kws = set(tokenize(name))
    lower = name.lower()
    additions = {
        "java": ["backend", "developer", "software", "programming"],
        "spring": ["java", "backend", "microservices"],
        "rest": ["api", "web services", "backend"],
        "sql": ["database", "data", "relational"],
        "aws": ["cloud", "deployment"],
        "docker": ["container", "devops", "cloud native"],
        "angular": ["frontend", "javascript", "typescript"],
        "react": ["frontend", "javascript"],
        "excel": ["spreadsheet", "office", "admin"],
        "word": ["office", "admin", "document"],
        "hipaa": ["healthcare", "patient", "security", "compliance"],
        "medical": ["healthcare", "patient"],
        "accounting": ["finance", "analyst"],
        "statistics": ["finance", "data", "analyst"],
        "svar": ["spoken", "language", "accent", "call center"],
        "contact center": ["call center", "customer service", "inbound"],
        "customer service": ["contact center", "support"],
        "safety": ["dependability", "plant", "industrial", "manufacturing"],
        "opq": ["personality", "behavior", "leadership", "fit"],
        "graduate": ["entry level", "campus", "trainee"],
        "global skills": ["reskill", "skills audit", "development"],
        "sales": ["selling", "commercial", "account"],
        "linux": ["systems", "infrastructure"],
        "networking": ["infrastructure", "systems"],
        "live coding": ["coding", "interview", "developer"],
    }
    for needle, vals in additions.items():
        if needle in lower:
            kws.update(tokenize(" ".join(vals)))
    for code in type_codes:
        if code == "A":
            kws.update(["cognitive", "reasoning", "aptitude"])
        elif code == "P":
            kws.update(["personality", "behavior"])
        elif code == "K":
            kws.update(["knowledge", "skills"])
        elif code == "S":
            kws.update(["simulation"])
        elif code == "B":
            kws.update(["situational", "judgment", "sjt"])
        elif code == "C":
            kws.update(["competency", "competencies"])
    return sorted(kws)


def main() -> None:
    rows = list(csv.DictReader(io.StringIO(fetch_csv())))
    products: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    for row in rows:
        raw_name = row["Assessment Name"].strip()
        name = DISPLAY_OVERRIDES.get(raw_name, raw_name)
        url = URL_OVERRIDES.get(raw_name, normalize_url(row["Relative URL"]))
        if not name or not url or url in seen_urls:
            continue
        seen_urls.add(url)
        type_codes = codes(row.get("Test Type", ""), raw_name)
        products.append(
            {
                "name": name,
                "source_name": raw_name,
                "url": url,
                "test_type": ",".join(type_codes),
                "type_codes": type_codes,
                "duration": duration(row.get("Assessment Length", "")),
                "remote_testing": row.get("Remote Testing", "").strip().lower() == "yes",
                "adaptive": row.get("Adaptive/IRT", "").strip().lower() == "yes",
                "keywords": keywords(name, type_codes),
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"source": SOURCE_URL, "products": products}, indent=2), encoding="utf-8")
    print(f"Wrote {len(products)} products to {OUT}")


if __name__ == "__main__":
    main()
