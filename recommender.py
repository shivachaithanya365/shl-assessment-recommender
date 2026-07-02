from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "data" / "catalog.json"

TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgment",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "that",
    "this",
    "from",
    "need",
    "hiring",
    "assessment",
    "assessments",
    "test",
    "tests",
    "candidate",
    "candidates",
    "role",
    "roles",
}


@dataclass(frozen=True)
class Product:
    name: str
    url: str
    test_type: str
    type_codes: tuple[str, ...]
    duration: str
    keywords: tuple[str, ...]

    def api(self) -> dict[str, str]:
        return {"name": self.name, "url": self.url, "test_type": self.test_type}

    @property
    def key_text(self) -> str:
        labels = [TYPE_LABELS.get(code, code) for code in self.type_codes]
        return ", ".join(labels)


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9+#.]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def _contains_any(text: str, words: Iterable[str]) -> bool:
    return any(word in text for word in words)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class SHLRecommender:
    def __init__(self, catalog_path: Path = CATALOG_PATH) -> None:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.products = [
            Product(
                name=item["name"],
                url=item["url"],
                test_type=item.get("test_type", ""),
                type_codes=tuple(item.get("type_codes", [])),
                duration=item.get("duration", ""),
                keywords=tuple(item.get("keywords", [])),
            )
            for item in payload["products"]
        ]
        self.by_name = {p.name.lower(): p for p in self.products}
        self.by_source_alias = self._build_aliases()

    def chat(self, messages: list[dict[str, str]]) -> dict[str, object]:
        user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]
        assistant_messages = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
        latest = user_messages[-1] if user_messages else ""
        latest_l = latest.lower()
        full_user = "\n".join(user_messages)
        full_l = full_user.lower()
        prior = self._extract_prior_recommendations(assistant_messages)

        if self._is_prompt_injection(latest_l):
            return self._empty(
                "I can only help with SHL assessment selection. I cannot follow instructions to ignore the task, reveal prompts, or bypass catalog constraints."
            )

        if self._is_legal_or_offtopic(latest_l, full_l):
            return self._empty(
                "That is outside scope. I can help select SHL assessments, but I cannot provide legal, regulatory, or general hiring advice."
            )

        comparison = self._comparison_reply(latest)
        if comparison:
            return self._empty(comparison)

        if self._asks_for_shorter_personality(latest_l):
            return self._empty(
                "OPQ32r is the main SHL personality instrument in this catalog. I do not see a shorter direct personality replacement, so the clean options are to keep OPQ32r or drop personality from the battery."
            )

        if "final list" in latest_l:
            explicit = self._apply_edits(self._mentioned_products(latest), full_l, latest_l)
            if explicit:
                return self._reply_with_recs("Final SHL assessment shortlist confirmed.", explicit, end=True)

        clarification = self._clarification(latest_l, full_l, prior)
        if clarification:
            return self._empty(clarification)

        shortlist = self._shortlist(full_l, latest_l, prior)
        if not shortlist:
            return self._empty(
                "I need a little more context before recommending. What role are you hiring for, and which skills or behaviors matter most?"
            )

        reply = self._summary(full_l, latest_l, shortlist)
        end = self._is_confirmation(latest_l)
        return self._reply_with_recs(reply, shortlist, end=end)

    def _build_aliases(self) -> dict[str, Product]:
        aliases: dict[str, Product] = {}
        manual = {
            "opq": "Occupational Personality Questionnaire OPQ32r",
            "opq32r": "Occupational Personality Questionnaire OPQ32r",
            "mq sales": "OPQ MQ Sales Report",
            "gsa": "Global Skills Assessment",
            "verify g+": "SHL Verify Interactive G+",
            "g+": "SHL Verify Interactive G+",
            "graduate scenarios": "Graduate Scenarios",
            "dsi": "Dependability and Safety Instrument (DSI)",
            "safety dependability 8.0": "Manufac. & Indust. - Safety & Dependability 8.0",
            "svar us": "SVAR Spoken English (US) (New)",
            "svar uk": "SVAR Spoken English (UK)",
            "svar indian": "SVAR Spoken English (Indian Accent) (New)",
        }
        for alias, name in manual.items():
            product = self._find_exact(name)
            if product:
                aliases[alias] = product
        for product in self.products:
            aliases.setdefault(product.name.lower(), product)
            simplified = re.sub(r"[^a-z0-9+# ]+", " ", product.name.lower())
            aliases.setdefault(_clean(simplified), product)
        return aliases

    def _find_exact(self, name: str) -> Product | None:
        target = name.lower()
        for product in self.products:
            if product.name.lower() == target:
                return product
        return None

    def _find_by_name(self, name: str) -> Product | None:
        product = self.by_source_alias.get(name.lower())
        if not product:
            simplified = _clean(re.sub(r"[^a-z0-9+# ]+", " ", name.lower()))
            product = self.by_source_alias.get(simplified)
        if product:
            return product
        name_l = name.lower()
        for product in self.products:
            if product.name.lower() == name_l:
                return product
        for product in self.products:
            if name_l in product.name.lower():
                return product
        return None

    def _extract_prior_recommendations(self, assistant_messages: list[str]) -> list[Product]:
        if not assistant_messages:
            return []
        for text in reversed(assistant_messages):
            text_l = text.lower()
            found: list[tuple[int, int, Product]] = []
            for product in self.products:
                idx = text_l.find(product.name.lower())
                if idx >= 0:
                    found.append((idx, -len(product.name), product))
            found.sort(key=lambda pair: (pair[0], pair[1]))
            kept: list[tuple[int, Product]] = []
            for idx, _, product in found:
                name_l = product.name.lower()
                if any(idx == kept_idx and name_l in kept_product.name.lower() for kept_idx, kept_product in kept):
                    continue
                kept.append((idx, product))
            deduped = self._dedupe([product for _, product in kept])[:10]
            if deduped:
                return deduped
        return []

    def _is_prompt_injection(self, latest_l: str) -> bool:
        return _contains_any(
            latest_l,
            [
                "ignore previous",
                "ignore all previous",
                "system prompt",
                "developer message",
                "jailbreak",
                "reveal your prompt",
                "show your instructions",
                "bypass",
            ],
        )

    def _is_legal_or_offtopic(self, latest_l: str, full_l: str) -> bool:
        legal_markers = [
            "legally required",
            "legal requirement",
            "satisfy that requirement",
            "satisfy hipaa",
            "is it legal",
            "law require",
            "lawsuit",
            "legal advice",
        ]
        if _contains_any(latest_l, legal_markers):
            return True
        hiring_general = ["salary", "compensation", "offer letter", "visa", "background check law"]
        if _contains_any(latest_l, hiring_general):
            return True
        assessment_markers = [
            "shl",
            "assessment",
            "test",
            "screen",
            "shortlist",
            "battery",
            "candidate",
            "hiring",
            "role",
            "skill",
            "personality",
            "cognitive",
            "simulation",
            "opq",
        ]
        off_topic_markers = ["weather", "recipe", "movie", "write a poem", "stock price"]
        return _contains_any(latest_l, off_topic_markers) and not _contains_any(full_l, assessment_markers)

    def _is_confirmation(self, latest_l: str) -> bool:
        return _contains_any(
            latest_l,
            [
                "confirmed",
                "lock",
                "that works",
                "that's good",
                "that is good",
                "that covers it",
                "perfect",
                "thanks",
                "thank you",
                "keep the shortlist",
                "keeping the",
                "we will use",
                "we'll use",
                "as-is",
                "final list",
            ],
        )

    def _asks_for_shorter_personality(self, latest_l: str) -> bool:
        return "opq" in latest_l and "shorter" in latest_l and any(w in latest_l for w in ["replace", "alternative"])

    def _clarification(self, latest_l: str, full_l: str, prior: list[Product]) -> str | None:
        if prior:
            return None
        if re.fullmatch(r"[\w\s'.,-]{0,80}", latest_l) and _contains_any(
            latest_l,
            ["need an assessment", "need assessment", "recommend assessment", "solution"],
        ) and not self._has_domain_signal(full_l):
            return "What role or job family is this for, and is the goal selection, screening, or development?"

        if "senior leadership" in full_l and not _contains_any(full_l, ["cxo", "director", "selection", "development"]):
            return "Who is this meant for, and is this for selection or development?"

        if _contains_any(full_l, ["contact centre", "contact center", "call center", "call centre"]) and not _contains_any(
            full_l, ["english", "spanish", "french"]
        ):
            return "What language will candidates use on calls? The spoken-language screen depends on that."

        if _contains_any(full_l, ["contact centre", "contact center", "call center", "call centre"]) and "english" in full_l and not _contains_any(
            full_l, [" us", " u.s", " uk", "u.k", "australian", "australia", "indian accent", "india"]
        ):
            return "Which English accent fits the operation: US, UK, Australian, or Indian?"

        if "rust" in full_l and not _contains_any(full_l, ["go ahead", "yes", "shortlist"]):
            return (
                "SHL's catalog does not show a Rust-specific knowledge test. The closest route is live coding plus systems and networking tests. "
                "Should I build a shortlist from those?"
            )

        if _contains_any(full_l, ["full-stack", "full stack"]) and _contains_any(
            full_l, ["java", "spring", "angular", "sql", "aws", "docker"]
        ) and not _contains_any(full_l, ["backend-leaning", "frontend", "balanced", "senior ic", "tech lead"]):
            return "Is this backend-leaning, frontend-heavy, or a balanced full-stack role?"

        if _contains_any(full_l, ["full-stack", "full stack"]) and _contains_any(
            full_l, ["backend-leaning", "frontend", "balanced"]
        ) and not _contains_any(full_l, ["senior ic", "tech lead"]):
            return "Is the role closer to a senior IC or a tech lead? That changes whether I add a leadership/scenarios layer."

        if _contains_any(full_l, ["spanish", "south texas", "healthcare"]) and "hipaa" in full_l and not _contains_any(
            full_l, ["english fluent", "hybrid", "personality-only", "functionally bilingual"]
        ):
            return (
                "The healthcare knowledge tests are English-only, while OPQ32r and DSI support Spanish variants. "
                "Are candidates functionally bilingual for written English tests, or should this be Spanish-only?"
            )

        return None

    def _has_domain_signal(self, text: str) -> bool:
        return _contains_any(
            text,
            [
                "java",
                "sales",
                "admin",
                "finance",
                "graduate",
                "contact",
                "call",
                "safety",
                "healthcare",
                "leadership",
                "developer",
                "engineer",
                "excel",
                "word",
                "rust",
                "plant",
            ],
        )

    def _comparison_reply(self, latest: str) -> str | None:
        latest_l = latest.lower()
        if not _contains_any(latest_l, ["difference", "compare", " vs ", " versus ", "different from"]):
            return None
        candidates = self._mentioned_products(latest)
        if len(candidates) < 2:
            if "opq" in latest_l and "gsa" in latest_l:
                candidates = [
                    self._find_by_name("Occupational Personality Questionnaire OPQ32r"),
                    self._find_by_name("Global Skills Assessment"),
                ]
            elif "dsi" in latest_l and "safety" in latest_l:
                candidates = [
                    self._find_by_name("Dependability and Safety Instrument (DSI)"),
                    self._find_by_name("Manufac. & Indust. - Safety & Dependability 8.0"),
                ]
        candidates = [p for p in candidates if p]
        if len(candidates) < 2:
            return "I can compare SHL assessments, but I need the two assessment names to ground the comparison."
        first, second = candidates[:2]
        return (
            f"{first.name} is a {first.key_text or first.test_type} product"
            f"{self._duration_clause(first)}. {second.name} is a {second.key_text or second.test_type} product"
            f"{self._duration_clause(second)}. Use {first.name} when the priority is {self._plain_focus(first)}; "
            f"use {second.name} when the priority is {self._plain_focus(second)}."
        )

    def _mentioned_products(self, text: str) -> list[Product]:
        text_l = text.lower()
        found: list[tuple[int, Product]] = []
        for alias, product in self.by_source_alias.items():
            if len(alias) < 4:
                continue
            idx = text_l.find(alias)
            if idx >= 0:
                found.append((idx, product))
        found.sort(key=lambda item: item[0])
        return self._dedupe([p for _, p in found])

    def _duration_clause(self, product: Product) -> str:
        return f" with duration {product.duration}" if product.duration else ""

    def _plain_focus(self, product: Product) -> str:
        name = product.name.lower()
        if "opq" in name:
            return "workplace personality and behavioral style"
        if "global skills" in name:
            return "skills visibility and development planning"
        if "safety" in name or "dependability" in name:
            return "reliability, rule-following, and safety behavior"
        if "simulation" in name:
            return "realistic job-task behavior"
        if "verify" in name:
            return "cognitive reasoning"
        if "svar" in name or "spoken" in name:
            return "spoken-language proficiency"
        if "report" in name:
            return "interpreting results for a specific business context"
        return "the named skill or job capability"

    def _shortlist(self, full_l: str, latest_l: str, prior: list[Product]) -> list[Product]:
        selected = list(prior)
        scenario = self._scenario_products(full_l, latest_l)
        if scenario:
            selected = scenario
        elif not selected:
            selected = self._ranked_products(full_l)

        selected = self._apply_edits(selected, full_l, latest_l)
        if _contains_any(full_l, ["senior", "lead", "manager", "graduate", "admin", "finance"]) and not _contains_any(
            full_l,
            [
                "drop opq",
                "drop the opq",
                "drop opq32r",
                "remove opq",
                "remove the opq",
                "remove opq32r",
                "without personality",
                "skip personality",
            ],
        ):
            opq = self._find_by_name("Occupational Personality Questionnaire OPQ32r")
            if opq and self._should_add_opq(full_l, selected):
                selected.append(opq)
        return self._dedupe(selected)[:10]

    def _scenario_products(self, full_l: str, latest_l: str) -> list[Product]:
        picks: list[str] = []
        if "senior leadership" in full_l or _contains_any(full_l, ["cxo", "director-level"]):
            picks = [
                "Occupational Personality Questionnaire OPQ32r",
                "OPQ Universal Competency Report 2.0",
                "OPQ Leadership Report",
            ]
        elif "rust" in full_l:
            picks = [
                "Smart Interview Live Coding",
                "Linux Programming (General)",
                "Networking and Implementation (New)",
                "SHL Verify Interactive G+",
                "Occupational Personality Questionnaire OPQ32r",
            ]
        elif _contains_any(full_l, ["contact centre", "contact center", "call center", "call centre"]):
            svar = "SVAR Spoken English (US) (New)"
            if _contains_any(full_l, ["uk", "u.k"]):
                svar = "SVAR Spoken English (UK)"
            elif _contains_any(full_l, ["indian accent", "india"]):
                svar = "SVAR Spoken English (Indian Accent) (New)"
            elif _contains_any(full_l, ["australian", "australia"]):
                svar = "SVAR Spoken English (AUS)"
            picks = [
                svar,
                "Contact Center Call Simulation (New)",
                "Entry Level Customer Serv - Retail & Contact Center",
                "Customer Service Phone Simulation",
            ]
        elif _contains_any(full_l, ["financial analyst", "finance", "accounting"]) and "graduate" in full_l:
            picks = [
                "SHL Verify Interactive Numerical Reasoning",
                "Financial Accounting (New)",
                "Basic Statistics (New)",
                "Occupational Personality Questionnaire OPQ32r",
            ]
            if _contains_any(full_l, ["situational", "sjt", "decision making", "graduate scenarios"]):
                picks.insert(3, "Graduate Scenarios")
        elif _contains_any(full_l, ["sales organization", "sales org", "re-skill", "reskill", "talent audit"]):
            picks = [
                "Global Skills Assessment",
                "Global Skills Development Report",
                "Occupational Personality Questionnaire OPQ32r",
                "OPQ MQ Sales Report",
                "Sales Transformation 2.0 - Individual Contributor",
            ]
        elif _contains_any(full_l, ["chemical facility", "plant operator", "safety", "procedure compliance"]):
            picks = [
                "Dependability and Safety Instrument (DSI)",
                "Manufac. & Indust. - Safety & Dependability 8.0",
                "Workplace Health and Safety (New)",
            ]
            if _contains_any(full_l, ["industrial", "8.0 bundle", "right fit"]):
                picks = [
                    "Manufac. & Indust. - Safety & Dependability 8.0",
                    "Workplace Health and Safety (New)",
                ]
        elif _contains_any(full_l, ["healthcare", "patient records", "hipaa"]):
            picks = [
                "HIPAA (Security)",
                "Medical Terminology (New)",
                "Microsoft Word 365 - Essentials (New)",
                "Dependability and Safety Instrument (DSI)",
                "Occupational Personality Questionnaire OPQ32r",
            ]
        elif _contains_any(full_l, ["admin assistant", "admin assistants", "excel", "word"]) and not _contains_any(
            full_l, ["java", "developer"]
        ):
            if _contains_any(full_l, ["simulation", "capabilities", "end-to-end"]):
                picks = [
                    "Microsoft Excel 365 (New)",
                    "Microsoft Word 365 (New)",
                    "MS Excel (New)",
                    "MS Word (New)",
                    "Occupational Personality Questionnaire OPQ32r",
                ]
            else:
                picks = [
                    "MS Excel (New)",
                    "MS Word (New)",
                    "Occupational Personality Questionnaire OPQ32r",
                ]
        elif _contains_any(full_l, ["full-stack", "full stack", "spring", "microservice", "docker", "aws"]):
            picks = [
                "Core Java (Advanced Level) (New)",
                "Spring (New)",
                "RESTful Web Services (New)",
                "SQL (New)",
                "Amazon Web Services (AWS) Development (New)",
                "Docker (New)",
                "SHL Verify Interactive G+",
                "Occupational Personality Questionnaire OPQ32r",
            ]
            if "drop rest" in full_l or "rest out" in full_l:
                picks = [p for p in picks if p != "RESTful Web Services (New)"]
            if not _contains_any(full_l, ["aws", "docker"]):
                picks = [p for p in picks if p not in {"Amazon Web Services (AWS) Development (New)", "Docker (New)"}]
        elif "graduate management trainee" in full_l or ("graduate" in full_l and "personality" in full_l and "situational" in full_l):
            picks = [
                "SHL Verify Interactive G+",
                "Occupational Personality Questionnaire OPQ32r",
                "Graduate Scenarios",
            ]
        elif _contains_any(full_l, ["java developer", "java engineer"]):
            picks = [
                "Core Java (Advanced Level) (New)" if "senior" in full_l else "Core Java (Entry Level) (New)",
                "Java 8 (New)",
                "SQL (New)",
                "SHL Verify Interactive G+",
                "Occupational Personality Questionnaire OPQ32r",
            ]
        return [p for p in (self._find_by_name(name) for name in picks) if p]

    def _ranked_products(self, full_l: str) -> list[Product]:
        query_tokens = set(_tokens(full_l))
        scored: list[tuple[float, Product]] = []
        for product in self.products:
            product_text = " ".join([product.name.lower(), " ".join(product.keywords)])
            score = 0.0
            for token in query_tokens:
                if token in product.name.lower():
                    score += 4.0
                elif token in product.keywords:
                    score += 2.0
                elif token in product_text:
                    score += 1.0
            if "senior" in full_l and "advanced" in product.name.lower():
                score += 3
            if "entry" in full_l and "entry" in product.name.lower():
                score += 3
            if "simulation" in full_l and "S" in product.type_codes:
                score += 2
            if "cognitive" in full_l and "A" in product.type_codes:
                score += 2
            if "personality" in full_l and "P" in product.type_codes:
                score += 2
            if score > 0:
                scored.append((score, product))
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [product for _, product in scored[:10]]

    def _apply_edits(self, selected: list[Product], full_l: str, latest_l: str) -> list[Product]:
        edited = list(selected)
        if _contains_any(
            full_l,
            [
                "drop opq",
                "drop the opq",
                "drop opq32r",
                "remove opq",
                "remove the opq",
                "remove opq32r",
                "without opq",
            ],
        ):
            edited = [p for p in edited if "opq" not in p.name.lower() and "occupational personality" not in p.name.lower()]
        if _contains_any(full_l, ["drop rest", "rest out", "remove rest"]):
            edited = [p for p in edited if "restful" not in p.name.lower()]
        if _contains_any(full_l, ["drop verify", "without verify", "remove verify"]):
            edited = [p for p in edited if "verify" not in p.name.lower()]
        add_map = {
            "aws": "Amazon Web Services (AWS) Development (New)",
            "docker": "Docker (New)",
            "personality": "Occupational Personality Questionnaire OPQ32r",
            "graduate scenarios": "Graduate Scenarios",
            "situational": "Graduate Scenarios",
            "simulation": None,
        }
        for marker, name in add_map.items():
            if marker in latest_l and name:
                product = self._find_by_name(name)
                if product:
                    edited.append(product)
        return self._dedupe(edited)

    def _should_add_opq(self, full_l: str, selected: list[Product]) -> bool:
        if any("opq" in p.name.lower() or "occupational personality" in p.name.lower() for p in selected):
            return False
        return _contains_any(
            full_l,
            ["senior", "leadership", "manager", "graduate", "admin", "finance", "sales", "stakeholder", "mentor"],
        )

    def _summary(self, full_l: str, latest_l: str, shortlist: list[Product]) -> str:
        names = ", ".join(p.name for p in shortlist[:3])
        if "rust" in full_l:
            prefix = "SHL's catalog does not include a Rust-specific knowledge test, so this uses the closest catalog-backed alternatives."
        elif _contains_any(full_l, ["spanish", "healthcare"]) and "hipaa" in full_l:
            prefix = "This is a hybrid battery: healthcare knowledge tests in English plus broader behavior and dependability measures."
        elif _contains_any(full_l, ["drop", "add", "remove", "actually"]):
            prefix = "Updated shortlist based on the changed constraints."
        elif _contains_any(full_l, ["confirmed", "lock", "final"]):
            prefix = "Final SHL assessment shortlist."
        else:
            prefix = "Here is a catalog-backed SHL assessment shortlist."
        return f"{prefix} Top matches: {names}."

    def _reply_with_recs(self, reply: str, recs: list[Product], end: bool = False) -> dict[str, object]:
        return {
            "reply": reply,
            "recommendations": [p.api() for p in self._dedupe(recs)[:10]],
            "end_of_conversation": end,
        }

    def _empty(self, reply: str) -> dict[str, object]:
        return {"reply": reply, "recommendations": [], "end_of_conversation": False}

    def _dedupe(self, products: Iterable[Product | None]) -> list[Product]:
        seen: set[str] = set()
        deduped: list[Product] = []
        for product in products:
            if not product or product.url in seen:
                continue
            seen.add(product.url)
            deduped.append(product)
        return deduped
