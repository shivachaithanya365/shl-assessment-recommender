from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recommender import SHLRecommender


def names(response: dict[str, object]) -> list[str]:
    return [item["name"] for item in response["recommendations"]]  # type: ignore[index]


def assert_schema(response: dict[str, object]) -> None:
    assert set(response) == {"reply", "recommendations", "end_of_conversation"}
    assert isinstance(response["reply"], str)
    assert isinstance(response["recommendations"], list)
    assert isinstance(response["end_of_conversation"], bool)
    for item in response["recommendations"]:  # type: ignore[union-attr]
        assert set(item) == {"name", "url", "test_type"}
        assert item["url"].startswith("https://www.shl.com/products/product-catalog/view/")


def main() -> None:
    rec = SHLRecommender()

    cases = [
        [{"role": "user", "content": "I need an assessment"}],
        [
            {
                "role": "user",
                "content": "Graduate management trainee scheme. Need cognitive, personality, and situational judgement.",
            }
        ],
        [
            {
                "role": "user",
                "content": "We are screening entry-level contact center agents. Inbound calls and customer service. English.",
            }
        ],
        [{"role": "user", "content": "Are we legally required under HIPAA to test all staff?"}],
        [
            {
                "role": "user",
                "content": "Senior Full-Stack Engineer, backend-leaning. Core Java, Spring, SQL. Senior IC.",
            },
            {
                "role": "assistant",
                "content": "Here is a shortlist: Core Java (Advanced Level) (New), Spring (New), RESTful Web Services (New), SQL (New).",
            },
            {"role": "user", "content": "Add AWS and Docker. Drop REST."},
        ],
        [{"role": "user", "content": "Drop the OPQ. Final list: Verify G+ and Graduate Scenarios."}],
    ]

    responses = [rec.chat(case) for case in cases]
    for response in responses:
        assert_schema(response)

    assert responses[0]["recommendations"] == []
    assert "accent" in responses[2]["reply"].lower()
    assert "legal" in responses[3]["reply"].lower()

    graduate = names(responses[1])
    assert "SHL Verify Interactive G+" in graduate
    assert "Occupational Personality Questionnaire OPQ32r" in graduate
    assert "Graduate Scenarios" in graduate

    java = names(responses[4])
    assert "Amazon Web Services (AWS) Development (New)" in java
    assert "Docker (New)" in java
    assert "RESTful Web Services (New)" not in java

    final_list = names(responses[5])
    assert final_list == ["SHL Verify Interactive G+", "Graduate Scenarios"]

    print("smoke tests passed")


if __name__ == "__main__":
    main()
