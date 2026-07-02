from recommender import SHLRecommender


def names(response):
    return [item["name"] for item in response["recommendations"]]


def test_health_schema_shape():
    rec = SHLRecommender()
    response = rec.chat([{"role": "user", "content": "I need an assessment"}])
    assert set(response) == {"reply", "recommendations", "end_of_conversation"}
    assert response["recommendations"] == []


def test_java_refinement_adds_aws_docker_and_drops_rest():
    rec = SHLRecommender()
    messages = [
        {
            "role": "user",
            "content": "Senior Full-Stack Engineer, backend-leaning. Core Java, Spring, SQL. Senior IC.",
        },
        {"role": "assistant", "content": "Here is a shortlist: Core Java (Advanced Level) (New), Spring (New), RESTful Web Services (New), SQL (New)."},
        {"role": "user", "content": "Add AWS and Docker. Drop REST."},
    ]
    response = rec.chat(messages)
    got = names(response)
    assert "Amazon Web Services (AWS) Development (New)" in got
    assert "Docker (New)" in got
    assert "RESTful Web Services (New)" not in got


def test_contact_center_clarifies_accent_after_english():
    rec = SHLRecommender()
    response = rec.chat(
        [
            {
                "role": "user",
                "content": "We are screening entry-level contact center agents. Inbound calls and customer service. English.",
            }
        ]
    )
    assert response["recommendations"] == []
    assert "accent" in response["reply"].lower()


def test_legal_question_refuses():
    rec = SHLRecommender()
    response = rec.chat([{"role": "user", "content": "Are we legally required under HIPAA to test all staff?"}])
    assert response["recommendations"] == []
    assert "legal" in response["reply"].lower()


def test_graduate_management_battery():
    rec = SHLRecommender()
    response = rec.chat(
        [
            {
                "role": "user",
                "content": "Graduate management trainee scheme. Need cognitive, personality, and situational judgement.",
            }
        ]
    )
    got = names(response)
    assert "SHL Verify Interactive G+" in got
    assert "Occupational Personality Questionnaire OPQ32r" in got
    assert "Graduate Scenarios" in got
