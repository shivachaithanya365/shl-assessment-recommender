# Conversational SHL Assessment Recommender

FastAPI service for the SHL AI Intern take-home assignment.

## Live Deployment

Public API:

https://shl-assessment-recommender-cjl9.onrender.com

Health check:

https://shl-assessment-recommender-cjl9.onrender.com/health

API docs:

https://shl-assessment-recommender-cjl9.onrender.com/docs

## Endpoints

- `GET /health` returns `{"status":"ok"}`
- `POST /chat` accepts stateless conversation history:

```json
{
  "messages": [
    {"role": "user", "content": "Hiring a senior Java developer with Spring and SQL"}
  ]
}
```

Response shape:

```json
{
  "reply": "...",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then test:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Graduate management trainee scheme. Need cognitive, personality, and situational judgement."}]}'
```

## Smoke Tests

```bash
python scripts/smoke_test.py
```

## Deployment

This repository is ready for Render/Railway/Fly-style deployment:

- Install command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

`Procfile` and `runtime.txt` are included for platforms that detect them.
