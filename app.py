from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from recommender import SHLRecommender


app = FastAPI(title="Conversational SHL Assessment Recommender")
recommender = SHLRecommender()


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


@app.get("/")
def root() -> dict[str, object]:
    return {
        "service": "Conversational SHL Assessment Recommender",
        "status": "ok",
        "endpoints": {
            "health": "GET /health",
            "chat": "POST /chat",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    messages = [message.model_dump() for message in request.messages if message.role in {"user", "assistant"}]
    return recommender.chat(messages)
