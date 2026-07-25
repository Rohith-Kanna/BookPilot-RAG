import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph import build_graph

app = FastAPI(title="BookPilot API")

# Allows a frontend (running on a different port) to call this API later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev, tighten later if you deploy
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the LangGraph agent once when the server starts (not per-request — reuse it)
agent_graph = build_graph()


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    used_retrieval: bool


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = agent_graph.invoke({"question": request.question})
    return ChatResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        used_retrieval=result.get("needs_retrieval", False),
    )


@app.get("/")
def health_check():
    return {"status": "BookPilot API is running"}