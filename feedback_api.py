
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

app = FastAPI()

# In-memory feedback storage
feedback_store = []

class Feedback(BaseModel):
    input: str
    tool: str
    result: str
    feedback: str  # 'like' or 'dislike'

@app.post("/api/feedback")
async def store_feedback(feedback: Feedback):
    feedback_store.append(feedback.dict())
    return {"status": "ok", "stored": feedback.dict()}

@app.get("/api/feedback")
async def get_feedback():
    return feedback_store

# Optional: Helper to bias routing based on past feedback
@app.get("/api/tool-bias")
async def get_tool_bias():
    summary = {}
    for item in feedback_store:
        key = item["tool"]
        score = 1 if item["feedback"] == "like" else -1
        summary[key] = summary.get(key, 0) + score
    return summary
