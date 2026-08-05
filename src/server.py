"""
新能源智能体 - FastAPI 后端
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.agent import NewEnergyAgent

logger = logging.getLogger(__name__)

app = FastAPI(title="New Energy Agent", docs_url=None, redoc_url=None)
agent = NewEnergyAgent()
UI_DIR = Path(__file__).parent.parent / "ui"


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    text: str
    status: str = ""


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    try:
        result = agent.process(req.message.strip())
        return ChatResponse(text=result["text"], status=result.get("status", ""))
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reset")
async def reset():
    agent.reset()
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = UI_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not found</h1>")


if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 50)
    logger.info("New Energy Agent starting (DeepSeek-native)...")
    logger.info("=" * 50)
    uvicorn.run("src.server:app", host="0.0.0.0", port=7860, reload=False, log_level="info")
