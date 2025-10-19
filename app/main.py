from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.db import db
from app.schemas import (
    DetectEmotionRequest,
    DetectEmotionResponse,
    EmotionLogItem,
    HistoryResponse,
    LogEmotionRequest,
    LogEmotionResponse,
)
from app.utils.sentiment import analyze_sentiment


app = FastAPI(title="AI Guardian 2.0 - Stage 1 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/detect_emotion", response_model=DetectEmotionResponse)
def detect_emotion(payload: DetectEmotionRequest) -> DetectEmotionResponse:
    if not payload.user_message or not payload.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message must not be empty")
    result = analyze_sentiment(payload.user_message)
    return DetectEmotionResponse(emotion=result["emotion"], emergency_level=result["emergency_level"])


@app.post("/log_emotion", response_model=LogEmotionResponse)
def log_emotion(payload: LogEmotionRequest) -> LogEmotionResponse:
    if not payload.user_message or not payload.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message must not be empty")
    timestamp = db.log_emotion(
        user_message=payload.user_message,
        emotion=payload.emotion,
        emergency_level=payload.emergency_level,
        user_id=payload.user_id,
    )
    return LogEmotionResponse(status="logged", timestamp=timestamp)


@app.get("/history", response_model=HistoryResponse)
def history(user_id: str | None = None, last_n: int | None = Query(None, ge=1, le=1000)) -> HistoryResponse:
    items = db.get_history(user_id=user_id, last_n=last_n)
    return HistoryResponse(items=[EmotionLogItem(**it) for it in items])


# To run locally: uvicorn app.main:app --reload

