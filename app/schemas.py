from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DetectEmotionRequest(BaseModel):
    user_message: str = Field(..., min_length=1)
    user_id: Optional[str] = Field(None, description="Optional user id for multi-user support")


class DetectEmotionResponse(BaseModel):
    emotion: str
    emergency_level: str


class LogEmotionRequest(BaseModel):
    user_message: str
    emotion: str
    emergency_level: str
    user_id: Optional[str] = None


class LogEmotionResponse(BaseModel):
    status: str
    timestamp: str


class EmotionLogItem(BaseModel):
    timestamp: str
    user_message: str
    emotion: str
    emergency_level: str
    user_id: Optional[str] = None


class HistoryResponse(BaseModel):
    items: List[EmotionLogItem]


