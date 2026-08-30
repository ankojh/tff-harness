from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=200)
    model: Optional[str] = Field(default=None, max_length=200)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1, le=32_768)


class ModelStatus(BaseModel):
    connected: bool
    base_url: str
    models: list[str]
    selected_model: Optional[str]
    terminal_mode: str
    sandbox_image: Optional[str] = None
    error: Optional[str] = None


class ApprovalDecision(BaseModel):
    approved: bool
