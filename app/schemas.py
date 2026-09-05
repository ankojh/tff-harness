from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=200)
    model: Optional[str] = Field(default=None, max_length=200)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1, le=32_768)


class AgentStartRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=20_000)
    model: Optional[str] = Field(default=None, max_length=200)
    temperature: float = Field(default=0.4, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1, le=32_768)
    grants: list[Literal["workspace_mutations", "terminal"]] = Field(
        default_factory=list,
        max_length=2,
    )


class AgentResumeRequest(BaseModel):
    instruction: Optional[str] = Field(default=None, max_length=20_000)


class AgentSteerRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=20_000)


class AgentCloneRequest(BaseModel):
    goal: Optional[str] = Field(default=None, min_length=1, max_length=20_000)


class RegressionScenarioRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    expected_status: Optional[
        Literal["completed", "blocked", "failed", "budget_exhausted", "stopped"]
    ] = None
    min_quality_score: Optional[int] = Field(default=None, ge=0, le=100)
    min_safety_score: Optional[int] = Field(default=None, ge=0, le=100)
    max_cost_usd: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    max_latency_ms: Optional[int] = Field(default=None, ge=0, le=604_800_000)


class MemoryFeedbackRequest(BaseModel):
    rating: Literal["helpful", "unhelpful"]
    reason: Optional[str] = Field(default=None, max_length=500)


class ModelStatus(BaseModel):
    connected: bool
    base_url: str
    models: list[str]
    selected_model: Optional[str]
    agent_version: str
    terminal_mode: str
    sandbox_image: Optional[str] = None
    sandbox_ready: Optional[bool] = None
    sandbox_error: Optional[str] = None
    error: Optional[str] = None


class ApprovalDecision(BaseModel):
    approved: bool
    accepted_paths: Optional[list[str]] = Field(default=None, max_length=500)
    accepted_hunks: Optional[dict[str, list[str]]] = None


class GraphNodeControlRequest(BaseModel):
    action: Literal["pause", "resume", "rerun"]
    reason: str = Field(min_length=1, max_length=1000)
    cascade: bool = True


class GraphNodeReplaceRequest(BaseModel):
    replacement: dict[str, Any]
    reason: str = Field(min_length=1, max_length=1000)
    cascade: bool = True
