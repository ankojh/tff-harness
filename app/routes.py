from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from app.agent_loop import AgentService
from app.agent_runs import AgentRunError
from app.approvals import ApprovalBroker
from app.model_gateway import ModelGateway, ModelGatewayError
from app.schemas import (
    AgentResumeRequest,
    AgentStartRequest,
    ApprovalDecision,
    ChatRequest,
    ModelStatus,
)
from app.tool_loop import ToolLoop


STATIC_DIR = Path(__file__).parent / "static"


def create_router(
    gateway: ModelGateway,
    tool_loop: ToolLoop,
    agent_service: AgentService,
    approvals: ApprovalBroker,
) -> APIRouter:
    router = APIRouter()

    @router.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @router.get("/static/markdown.js", include_in_schema=False)
    async def markdown_script() -> FileResponse:
        return FileResponse(STATIC_DIR / "markdown.js", media_type="text/javascript")

    @router.get("/api/status", response_model=ModelStatus)
    async def status() -> ModelStatus:
        settings = gateway.settings
        try:
            models = await gateway.list_models()
            selected_model = settings.model_name or (models[0] if models else None)
            return ModelStatus(
                connected=True,
                base_url=settings.model_base_url,
                models=models,
                selected_model=selected_model,
                terminal_mode=settings.terminal_mode,
                sandbox_image=(
                    settings.sandbox_image
                    if settings.terminal_mode == "sandbox"
                    else None
                ),
            )
        except ModelGatewayError as exc:
            return ModelStatus(
                connected=False,
                base_url=settings.model_base_url,
                models=[],
                selected_model=settings.model_name,
                terminal_mode=settings.terminal_mode,
                sandbox_image=(
                    settings.sandbox_image
                    if settings.terminal_mode == "sandbox"
                    else None
                ),
                error=str(exc),
            )

    @router.post("/api/chat")
    async def chat(body: ChatRequest) -> StreamingResponse:
        try:
            run = await tool_loop.start(body)
        except ModelGatewayError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return StreamingResponse(
            run.events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/api/agent/runs/current")
    async def current_agent_run() -> Optional[dict]:
        return agent_service.current()

    @router.post("/api/agent/runs")
    async def create_agent_run(body: AgentStartRequest) -> StreamingResponse:
        try:
            execution = await agent_service.create(body)
        except AgentRunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return StreamingResponse(
            execution.events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @router.post("/api/agent/runs/{run_id}/resume")
    async def resume_agent_run(
        run_id: str,
        body: AgentResumeRequest,
    ) -> StreamingResponse:
        try:
            execution = await agent_service.resume(run_id, body)
        except AgentRunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return StreamingResponse(
            execution.events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @router.post("/api/agent/runs/{run_id}/stop")
    async def stop_agent_run(run_id: str) -> dict:
        try:
            return await agent_service.stop(run_id)
        except AgentRunError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/api/approvals/{approval_id}")
    async def decide_approval(
        approval_id: str, decision: ApprovalDecision
    ) -> dict[str, bool]:
        if not approvals.decide(approval_id, decision.approved):
            raise HTTPException(status_code=404, detail="Approval is no longer pending.")
        return {"accepted": True}

    return router
