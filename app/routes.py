from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from app.agent_loop import AgentService
from app.agent_runs import AGENT_MODE_VERSION, AgentRunError
from app.approvals import ApprovalBroker
from app.model_gateway import ModelGateway, ModelGatewayError
from app.memory_tools import MEMORY_KINDS, MemoryToolError
from app.schemas import (
    AgentCloneRequest,
    AgentResumeRequest,
    AgentStartRequest,
    AgentSteerRequest,
    ApprovalDecision,
    ChatRequest,
    GraphNodeControlRequest,
    GraphNodeReplaceRequest,
    MemoryFeedbackRequest,
    ModelStatus,
    RegressionScenarioRequest,
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
        sandbox_ready, sandbox_error = await agent_service.terminal_tools.readiness()
        try:
            models = await gateway.list_models()
            selected_model = settings.model_name or (models[0] if models else None)
            return ModelStatus(
                connected=True,
                base_url=settings.model_base_url,
                models=models,
                selected_model=selected_model,
                agent_version=AGENT_MODE_VERSION,
                terminal_mode=settings.terminal_mode,
                sandbox_image=(
                    settings.sandbox_image
                    if settings.terminal_mode == "sandbox"
                    else None
                ),
                sandbox_ready=sandbox_ready,
                sandbox_error=sandbox_error,
            )
        except ModelGatewayError as exc:
            return ModelStatus(
                connected=False,
                base_url=settings.model_base_url,
                models=[],
                selected_model=settings.model_name,
                agent_version=AGENT_MODE_VERSION,
                terminal_mode=settings.terminal_mode,
                sandbox_image=(
                    settings.sandbox_image
                    if settings.terminal_mode == "sandbox"
                    else None
                ),
                sandbox_ready=sandbox_ready,
                sandbox_error=sandbox_error,
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

    @router.get("/api/agent/runs")
    async def agent_run_history() -> list[dict]:
        return agent_service.history()

    @router.get("/api/agent/runs/current")
    async def current_agent_run() -> Optional[dict]:
        return agent_service.current()

    @router.get("/api/agent/runs/compare")
    async def compare_agent_runs(left_id: str, right_id: str) -> dict:
        try:
            return agent_service.compare(left_id, right_id)
        except AgentRunError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/api/agent/runs/{run_id}")
    async def agent_run(run_id: str) -> dict:
        try:
            return agent_service.run(run_id)
        except AgentRunError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/api/agent/runs/{run_id}/clone")
    async def clone_agent_run(run_id: str, body: AgentCloneRequest) -> dict:
        try:
            return await agent_service.clone(run_id, body)
        except AgentRunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/api/agent/runs/{run_id}/debug")
    async def debug_agent_run(run_id: str) -> dict:
        try:
            return agent_service.debug(run_id)
        except AgentRunError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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

    @router.get("/api/agent/runs/{run_id}/events")
    async def agent_run_events(
        run_id: str,
        after: int = Query(default=0, ge=0),
    ) -> list[dict]:
        try:
            return agent_service.events(run_id, after)
        except AgentRunError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/api/agent/runs/{run_id}/trace")
    async def agent_run_trace(run_id: str) -> dict:
        try:
            return agent_service.trace(run_id)
        except AgentRunError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/api/agent/runs/{run_id}/evaluation")
    async def agent_run_evaluation(run_id: str) -> dict:
        try:
            return agent_service.evaluation(run_id)
        except AgentRunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/api/evals/scenarios")
    async def regression_scenarios() -> list[dict]:
        return agent_service.scenarios()

    @router.get("/api/evals/scenarios/{scenario_id}")
    async def regression_scenario(scenario_id: str) -> dict:
        try:
            return agent_service.scenario(scenario_id)
        except AgentRunError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/api/evals/scenarios/from-run/{run_id}")
    async def create_regression_scenario(
        run_id: str,
        body: RegressionScenarioRequest,
    ) -> dict:
        assertions = {
            "expected_status": body.expected_status,
            "min_quality_score": body.min_quality_score,
            "min_safety_score": body.min_safety_score,
            "max_cost_usd": body.max_cost_usd,
            "max_latency_ms": body.max_latency_ms,
        }
        try:
            return agent_service.create_scenario(
                run_id,
                name=body.name,
                assertions=assertions,
            )
        except AgentRunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/api/evals/scenarios/{scenario_id}/replay")
    async def replay_regression_scenario(scenario_id: str) -> StreamingResponse:
        try:
            execution = await agent_service.replay_scenario(scenario_id)
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

    @router.get("/api/agent/runs/{run_id}/tasks")
    async def agent_worker_tasks(run_id: str) -> list[dict]:
        try:
            return agent_service.tasks(run_id)
        except AgentRunError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/api/agent/runs/{run_id}/graph")
    async def agent_task_graph(run_id: str) -> Optional[dict]:
        try:
            return agent_service.graph(run_id)
        except AgentRunError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/api/agent/graph/templates")
    async def graph_templates() -> list[dict]:
        return agent_service.graph_templates()

    @router.get("/api/agent/runs/{run_id}/graph/nodes/{key}/artifacts/{name}")
    async def graph_artifact(
        run_id: str,
        key: str,
        name: str,
        task_id: Optional[str] = None,
    ) -> dict:
        try:
            return agent_service.graph_artifact(run_id, key, name, task_id)
        except AgentRunError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/api/agent/runs/{run_id}/graph/nodes/{key}/control")
    async def control_graph_node(
        run_id: str,
        key: str,
        body: GraphNodeControlRequest,
    ) -> dict:
        try:
            return await agent_service.control_graph_node(
                run_id,
                key,
                action=body.action,
                reason=body.reason,
                cascade=body.cascade,
            )
        except AgentRunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/api/agent/runs/{run_id}/graph/nodes/{key}/replace")
    async def replace_graph_node(
        run_id: str,
        key: str,
        body: GraphNodeReplaceRequest,
    ) -> dict:
        try:
            return await agent_service.replace_graph_node(
                run_id,
                key,
                replacement=body.replacement,
                reason=body.reason,
                cascade=body.cascade,
            )
        except AgentRunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/api/agent/runs/{run_id}/tasks/{task_id}/review")
    async def agent_worker_review(run_id: str, task_id: str) -> dict:
        try:
            return agent_service.review(run_id, task_id)
        except AgentRunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/api/memory/status")
    async def memory_status() -> dict:
        try:
            return await asyncio.to_thread(agent_service.memory_status)
        except MemoryToolError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/api/memory/search")
    async def search_memory(
        q: str = Query(min_length=1, max_length=500),
        kinds: Optional[list[str]] = Query(default=None),
        limit: int = Query(default=5, ge=1, le=10),
        include_stale: bool = False,
    ) -> dict:
        if kinds is not None and any(kind not in MEMORY_KINDS for kind in kinds):
            raise HTTPException(status_code=422, detail="Unsupported memory kind.")
        try:
            return await asyncio.to_thread(
                agent_service.search_memory,
                q,
                kinds=kinds,
                limit=limit,
                include_stale=include_stale,
            )
        except MemoryToolError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/api/memory/index/refresh")
    async def refresh_memory_index() -> dict:
        try:
            return await asyncio.to_thread(agent_service.refresh_memory_index)
        except MemoryToolError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/api/memory/entries/{memory_id}")
    async def read_memory_entry(memory_id: str, allow_stale: bool = False) -> dict:
        try:
            return await asyncio.to_thread(
                agent_service.read_memory,
                memory_id,
                allow_stale=allow_stale,
            )
        except MemoryToolError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/api/memory/lessons/{memory_id}/feedback")
    async def rate_memory_lesson(
        memory_id: str,
        body: MemoryFeedbackRequest,
    ) -> dict:
        try:
            return await asyncio.to_thread(
                agent_service.rate_memory_lesson,
                memory_id,
                rating=body.rating,
                reason=body.reason,
            )
        except MemoryToolError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/api/memory/lessons/consolidate")
    async def consolidate_memory_lessons() -> dict:
        try:
            return await asyncio.to_thread(agent_service.consolidate_memory_lessons)
        except MemoryToolError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/api/agent/runs/{run_id}/steer")
    async def steer_agent_run(run_id: str, body: AgentSteerRequest) -> dict:
        try:
            return await agent_service.steer(run_id, body)
        except AgentRunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

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
        selection = (
            {
                "accepted_paths": decision.accepted_paths,
                "accepted_hunks": decision.accepted_hunks,
            }
            if decision.accepted_paths is not None
            or decision.accepted_hunks is not None
            else None
        )
        if not approvals.decide(approval_id, decision.approved, selection):
            raise HTTPException(status_code=404, detail="Approval is no longer pending.")
        return {"accepted": True}

    return router
