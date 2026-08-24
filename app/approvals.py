from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass


@dataclass
class PendingApproval:
    name: str
    path: str
    future: asyncio.Future[bool]


class ApprovalBroker:
    def __init__(self, timeout_seconds: float = 300.0) -> None:
        self._pending: dict[str, PendingApproval] = {}
        self._timeout_seconds = timeout_seconds

    def register(self, name: str, path: str) -> str:
        approval_id = secrets.token_urlsafe(24)
        future = asyncio.get_running_loop().create_future()
        self._pending[approval_id] = PendingApproval(name, path, future)
        return approval_id

    async def wait(self, approval_id: str) -> bool:
        approval = self._pending.get(approval_id)
        if approval is None:
            return False
        try:
            return await asyncio.wait_for(
                asyncio.shield(approval.future),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            return False
        finally:
            self._pending.pop(approval_id, None)
            if not approval.future.done():
                approval.future.cancel()

    def decide(self, approval_id: str, approved: bool) -> bool:
        approval = self._pending.get(approval_id)
        if approval is None or approval.future.done():
            return False
        approval.future.set_result(approved)
        return True

    def cancel(self, approval_id: str) -> None:
        approval = self._pending.pop(approval_id, None)
        if approval and not approval.future.done():
            approval.future.cancel()
