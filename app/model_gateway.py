from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
import json
from typing import Optional

import httpx

from app.config import Settings
from app.schemas import ChatRequest


class ModelGatewayError(Exception):
    """Base error for communication with the model server."""


class ModelUnavailableError(ModelGatewayError):
    """The model server could not be reached."""


class ModelResponseError(ModelGatewayError):
    """The model server returned an unusable response."""


@dataclass
class ChatStream:
    client: httpx.AsyncClient
    response: httpx.Response
    model: str

    async def payloads(self) -> AsyncIterator[dict]:
        try:
            async for line in self.response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ModelResponseError("Model server returned invalid SSE JSON.") from exc
        finally:
            await self.response.aclose()
            await self.client.aclose()

    async def events(self) -> AsyncIterator[bytes]:
        """Relay OpenAI SSE events and always finish with a done marker."""
        async for payload in self.payloads():
            data = json.dumps(payload, separators=(",", ":"))
            yield f"data: {data}\n\n".encode()
        yield b"data: [DONE]\n\n"


class ModelGateway:
    def __init__(
        self,
        settings: Settings,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.settings = settings
        self._transport = transport
        self._stream_usage_supported: bool | None = None

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.settings.model_api_key:
            headers["Authorization"] = f"Bearer {self.settings.model_api_key}"
        return headers

    def _client(self, timeout: httpx.Timeout | float) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self._transport)

    async def list_models(self) -> list[str]:
        try:
            async with self._client(3.0) as client:
                return await self._list_models(client)
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(str(exc)) from exc

    async def open_chat(
        self,
        body: ChatRequest,
        tools: Optional[list[dict]] = None,
    ) -> ChatStream:
        return await self.open_completion(
            messages=[message.model_dump() for message in body.messages],
            requested_model=body.model,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            tools=tools,
        )

    async def open_completion(
        self,
        messages: list[dict],
        requested_model: Optional[str],
        temperature: float,
        max_tokens: int,
        tools: Optional[list[dict]] = None,
    ) -> ChatStream:
        client = self._client(httpx.Timeout(300.0, connect=10.0))
        try:
            model = await self._resolve_model(client, requested_model)
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
            if self._stream_usage_supported is not False:
                payload["stream_options"] = {"include_usage": True}
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            request = client.build_request(
                "POST",
                f"{self.settings.model_base_url}/v1/chat/completions",
                headers={**self.headers, "Content-Type": "application/json"},
                json=payload,
            )
            response = await client.send(request, stream=True)
        except asyncio.CancelledError:
            await client.aclose()
            raise
        except ModelGatewayError:
            await client.aclose()
            raise
        except (httpx.HTTPError, ValueError) as exc:
            await client.aclose()
            raise ModelUnavailableError(
                f"Could not reach the model server at "
                f"{self.settings.model_base_url}: {exc}"
            ) from exc

        if response.is_error and "stream_options" in payload:
            await response.aread()
            detail = self._error_detail(response)
            unsupported_usage = response.status_code in {400, 422} and any(
                marker in detail.lower()
                for marker in ("stream_options", "include_usage", "unknown field", "extra field")
            )
            if unsupported_usage:
                await response.aclose()
                self._stream_usage_supported = False
                payload.pop("stream_options", None)
                try:
                    retry = client.build_request(
                        "POST",
                        f"{self.settings.model_base_url}/v1/chat/completions",
                        headers={**self.headers, "Content-Type": "application/json"},
                        json=payload,
                    )
                    response = await client.send(retry, stream=True)
                except (httpx.HTTPError, ValueError) as exc:
                    await client.aclose()
                    raise ModelUnavailableError(
                        f"Could not reach the model server at {self.settings.model_base_url}: {exc}"
                    ) from exc
            else:
                await response.aclose()
                await client.aclose()
                raise ModelResponseError(detail)

        if response.is_error:
            await response.aread()
            detail = self._error_detail(response)
            await response.aclose()
            await client.aclose()
            raise ModelResponseError(detail)

        if "stream_options" in payload:
            self._stream_usage_supported = True

        return ChatStream(client=client, response=response, model=model)

    async def _resolve_model(
        self, client: httpx.AsyncClient, requested: Optional[str]
    ) -> str:
        if requested:
            return requested
        if self.settings.model_name:
            return self.settings.model_name

        models = await self._list_models(client)
        if not models:
            raise ModelResponseError(
                "The model server returned no models. Set MODEL_NAME explicitly."
            )
        return models[0]

    async def _list_models(self, client: httpx.AsyncClient) -> list[str]:
        response = await client.get(
            f"{self.settings.model_base_url}/v1/models",
            headers=self.headers,
        )
        response.raise_for_status()
        payload = response.json()
        return [item["id"] for item in payload.get("data", []) if item.get("id")]

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
            detail = payload.get("error", payload)
            if isinstance(detail, dict):
                return str(detail.get("message") or detail)
            return str(detail)
        except (ValueError, TypeError):
            return (
                response.text[:1000]
                or f"Model server returned HTTP {response.status_code}."
            )
