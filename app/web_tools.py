from __future__ import annotations

from html.parser import HTMLParser
import ipaddress
import json
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urljoin, urlsplit

import httpx


SEARCH_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = (
    "Mozilla/5.0 (compatible; LocalModelHarness/0.1; "
    "+https://localhost.invalid)"
)
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_PAGE_TEXT_CHARS = 20_000
MAX_PAGE_LINKS = 50
DEFAULT_SEARCH_RESULTS = 5
MAX_SEARCH_RESULTS = 10


class WebToolError(Exception):
    """A web tool request failed validation or execution."""


class PageParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[dict[str, str]] = []
        self._ignored_depth = 0
        self._in_title = False
        self._active_link: dict[str, Any] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = True
        if tag == "a" and len(self.links) < MAX_PAGE_LINKS:
            attributes = dict(attrs)
            href = attributes.get("href")
            if href:
                self._active_link = {
                    "url": urljoin(self.base_url, href),
                    "text_parts": [],
                }

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = False
        if tag == "a" and self._active_link is not None:
            text = _collapse_whitespace(" ".join(self._active_link["text_parts"]))
            self.links.append({"url": self._active_link["url"], "text": text})
            self._active_link = None

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        self.text_parts.append(text)
        if self._active_link is not None:
            self._active_link["text_parts"].append(text)

    @property
    def title(self) -> str:
        return _collapse_whitespace(" ".join(self.title_parts))

    @property
    def text(self) -> str:
        return _collapse_whitespace(" ".join(self.text_parts))


class SearchParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._title_depth = 0
        self._snippet_depth = 0
        self._active_result: dict[str, str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            href = attributes.get("href")
            if href:
                result = {"title": "", "url": _search_result_url(href), "snippet": ""}
                self.results.append(result)
                self._active_result = result
                self._title_depth = 1
                return
        if self._title_depth:
            self._title_depth += 1
        if "result__snippet" in classes and self.results:
            self._active_result = self.results[-1]
            self._snippet_depth = 1
            return
        if self._snippet_depth:
            self._snippet_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._title_depth:
            self._title_depth -= 1
            if not self._title_depth:
                self._active_result = None
        if self._snippet_depth:
            self._snippet_depth -= 1
            if not self._snippet_depth:
                self._active_result = None

    def handle_data(self, data: str) -> None:
        if self._active_result is None:
            return
        text = data.strip()
        if not text:
            return
        key = "title" if self._title_depth else "snippet"
        existing = self._active_result[key]
        self._active_result[key] = _collapse_whitespace(f"{existing} {text}")


class WebTools:
    def __init__(self, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self._transport = transport

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [
            self._definition(
                "search_web",
                "Search the public web and return result titles, URLs, and snippets.",
                {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return, from 1 to 10.",
                        "minimum": 1,
                        "maximum": MAX_SEARCH_RESULTS,
                        "default": DEFAULT_SEARCH_RESULTS,
                    },
                },
                ["query"],
            ),
            self._definition(
                "fetch_url",
                "Fetch an HTTP or HTTPS page and extract readable text and links.",
                {
                    "url": {
                        "type": "string",
                        "description": "Full public HTTP or HTTPS URL.",
                    }
                },
                ["url"],
            ),
        ]

    @property
    def names(self) -> set[str]:
        return {item["function"]["name"] for item in self.definitions}

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            if name == "search_web":
                result = await self.search_web(
                    self._string(arguments, "query"),
                    self._integer(
                        arguments,
                        "max_results",
                        DEFAULT_SEARCH_RESULTS,
                        minimum=1,
                        maximum=MAX_SEARCH_RESULTS,
                    ),
                )
            elif name == "fetch_url":
                result = await self.fetch_url(self._string(arguments, "url"))
            else:
                raise WebToolError(f"Unknown web tool: {name}")
            return json.dumps({"ok": True, **result}, ensure_ascii=False)
        except WebToolError as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        except httpx.HTTPError as exc:
            return json.dumps(
                {"ok": False, "error": f"Web request failed: {exc}"},
                ensure_ascii=False,
            )

    async def search_web(
        self, query: str, max_results: int = DEFAULT_SEARCH_RESULTS
    ) -> dict[str, Any]:
        body, _, _ = await self._request(
            SEARCH_URL,
            params={"q": query},
            accepted_types={"text/html", "application/xhtml+xml"},
        )
        parser = SearchParser()
        parser.feed(body.decode("utf-8", errors="replace"))
        results = [result for result in parser.results if result["title"] and result["url"]]
        return {"query": query, "results": results[:max_results]}

    async def fetch_url(self, url: str) -> dict[str, Any]:
        self._validate_public_url(url)
        body, content_type, final_url = await self._request(
            url,
            accepted_types={
                "text/html",
                "application/xhtml+xml",
                "text/plain",
                "application/json",
            },
        )
        decoded = body.decode("utf-8", errors="replace")
        if content_type in {"text/html", "application/xhtml+xml"}:
            parser = PageParser(final_url)
            parser.feed(decoded)
            text = parser.text[:MAX_PAGE_TEXT_CHARS]
            return {
                "url": final_url,
                "title": parser.title,
                "text": text,
                "links": parser.links,
                "truncated": len(parser.text) > len(text),
            }
        text = decoded[:MAX_PAGE_TEXT_CHARS]
        return {
            "url": final_url,
            "content_type": content_type,
            "text": text,
            "links": [],
            "truncated": len(decoded) > len(text),
        }

    async def _request(
        self,
        url: str,
        params: dict[str, str] | None = None,
        accepted_types: set[str] | None = None,
    ) -> tuple[bytes, str, str]:
        timeout = httpx.Timeout(15.0, connect=5.0)
        async with httpx.AsyncClient(
            timeout=timeout,
            transport=self._transport,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain,application/json"},
        ) as client:
            async with client.stream("GET", url, params=params) as response:
                response.raise_for_status()
                final_url = str(response.url)
                self._validate_public_url(final_url)
                content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if accepted_types and content_type not in accepted_types:
                    raise WebToolError(
                        f"Unsupported content type: {content_type or 'unknown'}"
                    )
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise WebToolError(
                            f"Response exceeded the {MAX_RESPONSE_BYTES}-byte limit."
                        )
                    chunks.append(chunk)
        return b"".join(chunks), content_type, final_url

    @staticmethod
    def display_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search_web":
            query = arguments.get("query")
            return {"query": query if isinstance(query, str) else ""}
        url = arguments.get("url")
        return {"url": url if isinstance(url, str) else ""}

    @staticmethod
    def _validate_public_url(url: str) -> None:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise WebToolError("url must be a full HTTP or HTTPS URL.")
        hostname = parsed.hostname.lower().rstrip(".")
        if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
            raise WebToolError("Local URLs are not available to web tools.")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            return
        if not address.is_global:
            raise WebToolError("Private and local network URLs are not available to web tools.")

    @staticmethod
    def _definition(
        name: str,
        description: str,
        properties: dict[str, Any],
        required: list[str],
    ) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _string(arguments: dict[str, Any], key: str) -> str:
        value = arguments.get(key)
        if not isinstance(value, str) or not value.strip():
            raise WebToolError(f"{key} must be a non-empty string.")
        return value.strip()

    @staticmethod
    def _integer(
        arguments: dict[str, Any],
        key: str,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        value = arguments.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise WebToolError(f"{key} must be an integer.")
        if not minimum <= value <= maximum:
            raise WebToolError(f"{key} must be between {minimum} and {maximum}.")
        return value


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def _search_result_url(href: str) -> str:
    absolute = urljoin(SEARCH_URL, href)
    parsed = urlsplit(absolute)
    redirected = parse_qs(parsed.query).get("uddg")
    return unquote(redirected[0]) if redirected else absolute
