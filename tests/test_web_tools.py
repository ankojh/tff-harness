import json

import httpx
import pytest

from app.web_tools import WebTools


@pytest.mark.asyncio
async def test_search_web_returns_parsed_results():
    html = """
    <html><body>
      <div class="result">
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs">
          Example <strong>Docs</strong>
        </a>
        <a class="result__snippet">A useful example page.</a>
      </div>
    </body></html>
    """

    def handler(request):
        assert request.url.params["q"] == "example docs"
        return httpx.Response(
            200,
            text=html,
            headers={"content-type": "text/html; charset=utf-8"},
        )

    tools = WebTools(httpx.MockTransport(handler))
    result = json.loads(
        await tools.execute("search_web", {"query": "example docs", "max_results": 1})
    )

    assert result["ok"] is True
    assert result["results"] == [
        {
            "title": "Example Docs",
            "url": "https://example.com/docs",
            "snippet": "A useful example page.",
        }
    ]


@pytest.mark.asyncio
async def test_fetch_url_extracts_text_and_links():
    html = """
    <html>
      <head><title>Example Page</title><style>hidden</style></head>
      <body><main>Hello <b>world</b>. <a href="/more">Read more</a></main></body>
    </html>
    """

    def handler(request):
        return httpx.Response(
            200,
            text=html,
            headers={"content-type": "text/html"},
        )

    tools = WebTools(httpx.MockTransport(handler))
    result = json.loads(await tools.execute("fetch_url", {"url": "https://example.com"}))

    assert result["ok"] is True
    assert result["title"] == "Example Page"
    assert "Hello world" in result["text"]
    assert result["links"] == [
        {"url": "https://example.com/more", "text": "Read more"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:8080", "http://localhost/private", "file:///etc/passwd"],
)
async def test_fetch_url_rejects_non_public_urls(url):
    tools = WebTools()

    result = json.loads(await tools.execute("fetch_url", {"url": url}))

    assert result["ok"] is False
