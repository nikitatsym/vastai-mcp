"""Live read-only tests against the vast.ai API.

Nothing here creates, changes or destroys anything: search/list/show only.
No API key means every test in this file fails - a skip would hide the API
drift these tests exist to catch.
"""

import os
import time
from typing import Any

import httpx
import pytest

from vastai_mcp import tools
from vastai_mcp.client import APIError, VastClient

pytestmark = pytest.mark.integration

# -- Captured rate limit --------------------

# Real 429 from POST /api/v0/bundles/, 2026-07-25. Volatile headers (date,
# set-cookie) dropped, the rest is verbatim. The body carries retry_after even
# though the vendor documents only the header, so the backoff reads the body.
RATE_LIMIT_HEADERS = {
    "content-type": "application/json",
    "connection": "keep-alive",
    "server": "gunicorn",
    "x-ratelimit-limit": "5.0",
    "x-ratelimit-remaining": "0",
    "x-ratelimit-reset": "1784961439",
    "retry-after": "10",
}

RATE_LIMIT_BODY = (
    b'{"error":"HTTPTooManyRequests","msg":"API requests too frequent",'
    b'"retry_after":10,"limit":5.0,"remaining":0}'
)

TOO_MANY_REQUESTS = 429

# Observed budget: about 5 requests per 10 seconds, shared across endpoints.
_MIN_INTERVAL_S = 4.0
_BACKOFF_START_S = 2.0
_MAX_ATTEMPTS = 5

BUNDLES_URL = "https://console.vast.ai/api/v0/bundles/"


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    body = response.json()
    if "retry_after" in body:
        return float(body["retry_after"])
    return _BACKOFF_START_S * 2**attempt


def _send_with_retry(send: Any, request: httpx.Request) -> httpx.Response:
    """Retries a 429 while it is still a response, below the layer that raises APIError.

    The last attempt is returned as it comes, so a rate limit the server cannot
    ride out still reaches the caller as APIError.
    """
    for attempt in range(_MAX_ATTEMPTS - 1):
        response = send(request)
        if response.status_code != TOO_MANY_REQUESTS:
            return response
        response.read()
        time.sleep(_retry_delay(response, attempt))
    return send(request)


class _RetryingTransport(httpx.HTTPTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return _send_with_retry(super().handle_request, request)


class _LiveApi:
    """The single gate for live calls: they run one at a time, spaced under the limit."""

    def __init__(self) -> None:
        self._last_call_at = 0.0

    def __call__(self, fn: Any, **kwargs: Any) -> Any:
        wait = _MIN_INTERVAL_S - (time.monotonic() - self._last_call_at)
        if wait > 0:
            time.sleep(wait)
        result = fn(**kwargs)
        self._last_call_at = time.monotonic()
        return result


@pytest.fixture(scope="session", autouse=True)
def require_api_key() -> None:
    key = os.environ["VASTAI_API_KEY"]
    assert key, "VASTAI_API_KEY is empty"


@pytest.fixture(scope="session", autouse=True)
def live_client(require_api_key):
    client = VastClient()
    client._http._transport = _RetryingTransport()
    client._run_http._transport = _RetryingTransport()
    tools._client = client
    yield
    tools._client = None


@pytest.fixture(scope="session")
def live() -> _LiveApi:
    return _LiveApi()


# -- Rate-limit harness --------------------

def _captured_429() -> httpx.Response:
    return httpx.Response(
        TOO_MANY_REQUESTS,
        headers=RATE_LIMIT_HEADERS,
        content=RATE_LIMIT_BODY,
        request=httpx.Request("POST", BUNDLES_URL),
    )


class _Sender:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = 0

    def __call__(self, request):
        self.sent += 1
        return self.responses.pop(0)


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)
    return slept


class TestRateLimitHandling:
    def test_captured_body_drives_the_delay(self):
        assert _retry_delay(_captured_429(), 0) == 10.0
        assert _retry_delay(_captured_429(), 3) == 10.0

    def test_captured_headers_agree_with_the_body(self):
        body = _captured_429().json()
        assert RATE_LIMIT_HEADERS["retry-after"] == str(body["retry_after"])
        assert RATE_LIMIT_HEADERS["x-ratelimit-limit"] == str(body["limit"])
        assert RATE_LIMIT_HEADERS["x-ratelimit-remaining"] == str(body["remaining"])

    def test_delay_is_exponential_without_retry_after(self):
        response = httpx.Response(
            TOO_MANY_REQUESTS, json={"error": "HTTPTooManyRequests"},
            request=httpx.Request("POST", BUNDLES_URL),
        )
        assert [_retry_delay(response, i) for i in range(4)] == [2.0, 4.0, 8.0, 16.0]

    def test_retries_a_429_then_returns(self, no_sleep):
        ok = httpx.Response(200, json={"offers": []}, request=httpx.Request("POST", BUNDLES_URL))
        sender = _Sender([_captured_429(), _captured_429(), ok])
        response = _send_with_retry(sender, httpx.Request("POST", BUNDLES_URL))
        assert response.status_code == 200
        assert sender.sent == 3
        assert no_sleep == [10.0, 10.0]

    def test_gives_up_after_max_attempts(self, no_sleep):
        sender = _Sender([_captured_429() for _ in range(_MAX_ATTEMPTS)])
        response = _send_with_retry(sender, httpx.Request("POST", BUNDLES_URL))
        assert sender.sent == _MAX_ATTEMPTS
        assert len(no_sleep) == _MAX_ATTEMPTS - 1
        with pytest.raises(APIError, match="429"):
            VastClient()._handle(response)

    def test_other_errors_are_not_retried(self, no_sleep):
        bad = httpx.Response(
            400, json={"msg": "limit: Input should be a valid integer"},
            request=httpx.Request("POST", BUNDLES_URL),
        )
        sender = _Sender([bad])
        response = _send_with_retry(sender, httpx.Request("POST", BUNDLES_URL))
        assert response.status_code == 400
        assert sender.sent == 1
        assert no_sleep == []
