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
from vastai_mcp.tools import (
    _SLIM_INSTANCE_FIELDS,
    _SLIM_OFFER_FIELDS,
    _SLIM_TEMPLATE_FIELDS,
)

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


# -- Shared live payloads --------------------

OFFER_LIMIT = 7
SPELLING_LIMIT = 10
TEMPLATE_LIMIT = 5
TEMPLATE_PAGE = 100
TEMPLATE_NAME = "pytorch"
GPU_NAME = "RTX 4090"


@pytest.fixture(scope="session")
def gpu_names(live):
    return live(tools.list_gpu_names)["gpu_names"]


@pytest.fixture(scope="session")
def offers(live):
    return live(tools.search_offers, limit=OFFER_LIMIT)["offers"]


@pytest.fixture(scope="session")
def offers_by_id(live):
    return live(tools.search_offers, limit=SPELLING_LIMIT, order="id")["offers"]


@pytest.fixture(scope="session")
def offers_spaced_name(live, gpu_names):
    return live(
        tools.search_offers, gpu_name=GPU_NAME, limit=SPELLING_LIMIT, order="id",
    )["offers"]


@pytest.fixture(scope="session")
def offers_underscored_name(live, gpu_names):
    return live(
        tools.search_offers, gpu_name=GPU_NAME.replace(" ", "_"),
        limit=SPELLING_LIMIT, order="id",
    )["offers"]


@pytest.fixture(scope="session")
def templates_page(live):
    return live(tools.search_templates, limit=TEMPLATE_PAGE)["templates"]


@pytest.fixture(scope="session")
def templates_limited(live):
    return live(tools.search_templates, limit=TEMPLATE_LIMIT)["templates"]


@pytest.fixture(scope="session")
def templates_named(live):
    return live(
        tools.search_templates, name=TEMPLATE_NAME, limit=TEMPLATE_PAGE,
    )["templates"]


# -- SearchOffers --------------------

class TestSearchOffers:
    def test_limit_is_honored(self, offers):
        """The March regression sent limit as {"eq": n}, which the API answers with 400."""
        assert len(offers) == OFFER_LIMIT

    def test_offers_are_slimmed(self, offers):
        for offer in offers:
            assert set(offer) <= _SLIM_OFFER_FIELDS, f"unslimmed offer {offer['id']}"
            assert offer["gpu_name"]
            assert offer["dph_total"] > 0

    def test_gpu_name_filter_narrows(self, offers_spaced_name, offers_by_id):
        assert offers_spaced_name
        assert {o["gpu_name"] for o in offers_spaced_name} == {GPU_NAME}
        assert {o["gpu_name"] for o in offers_by_id} != {GPU_NAME}

    def test_both_gpu_name_spellings_agree(self, offers_spaced_name, offers_underscored_name):
        """'RTX_4090' must resolve to the catalog name instead of matching nothing."""
        assert [o["id"] for o in offers_spaced_name] == [o["id"] for o in offers_underscored_name]

    def test_unknown_gpu_name_lists_the_catalog(self, gpu_names):
        with pytest.raises(ValueError, match="not a known GPU") as excinfo:
            tools.search_offers(gpu_name="RTX 9090")
        assert gpu_names[0] in str(excinfo.value)


# -- ListGpuNames --------------------

class TestListGpuNames:
    def test_catalog_is_a_non_empty_string_list(self, gpu_names):
        assert gpu_names
        assert all(isinstance(name, str) and name for name in gpu_names)

    def test_catalog_holds_the_name_offers_are_searched_by(self, gpu_names):
        assert GPU_NAME in gpu_names


# -- SearchTemplates --------------------

class TestSearchTemplates:
    def test_limit_cuts(self, templates_limited):
        assert len(templates_limited) == TEMPLATE_LIMIT

    def test_name_filter_narrows(self, templates_named, templates_page):
        assert templates_named
        assert len(templates_named) < len(templates_page)
        for template in templates_named:
            assert TEMPLATE_NAME in template["name"].lower()

    def test_no_fields_beyond_the_slim_set(self, templates_page):
        for template in templates_page:
            assert set(template) <= _SLIM_TEMPLATE_FIELDS, (
                f"unslimmed template {template['id']}"
            )


# -- ShowInvoicesV1 --------------------

WIDE_START = "2020-01-01"
WIDE_END = "2027-01-01"
WIDE_START_TS = 1577836800
WIDE_END_TS = 1798761600


class TestShowInvoicesV1:
    def test_default_range_succeeds(self, live):
        """An unbounded range answers 'Invalid date range', so the default must be bounded."""
        assert live(tools.show_invoices_v1)["success"] is True

    def test_explicit_window_bounds_the_results(self, live):
        result = live(
            tools.show_invoices_v1, start_date=WIDE_START, end_date=WIDE_END, limit=50,
        )
        assert result["success"] is True
        assert result["results"], "account has no invoice between 2020 and 2027"
        assert result["count"] == len(result["results"])
        for invoice in result["results"]:
            assert WIDE_START_TS <= invoice["start"] <= WIDE_END_TS

    def test_window_before_the_account_existed_is_empty(self, live):
        result = live(
            tools.show_invoices_v1, start_date="2000-01-01", end_date="2001-01-01",
        )
        assert result["success"] is True
        assert result["count"] == 0
        assert result["results"] == []


# -- Remaining read operations --------------------

class TestListInstances:
    def test_shape_and_slimming(self, live):
        result = live(tools.list_instances)
        assert isinstance(result["instances_found"], int)
        for instance in result["instances"]:
            assert set(instance) <= _SLIM_INSTANCE_FIELDS


class TestSearchVolumes:
    def test_offers_have_ids_and_space(self, live):
        offers = live(tools.search_volumes)["offers"]
        assert offers
        for offer in offers:
            assert isinstance(offer["id"], int)
            assert offer["disk_space"] > 0


class TestSearchBenchmarks:
    def test_rows_carry_machine_and_value(self, live):
        rows = live(tools.search_benchmarks)
        assert rows
        assert {"id", "machine_id", "value"} <= set(rows[0])


class TestListEndpoints:
    def test_shape(self, live):
        result = live(tools.list_endpoints)
        assert result["success"] is True
        assert isinstance(result["results"], list)


class TestListWorkergroups:
    def test_shape(self, live):
        result = live(tools.list_workergroups)
        assert result["success"] is True
        assert isinstance(result["results"], list)


class TestShowUser:
    def test_identity_and_balance(self, live):
        user = live(tools.show_user)
        assert isinstance(user["id"], int)
        assert isinstance(user["credit"], float)
        assert "email" in user
