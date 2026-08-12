"""Live tests against the vast.ai API.

Everything up to the paid rental harness is read-only: search/list/show only. The last
class rents a real GPU, runs a command on it and destroys it; every teardown layer aims at
that one instance. No API key means every test in this file fails - a skip would hide the
API drift these tests exist to catch.
"""

import atexit
import contextlib
import json
import os
import signal
import time
import uuid
import warnings
from typing import Any

import httpx
import pytest

import dev
from scripts.sweep import LABEL_PREFIX
from vastai_mcp import tools
from vastai_mcp.client import APIError, VastClient
from vastai_mcp.tools import (
    _BENCHMARK_COLUMNS,
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


# Machine 11's benchmark rows are dated 2018 and have not moved since, which is what makes
# the machine usable as a fixture.
BENCHMARK_MACHINE_ID = 11
BENCHMARK_PAGE = 100
BENCHMARK_LIMIT = 2


@pytest.fixture(scope="session")
def benchmarks_of_machine(live):
    return live(
        tools.search_benchmarks, machine_id=BENCHMARK_MACHINE_ID, limit=BENCHMARK_PAGE,
    )


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


# -- SearchInvoices --------------------

INVOICE_COLUMNS = {"id", "when", "amount_cents", "is_credit", "service"}
INVOICE_WINDOW_START = "2025-07-01"
INVOICE_PAGE = 100


def _invoices_query(**query: Any) -> Any:
    """Bypasses SearchInvoices to probe the endpoint contract its typed params rest on."""
    return tools._get_client().get("/api/v0/invoices/", params=query)


@pytest.fixture(scope="session")
def invoices_all(live):
    return live(tools.search_invoices, limit=INVOICE_PAGE)


class TestSearchInvoices:
    def test_rows_carry_the_billing_columns(self, invoices_all):
        assert invoices_all, "account has no invoice at all"
        assert INVOICE_COLUMNS <= set(invoices_all[0])

    def test_limit_is_honored_by_the_api(self, live, invoices_all):
        """Unlike /benchmarks/, this endpoint cuts the result itself."""
        assert len(invoices_all) > 1
        assert len(live(tools.search_invoices, limit=1)) == 1

    def test_window_narrows(self, live, invoices_all):
        narrowed = live(
            tools.search_invoices, start_date=INVOICE_WINDOW_START, limit=INVOICE_PAGE,
        )
        assert 0 < len(narrowed) < len(invoices_all)

    def test_flag_filter_reaches_the_api(self, live, invoices_all):
        credits = live(tools.search_invoices, is_credit=True, limit=INVOICE_PAGE)
        assert all(invoice["is_credit"] for invoice in credits)
        assert len(credits) <= len(invoices_all)

    def test_the_removed_type_param_is_still_a_no_op(self, live, invoices_all):
        """It was dropped because the API ignores it - a value it cannot know changes
        nothing. If this ever goes red, type became real and belongs back in the signature."""
        ignored = live(
            _invoices_query,
            type="no_such_type", limit=INVOICE_PAGE, select_filters=json.dumps({}),
        )
        assert [row["id"] for row in ignored] == [row["id"] for row in invoices_all]

    def test_a_bogus_filter_column_is_refused_loudly(self, live):
        """select_filters is validated upstream, which is why typed params can build it."""
        with pytest.raises(APIError) as excinfo:
            live(_invoices_query, select_filters=json.dumps({"bogus": {"eq": 1}}))
        assert excinfo.value.status == 400


# -- ShowCharges --------------------

# The shape and paging probes need rows, not a particular window: 90 days back instead of
# the default 30, so a quiet month on the account does not turn them red.
CHARGES_PROBE_START = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 90 * 24 * 3600))


class TestShowCharges:
    def test_default_range_succeeds(self, live):
        """A range missing either end answers 400 'Must provide both'."""
        result = live(tools.show_charges)
        assert result["success"] is True
        assert result["count"] == len(result["results"])
        assert isinstance(result["total"], int)

    def test_rows_carry_amounts_and_nested_items(self, live):
        result = live(tools.show_charges, start_date=CHARGES_PROBE_START, limit=5)
        assert result["results"], "account has no charge in the last 90 days"
        for charge in result["results"]:
            assert {"start", "end", "type", "amount", "items"} <= set(charge)
            assert isinstance(charge["amount"], (int, float))

    def test_type_filter_narrows_to_one_kind(self, live):
        result = live(tools.show_charges, type="instance", limit=5)
        assert result["success"] is True
        assert all(charge["type"] == "instance" for charge in result["results"])

    def test_paging_moves_forward(self, live):
        first = live(tools.show_charges, start_date=CHARGES_PROBE_START, limit=1)
        assert first["next_token"], "account has fewer than two charges to page through"
        second = live(
            tools.show_charges,
            start_date=CHARGES_PROBE_START, limit=1, next_token=first["next_token"],
        )
        assert second["results"][0]["source"] != first["results"][0]["source"]

    def test_window_before_the_account_existed_is_empty(self, live):
        result = live(tools.show_charges, start_date="2000-01-01", end_date="2001-01-01")
        assert result["success"] is True
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
    def test_machine_filter_returns_only_that_machine(self, benchmarks_of_machine):
        assert benchmarks_of_machine
        assert {row["machine_id"] for row in benchmarks_of_machine} == {BENCHMARK_MACHINE_ID}
        assert {"id", "machine_id", "value"} <= set(benchmarks_of_machine[0])

    def test_column_catalog_matches_the_api(self, benchmarks_of_machine):
        """select_cols validates against this set, and an unknown name is answered with a
        null 'anon_1' column rather than an error - so the set has to track the API."""
        for row in benchmarks_of_machine:
            assert set(row) == set(_BENCHMARK_COLUMNS)

    def test_select_cols_narrows_the_row(self, live):
        rows = live(
            tools.search_benchmarks,
            machine_id=BENCHMARK_MACHINE_ID, select_cols=["id", "value"],
        )
        assert rows
        assert set(rows[0]) == {"id", "value"}

    def test_limit_cuts_what_the_api_would_not(self, live, benchmarks_of_machine):
        """The endpoint ignores limit, so the cut has to happen after the response."""
        assert len(benchmarks_of_machine) > BENCHMARK_LIMIT
        rows = live(
            tools.search_benchmarks,
            machine_id=BENCHMARK_MACHINE_ID, limit=BENCHMARK_LIMIT,
        )
        assert len(rows) == BENCHMARK_LIMIT


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


# -- Paid rental harness --------------------

# A ~250MB image keeps both the pull and the 8GB disk small, and nvidia/cuda declares
# NVIDIA_DRIVER_CAPABILITIES, which is what makes nvidia-smi appear inside the container.
E2E_IMAGE = "nvidia/cuda:12.0.1-base-ubuntu22.04"
# The image is refused by the container runtime on a host whose driver is older than this.
E2E_IMAGE_CUDA = 12.0
# Entrypoint launch mode keeps the image's own entrypoint, so the container needs a command
# that does not exit. ExecuteCommand runs neither: the API allows only ls, rm and du, and
# only on a stopped instance. So nvidia-smi -L runs at startup and comes back through
# ShowLogs, and the probe file it leaves is what the constrained ls reads back later.
E2E_RUNTYPE = "args"
LOG_MARKER = "mcp-e2e-online"
PROBE_NAME = "mcp-e2e-probe.txt"
E2E_ARGS = f'bash -c "nvidia-smi -L | tee /{PROBE_NAME}; echo {LOG_MARKER}; sleep infinity"'

DISK_GB = 8.0
MAX_DPH = 0.15
OFFER_PAGE = 20
# Cheapest hosts are the flakiest: one took the contract and never left cur_state=running
# with actual_status=None. Each dud is destroyed before the next-cheapest offer gets a turn.
RENT_ATTEMPTS = 3

POLL_INTERVAL_S = 10.0
RUNNING_TIMEOUT_S = 8 * 60.0
STOPPED_TIMEOUT_S = 4 * 60.0
RESULT_TIMEOUT_S = 2 * 60.0
GONE_TIMEOUT_S = 2 * 60.0


def _poll(
    probe: Any, ready: Any, *, interval_s: float, timeout_s: float, what: str,
    sleep: Any = time.sleep, clock: Any = time.monotonic, raise_on_timeout: bool = True,
) -> Any:
    """Polls probe() until ready() holds and returns that value.

    On the cap it raises with the last value seen, so a stuck rental says what it was stuck
    on instead of hanging the run until the CI timeout kills it. With raise_on_timeout=False
    the cap returns None instead: the caller retries on another offer, and an except here
    would swallow.
    """
    deadline = clock() + timeout_s
    while True:
        value = probe()
        if ready(value):
            return value
        if clock() >= deadline:
            if not raise_on_timeout:
                return None
            raise TimeoutError(f"{what}: gave up after {timeout_s:.0f}s, last seen {value!r}")
        sleep(interval_s)


def _pick_offers(offers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Up to RENT_ATTEMPTS cheapest offers the rental can run on, every one under the price
    cap. Raises before anything is rented."""
    usable = [
        o for o in offers
        if o["disk_space"] >= DISK_GB and o["cuda_max_good"] >= E2E_IMAGE_CUDA
    ]
    assert usable, (
        f"none of the {len(offers)} cheapest offers has {DISK_GB:.0f}GB disk "
        f"and CUDA {E2E_IMAGE_CUDA}"
    )
    usable.sort(key=lambda o: o["dph_total"])
    assert usable[0]["dph_total"] <= MAX_DPH, (
        f"cheapest usable offer is ${usable[0]['dph_total']:.4f}/hr, over the ${MAX_DPH} cap - "
        f"renting nothing"
    )
    return [o for o in usable[:RENT_ATTEMPTS] if o["dph_total"] <= MAX_DPH]


class _Rental:
    """One rented instance, destroyed at most once.

    The flag is set before the call so a destroy that fails is not repeated by every
    teardown layer in turn; a rental that survives it is left to dev.py sweep.
    """

    def __init__(self, live: Any, instance_id: int, label: str) -> None:
        self._live = live
        self.id = instance_id
        self.label = label
        self.destroyed = False

    def destroy(self) -> Any:
        if self.destroyed:
            return None
        self.destroyed = True
        try:
            return self._live(tools.destroy_instance, id=self.id)
        except APIError as e:
            # no-report: vast.ai reaps a dud under us, and the listing proves it really went
            reaped = isinstance(e.body, dict) and e.body.get("error") == "no_such_instance"
            if not (e.status == 404 and reaped):
                raise
            # Same 404 answers "already reaped" and "never yours" (wrong id or account); only
            # the listing separates them, and guessing wrong keeps renting instances we think
            # we destroyed.
            if _find_instance(self._live, self.id) is not None:
                raise
            warnings.warn(f"instance {self.id} was already gone", stacklevel=2)
            return None


class _Reaper:
    """Teardown for a run that dies before the fixture finally: atexit covers a normal exit,
    the handlers cover SIGINT and SIGTERM, which skip atexit entirely."""

    def __init__(self) -> None:
        self._rentals: list[_Rental] = []
        self._installed = False

    def watch(self, rental: _Rental) -> None:
        if not self._installed:
            self._install()
        self._rentals.append(rental)

    def _install(self) -> None:
        atexit.register(self.reap)
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self._on_signal)
        self._installed = True

    def reap(self) -> None:
        while self._rentals:
            self._rentals.pop().destroy()

    def _on_signal(self, signum: int, frame: Any) -> None:
        self.reap()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


_REAPER = _Reaper()


@contextlib.contextmanager
def _rented(create: Any, reaper: _Reaper) -> Any:
    rental = create()
    reaper.watch(rental)
    try:
        yield rental
    finally:
        rental.destroy()


# -- Paid rental harness, offline --------------------

class _FakeLive:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def __call__(self, fn, **kwargs):
        self.calls.append((fn, kwargs))
        return {"success": True}


def _gone_404() -> APIError:
    return APIError(404, "DELETE", "/i/42/", {"success": False, "error": "no_such_instance"})


class _ReapedLive:
    """live() stand-in for the destroy paths: destroys raise, the listing shows `listed`."""

    def __init__(self, err: APIError, listed: tuple[int, ...] = ()) -> None:
        self.err = err
        self.listed = listed

    def __call__(self, fn, **kwargs):
        if fn is tools.list_instances:
            return {"instances": [{"id": i} for i in self.listed]}
        raise self.err


class _FakeProbe:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.values.pop(0)


class _FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def _offer(**overrides):
    offer = {"id": 1, "dph_total": 0.05, "disk_space": 100.0, "cuda_max_good": 12.8}
    offer.update(overrides)
    return offer


class TestPoll:
    def test_ready_on_the_first_probe_never_sleeps(self):
        clock = _FakeClock()
        value = _poll(
            lambda: "running", lambda v: v == "running",
            interval_s=POLL_INTERVAL_S, timeout_s=RUNNING_TIMEOUT_S, what="never",
            sleep=clock.sleep, clock=clock,
        )
        assert value == "running"
        assert clock.slept == []

    def test_sleeps_the_interval_between_probes(self):
        probe = _FakeProbe(["loading", "loading", "running"])
        clock = _FakeClock()
        value = _poll(
            probe, lambda v: v == "running",
            interval_s=POLL_INTERVAL_S, timeout_s=RUNNING_TIMEOUT_S, what="never",
            sleep=clock.sleep, clock=clock,
        )
        assert value == "running"
        assert clock.slept == [POLL_INTERVAL_S, POLL_INTERVAL_S]

    def test_gives_up_at_the_cap(self):
        """A rental that never comes up must fail on its own, well inside the CI timeout."""
        clock = _FakeClock()
        probe = _FakeProbe(["loading"] * 200)
        with pytest.raises(TimeoutError, match="never ran: gave up after 480s"):
            _poll(
                probe, lambda v: v == "running",
                interval_s=POLL_INTERVAL_S, timeout_s=RUNNING_TIMEOUT_S, what="never ran",
                sleep=clock.sleep, clock=clock,
            )
        assert clock.now == RUNNING_TIMEOUT_S
        assert probe.calls == int(RUNNING_TIMEOUT_S / POLL_INTERVAL_S) + 1

    def test_the_message_carries_the_last_state(self):
        clock = _FakeClock()
        with pytest.raises(TimeoutError, match="last seen 'created'"):
            _poll(
                lambda: "created", lambda v: False,
                interval_s=POLL_INTERVAL_S, timeout_s=POLL_INTERVAL_S, what="never ran",
                sleep=clock.sleep, clock=clock,
            )

    def test_timeout_returns_none_when_asked_not_to_raise(self):
        clock = _FakeClock()
        value = _poll(
            lambda: "loading", lambda v: v == "running",
            interval_s=POLL_INTERVAL_S, timeout_s=POLL_INTERVAL_S, what="never ran",
            sleep=clock.sleep, clock=clock, raise_on_timeout=False,
        )
        assert value is None


class TestPickOffers:
    def test_cheapest_first(self):
        picked = _pick_offers([_offer(id=2, dph_total=0.09), _offer(id=3, dph_total=0.04)])
        assert [o["id"] for o in picked] == [3, 2]

    def test_caps_the_number_of_attempts(self):
        offers = [_offer(id=n, dph_total=0.01 * n) for n in range(1, 6)]
        assert [o["id"] for o in _pick_offers(offers)] == [1, 2, 3]

    def test_skips_a_disk_too_small_for_the_image(self):
        cheap_and_full = _offer(id=2, dph_total=0.01, disk_space=DISK_GB - 1)
        assert [o["id"] for o in _pick_offers([cheap_and_full, _offer(id=3)])] == [3]

    def test_skips_a_host_whose_driver_predates_the_image(self):
        cheap_and_old = _offer(id=2, dph_total=0.01, cuda_max_good=11.8)
        assert [o["id"] for o in _pick_offers([cheap_and_old, _offer(id=3)])] == [3]

    def test_refuses_to_rent_over_the_cap(self):
        """The price gate has to fail before CreateInstance, not after."""
        with pytest.raises(AssertionError, match="over the .0.15 cap"):
            _pick_offers([_offer(dph_total=MAX_DPH + 0.01)])

    def test_over_cap_runners_up_are_not_attempted(self):
        """The cap holds for every attempt, not only the first."""
        picked = _pick_offers([_offer(id=1, dph_total=0.14), _offer(id=2, dph_total=0.16)])
        assert [o["id"] for o in picked] == [1]

    def test_refuses_when_nothing_fits(self):
        with pytest.raises(AssertionError, match="8GB disk"):
            _pick_offers([_offer(disk_space=1.0)])


@pytest.fixture
def quiet_signals(monkeypatch):
    """Keeps the offline tests from touching the process-wide handlers."""
    installed = {"atexit": [], "signals": {}}
    monkeypatch.setattr(atexit, "register", installed["atexit"].append)
    monkeypatch.setattr(signal, "signal", installed["signals"].__setitem__)
    return installed


@pytest.mark.usefixtures("quiet_signals")
class TestTeardownLayers:
    def test_destroy_is_idempotent(self):
        live = _FakeLive()
        rental = _Rental(live, 42, "mcp-e2e-x")
        rental.destroy()
        rental.destroy()
        assert live.calls == [(tools.destroy_instance, {"id": 42})]

    def test_destroy_accepts_a_dud_the_listing_confirms_is_gone(self):
        """The reap is the one tolerated failure, and only once the listing backs it."""
        live = _ReapedLive(_gone_404())
        with pytest.warns(UserWarning, match="42 was already gone"):
            assert _Rental(live, 42, "mcp-e2e-x").destroy() is None

    def test_destroy_raises_when_the_listing_still_shows_the_instance(self):
        """no_such_instance on an id that is still listed means wrong id or wrong account."""
        live = _ReapedLive(_gone_404(), listed=(42,))
        with pytest.raises(APIError):
            _Rental(live, 42, "mcp-e2e-x").destroy()

    def test_destroy_raises_on_no_such_instance_at_another_status(self):
        live = _ReapedLive(APIError(500, "DELETE", "/i/42/", {"error": "no_such_instance"}))
        with pytest.raises(APIError):
            _Rental(live, 42, "mcp-e2e-x").destroy()

    def test_destroy_raises_on_a_404_whose_body_is_not_json(self):
        """A proxy 404 arrives as text; probing it for an error key must not mask the APIError."""
        live = _ReapedLive(APIError(404, "DELETE", "/i/42/", "<html>not found</html>"))
        with pytest.raises(APIError):
            _Rental(live, 42, "mcp-e2e-x").destroy()

    def test_destroy_raises_on_a_404_that_is_not_no_such_instance(self):
        live = _ReapedLive(APIError(404, "DELETE", "/i/42/", {"error": "no_such_route"}))
        with pytest.raises(APIError):
            _Rental(live, 42, "mcp-e2e-x").destroy()

    def test_a_raising_destroy_is_not_retried(self):
        """The flag is set before the call, so no teardown layer re-attempts a failed destroy."""
        rental = _Rental(_ReapedLive(APIError(500, "DELETE", "/i/42/", {"error": "boom"})),
                         42, "mcp-e2e-x")
        with pytest.raises(APIError):
            rental.destroy()
        assert rental.destroy() is None

    def test_the_body_raising_still_destroys(self):
        """The whole point of the finally layer: a red test must not leave a GPU running."""
        live, reaper = _FakeLive(), _Reaper()
        with pytest.raises(RuntimeError, match="boom"), _rented(
            lambda: _Rental(live, 42, "mcp-e2e-x"), reaper,
        ):
            raise RuntimeError("boom")
        assert live.calls == [(tools.destroy_instance, {"id": 42})]

    def test_watch_installs_atexit_and_signal_handlers(self, quiet_signals):
        reaper = _Reaper()
        reaper.watch(_Rental(_FakeLive(), 42, "mcp-e2e-x"))
        assert quiet_signals["atexit"] == [reaper.reap]
        assert set(quiet_signals["signals"]) == {signal.SIGINT, signal.SIGTERM}

    def test_reap_destroys_what_the_fixture_never_reached(self):
        live, reaper = _FakeLive(), _Reaper()
        reaper.watch(_Rental(live, 42, "mcp-e2e-x"))
        reaper.reap()
        reaper.reap()
        assert live.calls == [(tools.destroy_instance, {"id": 42})]

    def test_a_signal_destroys_then_dies_by_default(self, quiet_signals, monkeypatch):
        """Ctrl-C skips atexit, so the handler destroys first and only then lets the signal
        through with its default disposition."""
        killed = []
        monkeypatch.setattr(os, "kill", lambda pid, signum: killed.append((pid, signum)))
        live, reaper = _FakeLive(), _Reaper()
        reaper.watch(_Rental(live, 42, "mcp-e2e-x"))
        reaper._on_signal(signal.SIGTERM, None)
        assert live.calls == [(tools.destroy_instance, {"id": 42})]
        assert quiet_signals["signals"][signal.SIGTERM] is signal.SIG_DFL
        assert killed == [(os.getpid(), signal.SIGTERM)]


class _FakeRentalApi:
    """live() stand-in: contracts are handed out in order, the listing shows only ids told
    to be up, destroys are recorded."""

    def __init__(self, contracts: list[int], running: set[int]) -> None:
        self.contracts = list(contracts)
        self.running = running
        self.destroyed: list[int] = []

    def __call__(self, fn, **kwargs):
        if fn is tools.create_instance:
            return {"success": True, "new_contract": self.contracts.pop(0)}
        if fn is tools.list_instances:
            return {"instances": [
                {"id": i, "actual_status": "running"} for i in sorted(self.running)
            ]}
        if fn is tools.destroy_instance:
            self.destroyed.append(kwargs["id"])
            return {"success": True}
        raise AssertionError(f"unexpected live call: {fn.__name__}")


@pytest.mark.usefixtures("quiet_signals")
class TestRentRunning:
    def test_first_offer_up_rents_once(self):
        api = _FakeRentalApi(contracts=[101], running={101})
        clock = _FakeClock()
        rental = _rent_running(api, [_offer(id=1)], _Reaper(), sleep=clock.sleep, clock=clock)
        assert rental.id == 101
        assert api.destroyed == []

    def test_dud_is_destroyed_and_the_next_offer_rented(self):
        api = _FakeRentalApi(contracts=[101, 102], running={102})
        clock = _FakeClock()
        rental = _rent_running(
            api, [_offer(id=1), _offer(id=2)], _Reaper(), sleep=clock.sleep, clock=clock,
        )
        assert rental.id == 102
        assert api.destroyed == [101]

    def test_all_duds_are_destroyed_and_the_failure_is_loud(self):
        """The retry must not leak the rentals it gave up on."""
        api = _FakeRentalApi(contracts=[101, 102, 103], running=set())
        clock = _FakeClock()
        with pytest.raises(TimeoutError, match="none of 3 offers"):
            _rent_running(
                api, [_offer(id=n) for n in (1, 2, 3)], _Reaper(),
                sleep=clock.sleep, clock=clock,
            )
        assert api.destroyed == [101, 102, 103]


class TestE2eSweepsFirst:
    def test_sweep_runs_before_the_suite(self, monkeypatch):
        order = []
        monkeypatch.setattr(dev, "sweep", lambda: order.append("sweep") or 0)
        monkeypatch.setattr(dev, "_pytest", lambda *args: order.append(args) or 0)
        assert dev.e2e() == 0
        assert order == ["sweep", ("-m", "integration")]

    def test_a_broken_sweep_aborts_the_run(self, monkeypatch):
        """Renting on top of instances we failed to account for is how bills run away.
        The sweep is a subprocess, so a broken one shows up as its exit code."""
        def unreachable(*args):
            raise AssertionError(f"pytest must not start after a broken sweep: {args}")

        monkeypatch.setattr(dev, "sweep", lambda: 1)
        monkeypatch.setattr(dev, "_pytest", unreachable)
        assert dev.e2e() == 1


# -- Paid instance lifecycle --------------------

def _rent(live, offer, label):
    result = live(
        tools.create_instance,
        id=offer["id"], image=E2E_IMAGE, disk=DISK_GB, label=label,
        runtype=E2E_RUNTYPE, args_str=E2E_ARGS,
    )
    assert result["success"] is True, result
    return _Rental(live, int(result["new_contract"]), label)


def _find_instance(live, instance_id):
    instances = live(tools.list_instances)["instances"]
    return next((i for i in instances if i["id"] == instance_id), None)


def _rent_running(
    live: Any, offers: list[dict[str, Any]], reaper: _Reaper,
    sleep: Any = time.sleep, clock: Any = time.monotonic,
) -> _Rental:
    """Rents offers cheapest-first until one reaches running; each dud is destroyed before
    the next attempt. Only the running-wait retries: a failure later in the scenario means
    the host worked, so re-renting would just hide it."""
    for offer in offers:
        label = f"{LABEL_PREFIX}{uuid.uuid4()}"
        rental = _rent(live, offer, label)
        reaper.watch(rental)
        print(f"rented {rental.id} as {label} at ${offer['dph_total']:.4f}/hr")
        record = _poll(
            lambda instance_id=rental.id: _find_instance(live, instance_id),
            lambda rec: rec is not None and rec["actual_status"] == "running",
            interval_s=POLL_INTERVAL_S, timeout_s=RUNNING_TIMEOUT_S,
            what="unreachable", sleep=sleep, clock=clock, raise_on_timeout=False,
        )
        if record is not None:
            return rental
        print(f"dud host: {rental.id} never reached running, destroying and moving on")
        rental.destroy()
    raise TimeoutError(
        f"none of {len(offers)} offers reached actual_status=running "
        f"within {RUNNING_TIMEOUT_S:.0f}s each"
    )


@pytest.fixture
def rental(live):
    """Rents the cheapest usable offer, retrying dud hosts, and destroys whatever it rented
    however the test ends.

    Ids are printed because pytest shows setup output on failure: a run that dies here
    is exactly the one whose instance ids you need.
    """
    offers = _pick_offers(live(tools.search_offers, limit=OFFER_PAGE, order="dph_total")["offers"])
    with _rented(lambda: _rent_running(live, offers, _REAPER), _REAPER) as rented:
        yield rented


class TestPaidInstanceLifecycle:
    def test_rent_run_and_destroy(self, live, rental):
        record = _poll(
            lambda: _find_instance(live, rental.id),
            lambda rec: rec is not None and rec["actual_status"] == "running",
            interval_s=POLL_INTERVAL_S, timeout_s=RUNNING_TIMEOUT_S,
            what=f"instance {rental.id} never reached actual_status=running",
        )
        assert record["label"] == rental.label, "sweep finds instances by label"
        assert record["image_uuid"] == E2E_IMAGE
        assert record["dph_total"] <= MAX_DPH

        logs = _poll(
            lambda: live(tools.show_logs, id=rental.id, tail=50),
            lambda out: isinstance(out, str) and LOG_MARKER in out,
            interval_s=POLL_INTERVAL_S, timeout_s=RESULT_TIMEOUT_S,
            what=f"logs of {rental.id} never carried the startup marker",
        )
        gpu_lines = [line for line in logs.splitlines() if line.startswith("GPU ")]
        assert len(gpu_lines) == record["num_gpus"], logs

        # The API refuses ExecuteCommand on a running instance and points at ssh instead, so
        # the box is stopped first; its disk survives the stop and still holds the probe file.
        live(tools.manage_instance, id=rental.id, state="stopped")
        _poll(
            lambda: _find_instance(live, rental.id),
            lambda rec: rec is not None and rec["actual_status"] == "exited",
            interval_s=POLL_INTERVAL_S, timeout_s=STOPPED_TIMEOUT_S,
            what=f"instance {rental.id} never reached actual_status=exited",
        )

        listing = _poll(
            lambda: live(tools.execute_command, id=rental.id, command="ls -la /"),
            lambda out: isinstance(out, str) and PROBE_NAME in out,
            interval_s=POLL_INTERVAL_S, timeout_s=RESULT_TIMEOUT_S,
            what=f"ls on {rental.id} never saw the file the container wrote",
        )
        assert PROBE_NAME in listing

        rental.destroy()
        _poll(
            lambda: _find_instance(live, rental.id),
            lambda rec: rec is None,
            interval_s=POLL_INTERVAL_S, timeout_s=GONE_TIMEOUT_S,
            what=f"instance {rental.id} still listed after DestroyInstance",
        )
