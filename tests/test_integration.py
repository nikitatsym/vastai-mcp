"""Live read-only tests against the vast.ai API.

Nothing here creates, changes or destroys anything: search/list/show only.
No API key means every test in this file fails - a skip would hide the API
drift these tests exist to catch.
"""

import atexit
import contextlib
import os
import signal
import time
from typing import Any

import httpx
import pytest

import dev
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


# -- Paid rental harness --------------------

LABEL_PREFIX = dev.SWEEP_LABEL_PREFIX

# A ~250MB image keeps both the pull and the 8GB disk small, and nvidia/cuda declares
# NVIDIA_DRIVER_CAPABILITIES, which is what makes nvidia-smi appear inside the container.
E2E_IMAGE = "nvidia/cuda:12.0.1-base-ubuntu22.04"
# The image is refused by the container runtime on a host whose driver is older than this.
E2E_IMAGE_CUDA = 12.0
# Entrypoint launch mode keeps the image's own entrypoint, so the container needs a command
# that does not exit; the marker proves ShowLogs returns this container's stdout.
E2E_RUNTYPE = "args"
LOG_MARKER = "mcp-e2e-online"
E2E_ARGS = f'bash -c "echo {LOG_MARKER}; sleep infinity"'

DISK_GB = 8.0
MAX_DPH = 0.15
OFFER_PAGE = 20

POLL_INTERVAL_S = 10.0
RUNNING_TIMEOUT_S = 8 * 60.0
RESULT_TIMEOUT_S = 2 * 60.0
GONE_TIMEOUT_S = 2 * 60.0


def _poll(
    probe: Any, ready: Any, *, interval_s: float, timeout_s: float, what: str,
    sleep: Any = time.sleep, clock: Any = time.monotonic,
) -> Any:
    """Polls probe() until ready() holds and returns that value.

    On the cap it raises with the last value seen, so a stuck rental says what it was stuck
    on instead of hanging the run until the CI timeout kills it.
    """
    deadline = clock() + timeout_s
    while True:
        value = probe()
        if ready(value):
            return value
        if clock() >= deadline:
            raise TimeoutError(f"{what}: gave up after {timeout_s:.0f}s, last seen {value!r}")
        sleep(interval_s)


def _pick_offer(offers: list[dict[str, Any]]) -> dict[str, Any]:
    """The cheapest offer the rental can run on. Raises before anything is rented."""
    usable = [
        o for o in offers
        if o["disk_space"] >= DISK_GB and o["cuda_max_good"] >= E2E_IMAGE_CUDA
    ]
    assert usable, (
        f"none of the {len(offers)} cheapest offers has {DISK_GB:.0f}GB disk "
        f"and CUDA {E2E_IMAGE_CUDA}"
    )
    cheapest = min(usable, key=lambda o: o["dph_total"])
    assert cheapest["dph_total"] <= MAX_DPH, (
        f"cheapest usable offer is ${cheapest['dph_total']:.4f}/hr, over the ${MAX_DPH} cap - "
        f"renting nothing"
    )
    return cheapest


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
        return self._live(tools.destroy_instance, id=self.id)


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


class TestPickOffer:
    def test_takes_the_cheapest(self):
        assert _pick_offer([_offer(id=2, dph_total=0.09), _offer(id=3, dph_total=0.04)])["id"] == 3

    def test_skips_a_disk_too_small_for_the_image(self):
        cheap_and_full = _offer(id=2, dph_total=0.01, disk_space=DISK_GB - 1)
        assert _pick_offer([cheap_and_full, _offer(id=3)])["id"] == 3

    def test_skips_a_host_whose_driver_predates_the_image(self):
        cheap_and_old = _offer(id=2, dph_total=0.01, cuda_max_good=11.8)
        assert _pick_offer([cheap_and_old, _offer(id=3)])["id"] == 3

    def test_refuses_to_rent_over_the_cap(self):
        """The price gate has to fail before CreateInstance, not after."""
        with pytest.raises(AssertionError, match="over the .0.15 cap"):
            _pick_offer([_offer(dph_total=MAX_DPH + 0.01)])

    def test_refuses_when_nothing_fits(self):
        with pytest.raises(AssertionError, match="8GB disk"):
            _pick_offer([_offer(disk_space=1.0)])


class TestTeardownLayers:
    @pytest.fixture(autouse=True)
    def quiet_signals(self, monkeypatch):
        """Keeps the offline tests from touching the process-wide handlers."""
        installed = {"atexit": [], "signals": {}}
        monkeypatch.setattr(atexit, "register", installed["atexit"].append)
        monkeypatch.setattr(signal, "signal", installed["signals"].__setitem__)
        return installed

    def test_destroy_is_idempotent(self):
        live = _FakeLive()
        rental = _Rental(live, 42, "mcp-e2e-x")
        rental.destroy()
        rental.destroy()
        assert live.calls == [(tools.destroy_instance, {"id": 42})]

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


class TestE2eSweepsFirst:
    def test_sweep_runs_before_the_suite(self, monkeypatch):
        order = []
        monkeypatch.setattr(dev, "sweep", lambda: order.append("sweep") or 0)
        monkeypatch.setattr(dev, "_pytest", lambda *args: order.append(args) or 0)
        assert dev.e2e() == 0
        assert order == ["sweep", ("-m", "integration")]

    def test_a_broken_sweep_aborts_the_run(self, monkeypatch):
        """Renting on top of instances we failed to account for is how bills run away."""
        def broken_sweep():
            raise APIError(500, "GET", "/api/v0/instances/", {"msg": "down"})

        def unreachable(*args):
            raise AssertionError(f"pytest must not start after a broken sweep: {args}")

        monkeypatch.setattr(dev, "sweep", broken_sweep)
        monkeypatch.setattr(dev, "_pytest", unreachable)
        with pytest.raises(APIError):
            dev.e2e()
