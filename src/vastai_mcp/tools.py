import json
import re
import time
from datetime import UTC, datetime
from importlib.metadata import version
from typing import Annotated, Any, Literal

import httpx
from pydantic import Field

from .client import VastClient
from .registry import ROOT, Group, _op

# -- Client singleton --------------------

_client: VastClient | None = None


def _get_client() -> VastClient:
    global _client
    if _client is None:
        _client = VastClient()
    return _client


def _ok(data: Any) -> Any:
    if data is None:
        return {"status": "ok"}
    return data


# -- Slim helpers --------------------

_SLIM_OFFER_FIELDS = {
    "id", "gpu_name", "num_gpus", "gpu_ram", "gpu_total_ram",
    "cpu_cores_effective", "cpu_ram", "disk_space",
    "dph_total", "min_bid", "reliability",
    "inet_down", "inet_up", "geolocation",
    "cuda_max_good", "driver_version", "verification", "static_ip", "datacenter",
}

_SLIM_TEMPLATE_FIELDS = {
    "id", "hash_id", "name", "image", "tag", "desc",
    "recommended", "private", "count_created",
    "runtype", "recommended_disk_space", "env", "onstart",
}

_SLIM_INSTANCE_FIELDS = {
    "id", "machine_id", "actual_status", "intended_status",
    "gpu_name", "num_gpus", "gpu_ram",
    "cpu_cores_effective", "cpu_ram", "disk_space",
    "dph_total", "label", "image_uuid",
    "ssh_host", "ssh_port", "jupyter_token", "start_date", "cur_state",
}


def _body(**fields: Any) -> dict[str, Any]:
    """Request body without unset fields: the API treats an absent key as 'leave alone'."""
    return {k: v for k, v in fields.items() if v is not None}


def _slim(item: dict[str, Any], fields: set[str]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if k in fields}


def _slim_list(items: list[Any], fields: set[str]) -> list[dict[str, Any]]:
    return [_slim(i, fields) for i in items if isinstance(i, dict)]


# -- Search helpers --------------------

def _parse_ram_mb(value: str) -> float:
    """Parse '24GB' or '24564MB' -> MB. Crashes on bare numbers or missing units."""
    if isinstance(value, (int, float)):
        raise TypeError(
            f"gpu_ram={value!r} - must include units, e.g. '24GB' or '24564MB'. "
            f"Bare numbers are ambiguous."
        )
    m = _GPU_RAM_RE.match(str(value))
    if not m:
        raise ValueError(
            f"gpu_ram={value!r} - must include units, e.g. '24GB' or '24564MB'"
        )
    num = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "MB":
        return num
    return num * 1024


def _ram_mb_floor(mb: float) -> int:
    """3% below nominal - GPUs report slightly less (24GB = 24564 not 24576 MB)."""
    return round(mb * 0.97)


def _ram_mb_ceil(mb: float) -> int:
    """3% above nominal - GPUs report slightly less (24GB = 24564 not 24576 MB)."""
    return round(mb * 1.03)


def _build_offer_query(
    gpu_name: str | None = None,
    num_gpus: int | None = None,
    gpu_ram_min_mb: float | None = None,
    gpu_ram_max_mb: float | None = None,
    dph_total: float | None = None,
    reliability: float | None = None,
    geolocation: str | None = None,
    type: Any = None,
    verified: bool | None = None,
    datacenter: bool | None = None,
) -> dict[str, Any]:
    q: dict[str, Any] = {"verified": {"eq": True}, "external": {"eq": False}, "rentable": {"eq": True}, "rented": {"eq": False}}
    if gpu_name is not None:
        q["gpu_name"] = {"eq": gpu_name}
    if num_gpus is not None:
        q["num_gpus"] = {"eq": num_gpus}
    if gpu_ram_min_mb is not None or gpu_ram_max_mb is not None:
        ram = {}
        if gpu_ram_min_mb is not None:
            ram["gte"] = _ram_mb_floor(gpu_ram_min_mb)
        if gpu_ram_max_mb is not None:
            ram["lte"] = _ram_mb_ceil(gpu_ram_max_mb)
        q["gpu_ram"] = ram
    if dph_total is not None:
        q["dph_total"] = {"lte": dph_total}
    if reliability is not None:
        q["reliability2"] = {"gte": reliability}
    if geolocation is not None:
        q["geolocation"] = {"eq": geolocation}
    if type is not None:
        q["type"] = type
    if verified is not None:
        q["verified"] = {"eq": verified}
    if datacenter is not None:
        q["datacenter"] = {"eq": datacenter}
    return q


def _parse_order(order: str | None) -> list[list[str]] | None:
    if not order:
        return None
    if order.startswith("-"):
        return [[order[1:], "desc"]]
    parts = order.rsplit("-", 1)
    if len(parts) == 2 and parts[1] in ("asc", "desc"):
        return [[parts[0], parts[1]]]
    return [[order, "asc"]]


def _parse_order_by(order: str | None) -> list[dict[str, str]] | None:
    """Same input as _parse_order, but /template/ wants {"col","dir"} dicts."""
    pairs = _parse_order(order)
    if pairs is None:
        return None
    return [{"col": col, "dir": direction} for col, direction in pairs]


_GPU_NAMES: list[str] | None = None


def _gpu_names() -> list[str]:
    global _GPU_NAMES
    if _GPU_NAMES is None:
        result = _get_client().get("/api/v0/gpu_names/unique/")
        _GPU_NAMES = result["gpu_names"]
    return _GPU_NAMES


def _normalize_gpu_name(name: str) -> str:
    return re.sub(r"[\s_-]+", " ", name).strip().lower()


def _resolve_gpu_name(name: str) -> str:
    """CLI spelling ('RTX_4090') to API spelling ('RTX 4090'). Unknown name crashes.

    The API silently returns zero offers for a name it doesn't know, so match
    against the catalog first.
    """
    key = _normalize_gpu_name(name)
    known = _gpu_names()
    for candidate in known:
        if _normalize_gpu_name(candidate) == key:
            return candidate
    raise ValueError(
        f"gpu_name={name!r} is not a known GPU. Valid names: {', '.join(known)}"
    )


def _parse_date_ts(value: str | int, field: str) -> int:
    """Parse 'YYYY-MM-DD' or epoch seconds into epoch seconds."""
    if isinstance(value, bool):
        raise TypeError(f"{field}={value!r} - expected 'YYYY-MM-DD' or epoch seconds")
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    try:
        return int(datetime.strptime(text + "+0000", "%Y-%m-%d%z").timestamp())
    except ValueError:
        raise ValueError(
            f"{field}={value!r} - expected 'YYYY-MM-DD' or epoch seconds"
        ) from None


# -- Validation helpers --------------------

_GPU_RAM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(GB|MB)\s*$", re.IGNORECASE)


def _parse_gpu_ram(value: str, target_unit: str) -> float:
    """Parse '48GB' or '49152MB' into target unit. Raises ValueError without unit suffix."""
    m = _GPU_RAM_RE.match(value)
    if not m:
        raise ValueError(
            f"gpu_ram={value!r} - must include units, e.g. '48GB' or '49152MB'"
        )
    num = float(m.group(1))
    unit = m.group(2).upper()
    if unit == target_unit:
        return num
    if unit == "GB" and target_unit == "MB":
        return num * 1024
    # MB -> GB
    return num / 1024


def _validate_env(env: str) -> None:
    """Raise ValueError if env string has common mistakes."""
    errors = []
    if "-p " in env or env.endswith("-p"):
        errors.append(
            "'-p PORT:PORT' in env does NOT create port mappings - "
            "vast.ai parses it as a broken env var. "
            "Ports are mapped automatically for serverless, or use direct_port_count for instances"
        )
    # Check for -e values with unquoted spaces: -e KEY=val1 val2
    for m in re.finditer(r"-e\s+(\S+=\S*)\s+([^-]\S*)", env):
        key_val, orphan = m.group(1), m.group(2)
        if not orphan.startswith("-"):
            errors.append(
                f"'{key_val}' is followed by '{orphan}' which will be lost - "
                f"vast.ai splits on spaces. Use '-e {key_val} {orphan}' won't work either. "
                f"Combine into the value without spaces or use a different approach"
            )
    if errors:
        raise ValueError(f"env string is invalid: {'; '.join(errors)}")


_SEARCH_GPU_RAM_RE = re.compile(r"gpu_ram\s*[><=!]+\s*(\S+)")


def _validate_search_params(params: str) -> str:
    """Validate and normalize search_params. Returns cleaned string."""
    errors = []
    if "rentable" not in params:
        errors.append("missing 'rentable=true' - engine will try to rent unrentable machines")
    if "rented" not in params:
        errors.append("missing 'rented=false' - engine will try to rent already-rented machines")
    # gpu_ram in search_params is interpreted as GB by the API.
    # Require explicit units to prevent e.g. gpu_ram>=10000 (=10TB) mistakes.
    ram_m = _SEARCH_GPU_RAM_RE.search(params)
    if ram_m:
        val = ram_m.group(1)
        if not _GPU_RAM_RE.match(val):
            errors.append(
                f"gpu_ram value '{val}' must include units (e.g. '12GB' or '12288MB'). "
                f"API interprets bare numbers as GB - gpu_ram>=10000 means 10TB"
            )
    if errors:
        raise ValueError(
            f"search_params={params!r} is invalid: {'; '.join(errors)}. "
            f"API default is 'verified=true rentable=true rented=false' - your params REPLACE it entirely."
        )
    # Convert gpu_ram units to bare GB number for API
    if ram_m:
        gb = _parse_gpu_ram(ram_m.group(1), "GB")
        params = params[:ram_m.start(1)] + str(gb) + params[ram_m.end(1):]
    return params



# -- Result URL helper --------------------

def _fetch_result(result: dict[str, Any] | None) -> Any:
    """Fetch async result from result_url, polling until ready.

    vast.ai API is async: PUT triggers log/command collection, result
    appears on S3 after a delay.  Official CLI polls 30x0.3s - we match that.
    """
    if not isinstance(result, dict):
        return result
    url = result.get("result_url")
    if not url:
        return result

    for _ in range(30):
        time.sleep(0.3)
        r = httpx.get(url, timeout=60.0)
        if r.status_code == 200:
            if not r.content:
                return None
            ct = r.headers.get("content-type", "")
            if "application/json" in ct:
                return r.json()
            return re.sub(r'\n\s*\n', '\n', r.text)

    return result.get("msg", f"Logs not available (result_url returned {r.status_code})")


# -- Shared param descriptions --------------------

_GPU_RAM_UNITS = "Units required: '24GB' or '24564MB'."
_ORDER_FORMAT = "Column name, '-column' or 'column-desc'"
_DATE_FORMAT = "'YYYY-MM-DD' or epoch seconds."
_WG_GPU_RAM = "String with units, e.g. '48GB' or '49152MB' (the API takes GB)."
_WG_SEARCH_PARAMS = (
    "CLI-format string, e.g. 'gpu_name=RTX_3060 gpu_ram>=12GB verified=true "
    "rentable=true rented=false'. GPU names use underscores, not spaces. "
    "Your params REPLACE the API default 'verified=true rentable=true rented=false' "
    "entirely, so they MUST keep 'rentable=true rented=false' or the engine will try "
    "to rent unrentable and already-rented machines."
)
_WG_LAUNCH_ARGS = "CLI-format string, e.g. '--model /model --ctx 4096'."
_TEMPLATE_ENV = (
    "Docker flags STRING, not a dict, e.g. '-e VAR=val -e FOO=bar'. "
    "'-p PORT:PORT' does NOT create port mappings - vast.ai parses it as a broken env "
    "var; ports are mapped automatically for serverless, use direct_port_count for "
    "instances. A value with spaces after '-e KEY=val' is split and lost."
)


# -- Groups --------------------

vastai_read = Group(
    "vastai_read",
    "Query Vast.ai data (safe, read-only).\n\n"
    "Call with operation=\"help\" to list all available read operations.\n"
    "Otherwise pass the operation name and a JSON object with parameters.\n\n"
    "Example: vastai_read(operation=\"SearchOffers\", "
    "params={\"gpu_name\": \"RTX 4090\", \"limit\": 10})",
)

vastai_write = Group(
    "vastai_write",
    "Create or update Vast.ai resources (non-destructive).\n\n"
    "Call with operation=\"help\" to list all available write operations.\n"
    "Otherwise pass the operation name and a JSON object with parameters.\n\n"
    "Example: vastai_write(operation=\"CreateInstance\", "
    "params={\"id\": 12345, \"image\": \"pytorch/pytorch\", \"disk\": 20})",
)

vastai_execute = Group(
    "vastai_execute",
    "Execute actions on Vast.ai resources (reboot, run commands, copy data).\n\n"
    "Call with operation=\"help\" to list all available execute operations.\n"
    "Otherwise pass the operation name and a JSON object with parameters.\n\n"
    "Example: vastai_execute(operation=\"ExecuteCommand\", "
    "params={\"id\": 12345, \"command\": \"nvidia-smi\"})",
)

vastai_delete = Group(
    "vastai_delete",
    "Delete Vast.ai resources (destructive, irreversible).\n\n"
    "Call with operation=\"help\" to list all available delete operations.\n"
    "Otherwise pass the operation name and a JSON object with parameters.\n\n"
    "Example: vastai_delete(operation=\"DestroyInstance\", "
    "params={\"id\": 12345})",
)


# -- ROOT --------------------

@_op(ROOT)
def vastai_version() -> Any:
    """Get the Vast.ai MCP server version and service status.

    An unreachable API surfaces as APIError, not as a status field."""
    _get_client().get("/api/v0/users/current/")
    return {"mcp": version("vastai-mcp"), "service": {"status": "ok"}}


# -- vastai_read --------------------

@_op(vastai_read)
def show_user() -> Any:
    """Get current user info."""
    return _get_client().get("/api/v0/users/current/")


@_op(vastai_read)
def list_api_keys() -> Any:
    """List API keys."""
    return _get_client().get("/api/v0/auth/apikeys/")


@_op(vastai_read)
def list_ssh_keys() -> Any:
    """List SSH keys."""
    return _get_client().get("/api/v0/ssh/")


@_op(vastai_read)
def list_secrets() -> Any:
    """List secrets."""
    return _get_client().get("/api/v0/secrets/")


@_op(vastai_read)
def search_offers(
    limit: int = 20,
    gpu_name: Annotated[str | None, Field(description=(
        "Either spelling works ('RTX 4090' or 'RTX_4090'); ListGpuNames has the catalog."
    ))] = None,
    num_gpus: int | None = None,
    gpu_ram_min: Annotated[str | None, Field(description=_GPU_RAM_UNITS)] = None,
    gpu_ram_max: Annotated[str | None, Field(description=_GPU_RAM_UNITS)] = None,
    dph_total: Annotated[float | None, Field(description="Price ceiling in $/hr.")] = None,
    reliability: float | None = None,
    geolocation: str | None = None,
    type: Literal["on-demand", "bid", "interruptible"] | None = None,
    verified: bool | None = None,
    datacenter: bool | None = None,
    order: Annotated[str | None, Field(
        description=f"{_ORDER_FORMAT}, e.g. '-reliability'.",
    )] = None,
) -> Any:
    """Search GPU offers."""
    ram_min_mb = _parse_ram_mb(gpu_ram_min) if gpu_ram_min is not None else None
    ram_max_mb = _parse_ram_mb(gpu_ram_max) if gpu_ram_max is not None else None
    if gpu_name is not None:
        gpu_name = _resolve_gpu_name(gpu_name)
    q = _build_offer_query(
        gpu_name=gpu_name, num_gpus=num_gpus,
        gpu_ram_min_mb=ram_min_mb, gpu_ram_max_mb=ram_max_mb,
        dph_total=dph_total, reliability=reliability, geolocation=geolocation,
        type=type, verified=verified, datacenter=datacenter,
    )
    q["limit"] = int(limit)
    order_val = _parse_order(order)
    if order_val:
        q["order"] = order_val
    result = _get_client().post("/api/v0/bundles/", json=q)
    if isinstance(result, dict) and "offers" in result:
        result["offers"] = _slim_list(result["offers"], _SLIM_OFFER_FIELDS)
    return _ok(result)


@_op(vastai_read)
def list_gpu_names() -> Any:
    """List GPU names accepted by SearchOffers."""
    return _ok({"gpu_names": _gpu_names()})


@_op(vastai_read)
def search_templates(
    name: Annotated[str | None, Field(description="Substring match.")] = None,
    image: Annotated[str | None, Field(description="Substring match.")] = None,
    recommended: bool | None = None,
    limit: int = 20,
    order: Annotated[str | None, Field(
        description=f"{_ORDER_FORMAT}, e.g. '-count_created' for the most used.",
    )] = None,
) -> Any:
    """Search instance templates (slimmed)."""
    filters: dict[str, Any] = {}
    if name is not None:
        filters["name"] = name
    if image is not None:
        filters["image"] = image
    if recommended is not None:
        filters["recommended"] = {"eq": recommended}
    params: dict[str, Any] = {"select_filters": json.dumps(filters), "limit": int(limit)}
    order_by = _parse_order_by(order)
    if order_by:
        params["order_by"] = json.dumps(order_by)
    result = _get_client().get("/api/v0/template/", params=params)
    if isinstance(result, dict) and "templates" in result:
        result["templates"] = _slim_list(result["templates"], _SLIM_TEMPLATE_FIELDS)
    return _ok(result)


@_op(vastai_read)
def search_benchmarks(query: str | None = None) -> Any:
    """Search benchmarks."""
    params: dict[str, str] = {}
    if query is not None:
        params["q"] = query
    return _ok(_get_client().get("/api/v0/benchmarks/", params=params))


@_op(vastai_read)
def list_instances() -> Any:
    """List all rented instances (slimmed)."""
    result = _get_client().get("/api/v0/instances/")
    if isinstance(result, dict) and "instances" in result:
        result["instances"] = _slim_list(result["instances"], _SLIM_INSTANCE_FIELDS)
    return _ok(result)


@_op(vastai_read)
def show_instance(id: int) -> Any:
    """Get full instance details by ID."""
    return _ok(_get_client().get(f"/api/v0/instances/{id}/"))


@_op(vastai_read)
def show_instance_ssh_keys(instance_id: int) -> Any:
    """Get SSH keys attached to an instance."""
    return _ok(_get_client().get(f"/api/v0/instances/{instance_id}/ssh/"))


@_op(vastai_read)
def show_logs(
    id: int,
    tail: Annotated[int, Field(description="Number of lines, 0 for all.")] = 500,
    filter: Annotated[str | None, Field(
        description="Regex kept over the fetched lines.",
    )] = None,
    daemon_logs: Annotated[bool, Field(
        description="Fetch daemon/system logs instead of user logs.",
    )] = False,
) -> Any:
    """Get instance logs (async - polls S3 up to ~9s for result).

    Instance must be running - loading instances have no logs yet."""
    body: dict[str, Any] = {"tail": str(tail)}
    if daemon_logs:
        body["daemon_logs"] = True
    result = _get_client().put(f"/api/v0/instances/request_logs/{id}/", json=body)
    content = _fetch_result(result)
    if filter and isinstance(content, str):
        pattern = re.compile(filter)
        lines = content.splitlines()
        content = "\n".join(l for l in lines if pattern.search(l))
    return _ok(content)


@_op(vastai_read)
def show_deposit(id: int) -> Any:
    """Get instance deposit/balance info."""
    return _ok(_get_client().get(f"/api/v0/instances/balance/{id}/"))


@_op(vastai_read)
def search_invoices(type: str | None = None, select_filters: str | None = None) -> Any:
    """Search invoices."""
    params: dict[str, str] = {}
    if type is not None:
        params["type"] = type
    if select_filters is not None:
        params["select_filters"] = select_filters
    return _ok(_get_client().get("/api/v0/invoices/", params=params))


@_op(vastai_read)
def show_invoices_v1(
    start_date: Annotated[str | None, Field(
        description=f"{_DATE_FORMAT} Defaults to 30 days before end_date.",
    )] = None,
    end_date: Annotated[str | None, Field(
        description=f"{_DATE_FORMAT} Defaults to now.",
    )] = None,
    latest_first: bool = True,
    limit: int = 20,
    next_token: Annotated[str | None, Field(
        description="Pagination token from a previous call.",
    )] = None,
) -> Any:
    """Get invoices (v1 API, paginated).

    The range defaults to the last 30 days - the API rejects an unbounded one."""
    end_ts = _parse_date_ts(end_date, "end_date") if end_date is not None \
        else int(datetime.now(UTC).timestamp())
    start_ts = _parse_date_ts(start_date, "start_date") if start_date is not None \
        else end_ts - 30 * 24 * 60 * 60
    params: dict[str, Any] = {
        "select_filters": json.dumps({"when": {"gte": start_ts, "lte": end_ts}}),
        "limit": int(limit),
        "latest_first": "true" if latest_first else "false",
    }
    if next_token is not None:
        params["after_token"] = next_token
    return _ok(_get_client().get("/api/v1/invoices/", params=params))


@_op(vastai_read)
def list_volumes() -> Any:
    """List volumes."""
    return _ok(_get_client().get("/api/v0/volumes/"))


@_op(vastai_read)
def search_volumes(q: str | None = None, limit: int = 20) -> Any:
    """Search volumes."""
    body: dict[str, Any] = {"limit": limit}
    if q is not None:
        body["q"] = q
    return _ok(_get_client().post("/api/v0/volumes/search/", json=body))


@_op(vastai_read)
def search_network_volumes(q: str | None = None) -> Any:
    """Search network volumes."""
    body: dict[str, Any] = {}
    if q is not None:
        body["q"] = q
    return _ok(_get_client().post("/api/v0/network_volumes/search/", json=body))


@_op(vastai_read)
def list_endpoints() -> Any:
    """List serverless endpoints."""
    return _ok(_get_client().get("/api/v0/endptjobs/"))


@_op(vastai_read)
def list_workergroups() -> Any:
    """List worker groups."""
    return _ok(_get_client().get("/api/v0/workergroups/"))


@_op(vastai_read)
def get_endpoint_logs(
    endpoint: str,
    tail: Annotated[int, Field(
        description="Characters per log level. Increase for more history.",
    )] = 500,
) -> Any:
    """Get endpoint logs."""
    return _ok(_get_client().run_post(
        "/get_endpoint_logs/", json={"endpoint": endpoint, "tail": tail},
    ))


@_op(vastai_read)
def get_endpoint_workers(id: int) -> Any:
    """Get endpoint workers."""
    return _ok(_get_client().run_post("/get_endpoint_workers/", json={"id": id}))


@_op(vastai_read)
def get_workergroup_logs(
    id: int,
    tail: Annotated[int, Field(
        description="Characters per log level. Increase for more history.",
    )] = 500,
) -> Any:
    """Get worker group logs."""
    return _ok(_get_client().run_post(
        "/get_workergroup_logs/", json={"id": id, "tail": tail},
    ))


@_op(vastai_read)
def get_workergroup_workers(id: int) -> Any:
    """Get worker group workers."""
    return _ok(_get_client().run_post("/get_workergroup_workers/", json={"id": id}))


# -- vastai_write --------------------

@_op(vastai_write)
def create_api_key(name: str, permissions: str | None = None) -> Any:
    """Create an API key."""
    body: dict[str, str] = {"name": name}
    if permissions is not None:
        body["permissions"] = permissions
    return _ok(_get_client().post("/api/v0/auth/apikeys/", json=body))


@_op(vastai_write)
def create_ssh_key(ssh_key: str) -> Any:
    """Add an SSH public key."""
    return _ok(_get_client().post("/api/v0/ssh/", json={"ssh_key": ssh_key}))


@_op(vastai_write)
def update_ssh_key(id: int, ssh_key: str) -> Any:
    """Update an SSH key."""
    return _ok(_get_client().put(f"/api/v0/ssh/{id}/", json={"ssh_key": ssh_key}))


@_op(vastai_write)
def create_secret(key: str, value: str) -> Any:
    """Create a secret."""
    return _ok(_get_client().post("/api/v0/secrets/", json={"key": key, "value": value}))


@_op(vastai_write)
def update_secret(key: str, value: str) -> Any:
    """Update a secret."""
    return _ok(_get_client().put("/api/v0/secrets/", json={"key": key, "value": value}))


@_op(vastai_write)
def create_instance(
    id: Annotated[int, Field(description="Offer ID from SearchOffers.")],
    image: str,
    disk: float,
    label: str | None = None,
    onstart: str | None = None,
    env: Annotated[dict[str, str] | None, Field(description=(
        "Dict of variables, e.g. {'VAR': 'val'} - unlike templates, where env is a "
        "Docker flags string."
    ))] = None,
    runtype: str | None = None,
    price: float | None = None,
    args_str: str | None = None,
    use_jupyter_lab: bool | None = None,
    jupyter_dir: str | None = None,
    python_utf8: bool | None = None,
    lang_utf8: bool | None = None,
) -> Any:
    """Rent a GPU instance from an offer."""
    body: dict[str, Any] = {"client_id": "me", "image": image, "disk": disk}
    if label is not None:
        body["label"] = label
    if onstart is not None:
        body["onstart"] = onstart
    if env is not None:
        body["env"] = env
    if runtype is not None:
        body["runtype"] = runtype
    if price is not None:
        body["price"] = price
    if args_str is not None:
        body["args_str"] = args_str
    if use_jupyter_lab is not None:
        body["use_jupyter_lab"] = use_jupyter_lab
    if jupyter_dir is not None:
        body["jupyter_dir"] = jupyter_dir
    if python_utf8 is not None:
        body["python_utf8"] = python_utf8
    if lang_utf8 is not None:
        body["lang_utf8"] = lang_utf8
    return _ok(_get_client().put(f"/api/v0/asks/{id}/", json=body))


@_op(vastai_write)
def manage_instance(id: int, state: Literal["running", "stopped"] | None = None, label: str | None = None) -> Any:
    """Start, stop, or relabel an instance."""
    body: dict[str, str | None] = {}
    if state is not None:
        body["state"] = state
    if label is not None:
        body["label"] = label
    return _ok(_get_client().put(f"/api/v0/instances/{id}/", json=body))


@_op(vastai_write)
def change_bid(id: int, price: float) -> Any:
    """Change bid price for an interruptible instance."""
    return _ok(_get_client().put(
        f"/api/v0/instances/bid_price/{id}/",
        json={"client_id": "me", "price": price},
    ))


@_op(vastai_write)
def prepay_instance(id: int, amount: float) -> Any:
    """Prepay balance for an instance."""
    return _ok(_get_client().put(
        f"/api/v0/instances/prepay/{id}/", json={"amount": amount},
    ))


@_op(vastai_write)
def attach_ssh_key(id: int, ssh_key: str) -> Any:
    """Attach an SSH key to an instance."""
    return _ok(_get_client().post(
        f"/api/v0/instances/{id}/ssh/", json={"ssh_key": ssh_key},
    ))


@_op(vastai_write)
def create_template(
    name: str,
    image: str,
    tag: str | None = None,
    env: Annotated[str | None, Field(description=_TEMPLATE_ENV)] = None,
    onstart: str | None = None,
    runtype: str | None = None,
    desc: str | None = None,
    readme: str | None = None,
    recommended_disk_space: float | None = None,
    ssh_direct: bool | None = None,
    use_ssh: bool | None = None,
    private: bool | None = None,
    args_str: str | None = None,
) -> Any:
    """Create an instance template."""
    if env is not None:
        _validate_env(env)
    body: dict[str, Any] = {"name": name, "image": image}
    if tag is not None:
        body["tag"] = tag
    if env is not None:
        body["env"] = env
    if onstart is not None:
        body["onstart"] = onstart
    if runtype is not None:
        body["runtype"] = runtype
    if desc is not None:
        body["desc"] = desc
    if readme is not None:
        body["readme"] = readme
    if recommended_disk_space is not None:
        body["recommended_disk_space"] = recommended_disk_space
    if ssh_direct is not None:
        body["ssh_direct"] = ssh_direct
    if use_ssh is not None:
        body["use_ssh"] = use_ssh
    if private is not None:
        body["private"] = private
    if args_str is not None:
        body["args_str"] = args_str
    return _ok(_get_client().post("/api/v0/template/", json=body))


@_op(vastai_write)
def edit_template(
    hash_id: str,
    name: str | None = None,
    image: str | None = None,
    env: Annotated[str | None, Field(description=_TEMPLATE_ENV)] = None,
    desc: str | None = None,
    recommended_disk_space: float | None = None,
) -> Any:
    """Edit an existing template."""
    if env is not None:
        _validate_env(env)
    body: dict[str, Any] = {"hash_id": hash_id}
    if name is not None:
        body["name"] = name
    if image is not None:
        body["image"] = image
    if env is not None:
        body["env"] = env
    if desc is not None:
        body["desc"] = desc
    if recommended_disk_space is not None:
        body["recommended_disk_space"] = recommended_disk_space
    return _ok(_get_client().put("/api/v0/template/", json=body))


@_op(vastai_write)
def rent_volume(id: int, size: float) -> Any:
    """Rent a volume."""
    return _ok(_get_client().put("/api/v0/volumes/", json={"id": id, "size": size}))


@_op(vastai_write)
def create_network_volume(id: int, size: float, name: str | None = None) -> Any:
    """Create a network volume."""
    body: dict[str, Any] = {"id": id, "size": size}
    if name is not None:
        body["name"] = name
    return _ok(_get_client().put("/api/v0/network_volume/", json=body))


@_op(vastai_write)
def create_endpoint(
    endpoint_name: str,
    min_load: float | None = None,
    target_util: float | None = None,
    cold_mult: float | None = None,
    cold_workers: int | None = None,
    max_workers: int | None = None,
) -> Any:
    """Create a serverless endpoint."""
    body = _body(
        endpoint_name=endpoint_name, min_load=min_load, target_util=target_util,
        cold_mult=cold_mult, cold_workers=cold_workers, max_workers=max_workers,
    )
    return _ok(_get_client().post("/api/v0/endptjobs/", json=body))


@_op(vastai_write)
def update_endpoint(
    id: int,
    endpoint_name: str | None = None,
    min_load: float | None = None,
    target_util: float | None = None,
    cold_mult: float | None = None,
    cold_workers: int | None = None,
    max_workers: int | None = None,
) -> Any:
    """Update a serverless endpoint."""
    body = _body(
        endpoint_name=endpoint_name, min_load=min_load, target_util=target_util,
        cold_mult=cold_mult, cold_workers=cold_workers, max_workers=max_workers,
    )
    return _ok(_get_client().put(f"/api/v0/endptjobs/{id}/", json=body))


@_op(vastai_write)
def create_workergroup(
    endpoint_name: str,
    endpoint_id: int | None = None,
    template_hash: str | None = None,
    template_id: int | None = None,
    search_params: Annotated[str | None, Field(description=_WG_SEARCH_PARAMS)] = None,
    launch_args: Annotated[str | None, Field(description=_WG_LAUNCH_ARGS)] = None,
    min_load: float | None = None,
    target_util: float | None = None,
    cold_mult: float | None = None,
    cold_workers: int | None = None,
    max_workers: int | None = None,
    test_workers: int | None = None,
    gpu_ram: Annotated[str | None, Field(description=_WG_GPU_RAM)] = None,
) -> Any:
    """Create a worker group for a serverless endpoint."""
    if search_params is not None:
        search_params = _validate_search_params(search_params)
    gpu_ram_gb = _parse_gpu_ram(gpu_ram, "GB") if gpu_ram is not None else None
    body = _body(
        endpoint_name=endpoint_name, client_id="me", endpoint_id=endpoint_id,
        template_hash=template_hash, template_id=template_id,
        search_params=search_params, launch_args=launch_args,
        min_load=min_load, target_util=target_util, cold_mult=cold_mult,
        cold_workers=cold_workers,
    )
    # The API defaults these to 0, which starts a group that never scales up.
    body["max_workers"] = max_workers if max_workers is not None else 20
    body["test_workers"] = test_workers if test_workers is not None else 3
    if gpu_ram_gb is not None:
        body["gpu_ram"] = gpu_ram_gb
    return _ok(_get_client().post("/api/v0/workergroups/", json=body))


@_op(vastai_write)
def update_workergroup(
    id: int,
    template_hash: str | None = None,
    template_id: int | None = None,
    search_params: Annotated[str | None, Field(description=_WG_SEARCH_PARAMS)] = None,
    launch_args: Annotated[str | None, Field(description=_WG_LAUNCH_ARGS)] = None,
    min_load: float | None = None,
    target_util: float | None = None,
    cold_mult: float | None = None,
    test_workers: int | None = None,
    gpu_ram: Annotated[str | None, Field(description=_WG_GPU_RAM)] = None,
    endpoint_name: str | None = None,
    endpoint_id: int | None = None,
) -> Any:
    """Update a worker group."""
    if search_params is not None:
        search_params = _validate_search_params(search_params)
    gpu_ram_gb = _parse_gpu_ram(gpu_ram, "GB") if gpu_ram is not None else None
    body = _body(
        client_id="me", template_hash=template_hash, template_id=template_id,
        search_params=search_params, launch_args=launch_args,
        min_load=min_load, target_util=target_util, cold_mult=cold_mult,
        test_workers=test_workers, gpu_ram=gpu_ram_gb,
        endpoint_name=endpoint_name, endpoint_id=endpoint_id,
    )
    return _ok(_get_client().put(f"/api/v0/workergroups/{id}/", json=body))


# -- vastai_execute --------------------

@_op(vastai_execute)
def reboot_instance(id: int) -> Any:
    """Reboot an instance (docker stop/start, keeps GPU)."""
    return _ok(_get_client().put(f"/api/v0/instances/reboot/{id}/", json={}))


@_op(vastai_execute)
def recycle_instance(id: int) -> Any:
    """Recycle an instance (destroy+recreate, re-pulls image)."""
    return _ok(_get_client().put(f"/api/v0/instances/recycle/{id}/", json={}))


@_op(vastai_execute)
def execute_command(id: int, command: str) -> Any:
    """Execute a command on an instance (async - polls S3 up to ~9s for result)."""
    result = _get_client().put(
        f"/api/v0/instances/command/{id}/", json={"command": command},
    )
    return _ok(_fetch_result(result))


@_op(vastai_execute)
def copy_data(src_id: int, dst_id: int, src_path: str, dst_path: str) -> Any:
    """Copy data between instances."""
    return _ok(_get_client().put("/api/v0/commands/copy_direct/", json={
        "src_id": src_id, "dst_id": dst_id,
        "src_path": src_path, "dst_path": dst_path,
    }))


@_op(vastai_execute)
def cloud_copy(
    instance_id: int,
    src: str,
    dst: str,
    selected: str | None = None,
    transfer: str | None = None,
) -> Any:
    """Cloud copy (rclone) data to/from an instance."""
    body: dict[str, Any] = {"instance_id": instance_id, "src": src, "dst": dst}
    if selected is not None:
        body["selected"] = selected
    if transfer is not None:
        body["transfer"] = transfer
    return _ok(_get_client().post("/api/v0/commands/rclone/", json=body))


@_op(vastai_execute)
def route_request(
    endpoint: Annotated[str, Field(description="Endpoint name.")],
    cost: float | None = None,
) -> Any:
    """Route a request to a serverless endpoint.

    Returns the worker URL when one is available."""
    body: dict[str, Any] = {"endpoint": endpoint}
    if cost is not None:
        body["cost"] = cost
    return _ok(_get_client().run_post("/route/", json=body))


# -- vastai_delete --------------------

@_op(vastai_delete)
def destroy_instance(id: int) -> Any:
    """Destroy an instance. Irreversible."""
    return _ok(_get_client().delete(f"/api/v0/instances/{id}/"))


@_op(vastai_delete)
def delete_api_key(id: int) -> Any:
    """Delete an API key."""
    return _ok(_get_client().delete(f"/api/v0/auth/apikeys/{id}/"))


@_op(vastai_delete)
def delete_ssh_key(id: int) -> Any:
    """Delete an SSH key."""
    return _ok(_get_client().delete(f"/api/v0/ssh/{id}/"))


@_op(vastai_delete)
def detach_ssh_key(id: int, ssh_key_id: int) -> Any:
    """Detach an SSH key from an instance."""
    return _ok(_get_client().delete(f"/api/v0/instances/{id}/ssh/{ssh_key_id}/"))


@_op(vastai_delete)
def delete_secret(key: str) -> Any:
    """Delete a secret by key name."""
    return _ok(_get_client().delete("/api/v0/secrets/", json={"key": key}))


@_op(vastai_delete)
def delete_template(template_id: int) -> Any:
    """Delete a template."""
    return _ok(_get_client().delete(
        "/api/v0/template/", json={"template_id": template_id},
    ))


@_op(vastai_delete)
def delete_volume(id: int) -> Any:
    """Delete a volume."""
    return _ok(_get_client().delete("/api/v0/volumes/", json={"id": id}))


@_op(vastai_delete)
def unlist_volume(id: int) -> Any:
    """Unlist a volume from marketplace."""
    return _ok(_get_client().post("/api/v0/volumes/unlist/", json={"id": id}))


@_op(vastai_delete)
def delete_endpoint(id: int) -> Any:
    """Delete a serverless endpoint."""
    return _ok(_get_client().delete(f"/api/v0/endptjobs/{id}/"))


@_op(vastai_delete)
def delete_workergroup(id: int) -> Any:
    """Delete a worker group."""
    return _ok(_get_client().delete(f"/api/v0/workergroups/{id}/"))


@_op(vastai_delete)
def cancel_copy(dst_id: int) -> Any:
    """Cancel a direct copy operation."""
    return _ok(_get_client().delete(
        "/api/v0/commands/copy_direct/", json={"dst_id": dst_id},
    ))


@_op(vastai_delete)
def cancel_sync(dst_id: int) -> Any:
    """Cancel a cloud sync (rclone) operation."""
    return _ok(_get_client().delete(
        "/api/v0/commands/rclone/", json={"dst_id": dst_id},
    ))
