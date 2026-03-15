import re

from .client import VastClient
from .registry import ROOT, Group, _op

# ── Client singleton ──────────────────────────────────────────────────

_client: VastClient | None = None


def _get_client() -> VastClient:
    global _client
    if _client is None:
        _client = VastClient()
    return _client


def _ok(data):
    if data is None:
        return {"status": "ok"}
    return data


# ── Slim helpers ──────────────────────────────────────────────────────

_SLIM_OFFER_FIELDS = {
    "id", "gpu_name", "num_gpus", "gpu_ram", "gpu_total_ram",
    "cpu_cores_effective", "cpu_ram", "disk_space",
    "dph_total", "min_bid", "reliability",
    "inet_down", "inet_up", "geolocation",
    "cuda_max_good", "driver_version", "verification", "static_ip", "datacenter",
}

_SLIM_INSTANCE_FIELDS = {
    "id", "machine_id", "actual_status", "intended_status",
    "gpu_name", "num_gpus", "gpu_ram",
    "cpu_cores_effective", "cpu_ram", "disk_space",
    "dph_total", "label", "image_uuid",
    "ssh_host", "ssh_port", "jupyter_token", "start_date", "cur_state",
}


def _slim(item: dict, fields: set) -> dict:
    return {k: v for k, v in item.items() if k in fields}


def _slim_list(items: list, fields: set) -> list:
    return [_slim(i, fields) for i in items if isinstance(i, dict)]


# ── Search helpers ────────────────────────────────────────────────────

def _build_offer_query(
    gpu_name=None, num_gpus=None, gpu_ram=None, dph_total=None,
    reliability=None, geolocation=None, type=None, verified=None,
    datacenter=None, raw_query=None,
) -> dict:
    if raw_query is not None:
        return raw_query if isinstance(raw_query, dict) else {}
    q: dict = {"verified": {"eq": True}, "external": {"eq": False}, "rentable": {"eq": True}, "rented": {"eq": False}}
    if gpu_name is not None:
        q["gpu_name"] = {"eq": gpu_name}
    if num_gpus is not None:
        q["num_gpus"] = {"eq": num_gpus}
    if gpu_ram is not None:
        q["gpu_ram"] = {"gte": gpu_ram}
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


def _parse_order(order: str | None) -> list | None:
    if not order:
        return None
    if order.startswith("-"):
        return [[order[1:], "desc"]]
    parts = order.rsplit("-", 1)
    if len(parts) == 2 and parts[1] in ("asc", "desc"):
        return [[parts[0], parts[1]]]
    return [[order, "asc"]]


# ── Result URL helper ─────────────────────────────────────────────────

def _fetch_result(result: dict | None) -> str | dict | None:
    if not isinstance(result, dict):
        return result
    url = result.get("result_url")
    if not url:
        return result
    return _get_client().fetch_url(url)


# ── Groups ────────────────────────────────────────────────────────────

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


# ── ROOT ──────────────────────────────────────────────────────────────

@_op(ROOT)
def vastai_version():
    """Get the Vast.ai MCP server version."""
    from importlib.metadata import version
    return version("vastai-mcp")


# ── vastai_read ───────────────────────────────────────────────────────

@_op(vastai_read)
def show_user():
    """Get current user info."""
    return _get_client().get("/api/v0/users/current/")


@_op(vastai_read)
def list_api_keys():
    """List API keys."""
    return _get_client().get("/api/v0/auth/apikeys/")


@_op(vastai_read)
def list_ssh_keys():
    """List SSH keys."""
    return _get_client().get("/api/v0/ssh/")


@_op(vastai_read)
def list_secrets():
    """List secrets."""
    return _get_client().get("/api/v0/secrets/")


@_op(vastai_read)
def search_offers(
    limit: int = 20,
    gpu_name: str | None = None,
    num_gpus: int | None = None,
    gpu_ram: float | None = None,
    dph_total: float | None = None,
    reliability: float | None = None,
    geolocation: str | None = None,
    type: str | None = None,
    verified: bool | None = None,
    datacenter: bool | None = None,
    order: str | None = None,
    raw_query: dict | None = None,
):
    """Search GPU offers. gpu_ram/dph_total filter by gte/lte. raw_query overrides all filters."""
    q = _build_offer_query(
        gpu_name=gpu_name, num_gpus=num_gpus, gpu_ram=gpu_ram,
        dph_total=dph_total, reliability=reliability, geolocation=geolocation,
        type=type, verified=verified, datacenter=datacenter, raw_query=raw_query,
    )
    q["limit"] = {"eq": limit}
    order_val = _parse_order(order)
    if order_val:
        q["order"] = order_val
    result = _get_client().post("/api/v0/bundles/", json=q)
    if isinstance(result, dict) and "offers" in result:
        result["offers"] = _slim_list(result["offers"], _SLIM_OFFER_FIELDS)
    return _ok(result)


@_op(vastai_read)
def search_templates(
    select_filters: str | None = None,
    select_cols: str | None = None,
    order_by: str | None = None,
):
    """Search templates."""
    params = {}
    if select_filters is not None:
        params["select_filters"] = select_filters
    if select_cols is not None:
        params["select_cols"] = select_cols
    if order_by is not None:
        params["order_by"] = order_by
    return _ok(_get_client().get("/api/v0/template/", params=params))


@_op(vastai_read)
def search_benchmarks(query: str | None = None):
    """Search benchmarks."""
    params = {}
    if query is not None:
        params["q"] = query
    return _ok(_get_client().get("/api/v0/benchmarks/", params=params))


@_op(vastai_read)
def list_instances():
    """List all rented instances (slimmed)."""
    result = _get_client().get("/api/v0/instances/")
    if isinstance(result, dict) and "instances" in result:
        result["instances"] = _slim_list(result["instances"], _SLIM_INSTANCE_FIELDS)
    return _ok(result)


@_op(vastai_read)
def show_instance(id: int):
    """Get full instance details by ID."""
    return _ok(_get_client().get(f"/api/v0/instances/{id}/"))


@_op(vastai_read)
def show_instance_ssh_keys(instance_id: int):
    """Get SSH keys attached to an instance."""
    return _ok(_get_client().get(f"/api/v0/instances/{instance_id}/ssh/"))


@_op(vastai_read)
def show_logs(
    id: int,
    tail: int = 100,
    filter: str | None = None,
    daemon_logs: bool = False,
):
    """Get instance logs. tail: number of lines (0=all). filter: regex grep."""
    body: dict = {"tail": str(tail)}
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
def show_deposit(id: int):
    """Get instance deposit/balance info."""
    return _ok(_get_client().get(f"/api/v0/instances/balance/{id}/"))


@_op(vastai_read)
def search_invoices(type: str | None = None, select_filters: str | None = None):
    """Search invoices."""
    params = {}
    if type is not None:
        params["type"] = type
    if select_filters is not None:
        params["select_filters"] = select_filters
    return _ok(_get_client().get("/api/v0/invoices", params=params))


@_op(vastai_read)
def show_invoices_v1(
    select_filters: str | None = None,
    latest_first: bool = True,
    limit: int = 20,
):
    """Get invoices (v1 API). latest_first: sort by most recent."""
    params: dict = {"limit": limit}
    if select_filters is not None:
        params["select_filters"] = select_filters
    if latest_first:
        params["latest_first"] = "true"
    return _ok(_get_client().get("/api/v1/invoices/", params=params))


@_op(vastai_read)
def list_volumes():
    """List volumes."""
    return _ok(_get_client().get("/api/v0/volumes/"))


@_op(vastai_read)
def search_volumes(q: str | None = None, limit: int = 20):
    """Search volumes."""
    body: dict = {"limit": limit}
    if q is not None:
        body["q"] = q
    return _ok(_get_client().post("/api/v0/volumes/search/", json=body))


@_op(vastai_read)
def search_network_volumes(q: str | None = None):
    """Search network volumes."""
    body: dict = {}
    if q is not None:
        body["q"] = q
    return _ok(_get_client().post("/api/v0/network_volumes/search/", json=body))


@_op(vastai_read)
def list_endpoints():
    """List serverless endpoints."""
    return _ok(_get_client().get("/api/v0/endptjobs/"))


@_op(vastai_read)
def list_workergroups():
    """List worker groups."""
    return _ok(_get_client().get("/api/v0/workergroups/"))


@_op(vastai_read)
def get_endpoint_logs(endpoint: str, tail: int = 100):
    """Get endpoint logs. tail: number of lines."""
    return _ok(_get_client().post(
        "/get_endpoint_logs/", json={"endpoint": endpoint, "tail": tail},
    ))


@_op(vastai_read)
def get_endpoint_workers(id: int):
    """Get endpoint workers."""
    return _ok(_get_client().post("/get_endpoint_workers/", json={"id": id}))


@_op(vastai_read)
def get_workergroup_logs(id: int, tail: int = 100):
    """Get worker group logs. tail: number of lines."""
    return _ok(_get_client().post(
        "/get_workergroup_logs/", json={"id": id, "tail": tail},
    ))


@_op(vastai_read)
def get_workergroup_workers(id: int):
    """Get worker group workers."""
    return _ok(_get_client().post("/get_workergroup_workers/", json={"id": id}))


# ── vastai_write ──────────────────────────────────────────────────────

@_op(vastai_write)
def create_api_key(name: str, permissions: str | None = None):
    """Create an API key."""
    body: dict = {"name": name}
    if permissions is not None:
        body["permissions"] = permissions
    return _ok(_get_client().post("/api/v0/auth/apikeys/", json=body))


@_op(vastai_write)
def create_ssh_key(ssh_key: str):
    """Add an SSH public key."""
    return _ok(_get_client().post("/api/v0/ssh/", json={"ssh_key": ssh_key}))


@_op(vastai_write)
def update_ssh_key(id: int, ssh_key: str):
    """Update an SSH key."""
    return _ok(_get_client().put(f"/api/v0/ssh/{id}/", json={"ssh_key": ssh_key}))


@_op(vastai_write)
def create_secret(key: str, value: str):
    """Create a secret."""
    return _ok(_get_client().post("/api/v0/secrets/", json={"key": key, "value": value}))


@_op(vastai_write)
def update_secret(key: str, value: str):
    """Update a secret."""
    return _ok(_get_client().put("/api/v0/secrets/", json={"key": key, "value": value}))


@_op(vastai_write)
def create_instance(
    id: int,
    image: str,
    disk: float,
    label: str | None = None,
    onstart: str | None = None,
    env: dict | None = None,
    runtype: str | None = None,
    price: float | None = None,
    args_str: str | None = None,
    use_jupyter_lab: bool | None = None,
    jupyter_dir: str | None = None,
    python_utf8: bool | None = None,
    lang_utf8: bool | None = None,
):
    """Rent a GPU instance from an offer. id = offer ID."""
    body: dict = {"client_id": "me", "image": image, "disk": disk}
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
def manage_instance(id: int, state: str | None = None, label: str | None = None):
    """Start, stop, or relabel an instance. state: 'running' or 'stopped'."""
    body: dict = {}
    if state is not None:
        body["state"] = state
    if label is not None:
        body["label"] = label
    return _ok(_get_client().put(f"/api/v0/instances/{id}/", json=body))


@_op(vastai_write)
def change_bid(id: int, price: float):
    """Change bid price for an interruptible instance."""
    return _ok(_get_client().put(
        f"/api/v0/instances/bid_price/{id}/",
        json={"client_id": "me", "price": price},
    ))


@_op(vastai_write)
def prepay_instance(id: int, amount: float):
    """Prepay balance for an instance."""
    return _ok(_get_client().put(
        f"/api/v0/instances/prepay/{id}/", json={"amount": amount},
    ))


@_op(vastai_write)
def attach_ssh_key(id: int, ssh_key: str):
    """Attach an SSH key to an instance."""
    return _ok(_get_client().post(
        f"/api/v0/instances/{id}/ssh/", json={"ssh_key": ssh_key},
    ))


@_op(vastai_write)
def create_template(
    name: str,
    image: str,
    tag: str | None = None,
    env: dict | None = None,
    onstart: str | None = None,
    runtype: str | None = None,
    desc: str | None = None,
    readme: str | None = None,
    recommended_disk_space: float | None = None,
    ssh_direct: bool | None = None,
    use_ssh: bool | None = None,
    private: bool | None = None,
    args_str: str | None = None,
):
    """Create an instance template."""
    body: dict = {"name": name, "image": image}
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
    desc: str | None = None,
    recommended_disk_space: float | None = None,
):
    """Edit an existing template."""
    body: dict = {"hash_id": hash_id}
    if name is not None:
        body["name"] = name
    if image is not None:
        body["image"] = image
    if desc is not None:
        body["desc"] = desc
    if recommended_disk_space is not None:
        body["recommended_disk_space"] = recommended_disk_space
    return _ok(_get_client().put("/api/v0/template/", json=body))


@_op(vastai_write)
def rent_volume(id: int, size: float):
    """Rent a volume."""
    return _ok(_get_client().put("/api/v0/volumes/", json={"id": id, "size": size}))


@_op(vastai_write)
def create_network_volume(id: int, size: float, name: str | None = None):
    """Create a network volume."""
    body: dict = {"id": id, "size": size}
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
):
    """Create a serverless endpoint."""
    body: dict = {"endpoint_name": endpoint_name}
    if min_load is not None:
        body["min_load"] = min_load
    if target_util is not None:
        body["target_util"] = target_util
    if cold_mult is not None:
        body["cold_mult"] = cold_mult
    if cold_workers is not None:
        body["cold_workers"] = cold_workers
    if max_workers is not None:
        body["max_workers"] = max_workers
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
):
    """Update a serverless endpoint."""
    body: dict = {}
    if endpoint_name is not None:
        body["endpoint_name"] = endpoint_name
    if min_load is not None:
        body["min_load"] = min_load
    if target_util is not None:
        body["target_util"] = target_util
    if cold_mult is not None:
        body["cold_mult"] = cold_mult
    if cold_workers is not None:
        body["cold_workers"] = cold_workers
    if max_workers is not None:
        body["max_workers"] = max_workers
    return _ok(_get_client().put(f"/api/v0/endptjobs/{id}/", json=body))


@_op(vastai_write)
def create_workergroup(
    endpoint_name: str,
    endpoint_id: int | None = None,
    template_hash: str | None = None,
    template_id: int | None = None,
    search_params: dict | None = None,
    launch_args: dict | None = None,
    min_load: float | None = None,
    target_util: float | None = None,
    cold_mult: float | None = None,
    cold_workers: int | None = None,
    max_workers: int | None = None,
    test_workers: int | None = None,
    gpu_ram: float | None = None,
):
    """Create a worker group for a serverless endpoint."""
    body: dict = {"endpoint_name": endpoint_name}
    if endpoint_id is not None:
        body["endpoint_id"] = endpoint_id
    if template_hash is not None:
        body["template_hash"] = template_hash
    if template_id is not None:
        body["template_id"] = template_id
    if search_params is not None:
        body["search_params"] = search_params
    if launch_args is not None:
        body["launch_args"] = launch_args
    if min_load is not None:
        body["min_load"] = min_load
    if target_util is not None:
        body["target_util"] = target_util
    if cold_mult is not None:
        body["cold_mult"] = cold_mult
    if cold_workers is not None:
        body["cold_workers"] = cold_workers
    if max_workers is not None:
        body["max_workers"] = max_workers
    if test_workers is not None:
        body["test_workers"] = test_workers
    if gpu_ram is not None:
        body["gpu_ram"] = gpu_ram
    return _ok(_get_client().post("/api/v0/workergroups/", json=body))


@_op(vastai_write)
def update_workergroup(
    id: int,
    template_hash: str | None = None,
    template_id: int | None = None,
    search_params: dict | None = None,
    launch_args: dict | None = None,
    min_load: float | None = None,
    target_util: float | None = None,
    cold_mult: float | None = None,
    test_workers: int | None = None,
    gpu_ram: float | None = None,
    endpoint_name: str | None = None,
    endpoint_id: int | None = None,
):
    """Update a worker group."""
    body: dict = {}
    if template_hash is not None:
        body["template_hash"] = template_hash
    if template_id is not None:
        body["template_id"] = template_id
    if search_params is not None:
        body["search_params"] = search_params
    if launch_args is not None:
        body["launch_args"] = launch_args
    if min_load is not None:
        body["min_load"] = min_load
    if target_util is not None:
        body["target_util"] = target_util
    if cold_mult is not None:
        body["cold_mult"] = cold_mult
    if test_workers is not None:
        body["test_workers"] = test_workers
    if gpu_ram is not None:
        body["gpu_ram"] = gpu_ram
    if endpoint_name is not None:
        body["endpoint_name"] = endpoint_name
    if endpoint_id is not None:
        body["endpoint_id"] = endpoint_id
    return _ok(_get_client().put(f"/api/v0/workergroups/{id}/", json=body))


# ── vastai_execute ────────────────────────────────────────────────────

@_op(vastai_execute)
def reboot_instance(id: int):
    """Reboot an instance (docker stop/start, keeps GPU)."""
    return _ok(_get_client().put(f"/api/v0/instances/reboot/{id}/", json={}))


@_op(vastai_execute)
def recycle_instance(id: int):
    """Recycle an instance (destroy+recreate, re-pulls image)."""
    return _ok(_get_client().put(f"/api/v0/instances/recycle/{id}/", json={}))


@_op(vastai_execute)
def execute_command(id: int, command: str):
    """Execute a command on an instance. Returns command output."""
    result = _get_client().put(
        f"/api/v0/instances/command/{id}/", json={"command": command},
    )
    return _ok(_fetch_result(result))


@_op(vastai_execute)
def copy_data(src_id: int, dst_id: int, src_path: str, dst_path: str):
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
):
    """Cloud copy (rclone) data to/from an instance."""
    body: dict = {"instance_id": instance_id, "src": src, "dst": dst}
    if selected is not None:
        body["selected"] = selected
    if transfer is not None:
        body["transfer"] = transfer
    return _ok(_get_client().post("/api/v0/commands/rclone/", json=body))


@_op(vastai_execute)
def route_request(endpoint: str, cost: float | None = None):
    """Route a request to a serverless endpoint."""
    body: dict = {"endpoint": endpoint}
    if cost is not None:
        body["cost"] = cost
    return _ok(_get_client().post("/route/", json=body))


# ── vastai_delete ─────────────────────────────────────────────────────

@_op(vastai_delete)
def destroy_instance(id: int):
    """Destroy an instance. Irreversible."""
    return _ok(_get_client().delete(f"/api/v0/instances/{id}/"))


@_op(vastai_delete)
def delete_api_key(id: int):
    """Delete an API key."""
    return _ok(_get_client().delete(f"/api/v0/auth/apikeys/{id}/"))


@_op(vastai_delete)
def delete_ssh_key(id: int):
    """Delete an SSH key."""
    return _ok(_get_client().delete(f"/api/v0/ssh/{id}/"))


@_op(vastai_delete)
def detach_ssh_key(id: int, ssh_key_id: int):
    """Detach an SSH key from an instance."""
    return _ok(_get_client().delete(f"/api/v0/instances/{id}/ssh/{ssh_key_id}/"))


@_op(vastai_delete)
def delete_secret(key: str):
    """Delete a secret by key name."""
    return _ok(_get_client().delete("/api/v0/secrets/", json={"key": key}))


@_op(vastai_delete)
def delete_template(template_id: int):
    """Delete a template."""
    return _ok(_get_client().delete(
        "/api/v0/template/", json={"template_id": template_id},
    ))


@_op(vastai_delete)
def delete_volume(id: int):
    """Delete a volume."""
    return _ok(_get_client().delete("/api/v0/volumes/", json={"id": id}))


@_op(vastai_delete)
def unlist_volume(id: int):
    """Unlist a volume from marketplace."""
    return _ok(_get_client().post("/api/v0/volumes/unlist/", json={"id": id}))


@_op(vastai_delete)
def delete_endpoint(id: int):
    """Delete a serverless endpoint."""
    return _ok(_get_client().delete(f"/api/v0/endptjobs/{id}/"))


@_op(vastai_delete)
def delete_workergroup(id: int):
    """Delete a worker group."""
    return _ok(_get_client().delete(f"/api/v0/workergroups/{id}/"))


@_op(vastai_delete)
def cancel_copy(dst_id: int):
    """Cancel a direct copy operation."""
    return _ok(_get_client().delete(
        "/api/v0/commands/copy_direct/", json={"dst_id": dst_id},
    ))


@_op(vastai_delete)
def cancel_sync(dst_id: int):
    """Cancel a cloud sync (rclone) operation."""
    return _ok(_get_client().delete(
        "/api/v0/commands/rclone/", json={"dst_id": dst_id},
    ))
