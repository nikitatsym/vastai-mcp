# vastai-mcp

MCP server for Vast.ai GPU marketplace.

- **MCP standard:** `/home/ari/src/obsidian_vault/specs/mcp-server.md` — follow it exactly (structure, registry, server dispatch, groups, config, client patterns)
- **Reference implementations:** komodo-mcp (`/home/ari/src/komodo-mcp/`), ticktick-mcp (`/home/ari/src/ticktick-mcp/`)
- **API base:** `https://console.vast.ai`, auth: `Bearer {VASTAI_API_KEY}`
- **OpenAPI spec:** `https://docs.vast.ai/api-reference/openapi.json`
- **Gitea repo:** `git.tsym.nl/ari/vastai-mcp`, CI/CD: copy from `/home/ari/src/page-mcp/.gitea/workflows/build.yml`

## Groups

```
vastai_read    — search, list, show (safe, read-only)
vastai_write   — create, update (non-destructive)
vastai_execute — reboot, recycle, execute command, copy, route
vastai_delete  — destroy, delete, cancel (destructive)
```

## Operations

Format: `function_name(params)` → `HTTP_METHOD /path` — description.

### vastai_read

| Operation | Endpoint | Notes |
|---|---|---|
| `show_user()` | `GET /api/v0/users/current/` | |
| `list_api_keys()` | `GET /api/v0/auth/apikeys/` | |
| `list_ssh_keys()` | `GET /api/v0/ssh/` | |
| `list_secrets()` | `GET /api/v0/secrets/` | |
| `search_offers(limit, gpu_name, num_gpus, gpu_ram, dph_total, reliability, geolocation, type, verified, datacenter, order, raw_query)` | `POST /api/v0/bundles/` | Main search. Params build filter objects: `gpu_name→{"eq":v}`, `gpu_ram→{"gte":v}`, `dph_total→{"lte":v}`, etc. `raw_query` overrides all. Slim results with `_slim_offer`. |
| `search_templates(select_filters, select_cols, order_by)` | `GET /api/v0/template/` | |
| `search_benchmarks(query)` | `GET /api/v0/benchmarks/` | |
| `list_instances()` | `GET /api/v0/instances/` | Slim with `_slim_instance` |
| `show_instance(id)` | `GET /api/v0/instances/{id}/` | Full, not slimmed |
| `show_instance_ssh_keys(instance_id)` | `GET /api/v0/instances/{instance_id}/ssh/` | |
| `show_logs(id, tail, filter, daemon_logs)` | `PUT /api/v0/instances/request_logs/{id}` | Returns result_url — fetch it, return content |
| `show_deposit(id)` | `GET /api/v0/instances/balance/{id}/` | |
| `search_invoices(type, select_filters)` | `GET /api/v0/invoices` | |
| `show_invoices_v1(select_filters, latest_first, limit)` | `GET /api/v1/invoices/` | |
| `list_volumes()` | `GET /api/v0/volumes/` | |
| `search_volumes(q, limit)` | `POST /api/v0/volumes/search/` | |
| `search_network_volumes(q)` | `POST /api/v0/network_volumes/search/` | |
| `list_endpoints()` | `GET /api/v0/endptjobs/` | |
| `list_workergroups()` | `GET /api/v0/workergroups/` | |
| `get_endpoint_logs(endpoint, tail)` | `POST /get_endpoint_logs/` | |
| `get_endpoint_workers(id)` | `POST /get_endpoint_workers/` | |
| `get_workergroup_logs(id, tail)` | `POST /get_workergroup_logs/` | |
| `get_workergroup_workers(id)` | `POST /get_workergroup_workers/` | |

### vastai_write

| Operation | Endpoint | Notes |
|---|---|---|
| `create_api_key(name, permissions)` | `POST /api/v0/auth/apikeys/` | |
| `create_ssh_key(ssh_key)` | `POST /api/v0/ssh/` | |
| `update_ssh_key(id, ssh_key)` | `PUT /api/v0/ssh/{id}/` | |
| `create_secret(key, value)` | `POST /api/v0/secrets/` | |
| `update_secret(key, value)` | `PUT /api/v0/secrets/` | |
| `create_instance(id, image, disk, label, onstart, env, runtype, price, args_str, use_jupyter_lab, jupyter_dir, python_utf8, lang_utf8)` | `PUT /api/v0/asks/{id}/` | `id` = offer ID |
| `manage_instance(id, state, label)` | `PUT /api/v0/instances/{id}/` | state: "running"/"stopped" |
| `change_bid(id, price)` | `PUT /api/v0/instances/bid_price/{id}/` | Body: `{"client_id":"me","price":price}` |
| `prepay_instance(id, amount)` | `PUT /api/v0/instances/prepay/{id}/` | |
| `attach_ssh_key(id, ssh_key)` | `POST /api/v0/instances/{id}/ssh/` | |
| `create_template(name, image, tag, env, onstart, runtype, desc, readme, recommended_disk_space, ssh_direct, use_ssh, private, args_str)` | `POST /api/v0/template/` | |
| `edit_template(hash_id, name, image, desc, recommended_disk_space)` | `PUT /api/v0/template/` | |
| `rent_volume(id, size)` | `PUT /api/v0/volumes/` | |
| `create_network_volume(id, size, name)` | `PUT /api/v0/network_volume/` | |
| `create_endpoint(endpoint_name, min_load, target_util, cold_mult, cold_workers, max_workers)` | `POST /api/v0/endptjobs/` | |
| `update_endpoint(id, endpoint_name, min_load, target_util, cold_mult, cold_workers, max_workers)` | `PUT /api/v0/endptjobs/{id}/` | |
| `create_workergroup(endpoint_name, endpoint_id, template_hash, template_id, search_params, launch_args, min_load, target_util, cold_mult, cold_workers, max_workers, test_workers, gpu_ram)` | `POST /api/v0/workergroups/` | |
| `update_workergroup(id, template_hash, template_id, search_params, launch_args, min_load, target_util, cold_mult, test_workers, gpu_ram, endpoint_name, endpoint_id)` | `PUT /api/v0/workergroups/{id}/` | |

### vastai_execute

| Operation | Endpoint | Notes |
|---|---|---|
| `reboot_instance(id)` | `PUT /api/v0/instances/reboot/{id}/` | docker stop/start, keeps GPU |
| `recycle_instance(id)` | `PUT /api/v0/instances/recycle/{id}/` | destroy+recreate, re-pull image |
| `execute_command(id, command)` | `PUT /api/v0/instances/command/{id}/` | Returns result_url — fetch and return |
| `copy_data(src_id, dst_id, src_path, dst_path)` | `PUT /api/v0/commands/copy_direct/` | |
| `cloud_copy(instance_id, src, dst, selected, transfer)` | `POST /api/v0/commands/rclone/` | |
| `route_request(endpoint, cost)` | `POST /route/` | |

### vastai_delete

| Operation | Endpoint | Notes |
|---|---|---|
| `destroy_instance(id)` | `DELETE /api/v0/instances/{id}/` | Irreversible |
| `delete_api_key(id)` | `DELETE /api/v0/auth/apikeys/{id}/` | |
| `delete_ssh_key(id)` | `DELETE /api/v0/ssh/{id}/` | |
| `detach_ssh_key(id, ssh_key_id)` | `DELETE /api/v0/instances/{id}/ssh/{ssh_key_id}/` | |
| `delete_secret(key)` | `DELETE /api/v0/secrets/` | Body: `{"key":key}` |
| `delete_template(template_id)` | `DELETE /api/v0/template/` | Body: `{"template_id":id}` |
| `delete_volume(id)` | `DELETE /api/v0/volumes/` | Body: `{"id":id}` |
| `unlist_volume(id)` | `POST /api/v0/volumes/unlist/` | Body: `{"id":id}` |
| `delete_endpoint(id)` | `DELETE /api/v0/endptjobs/{id}/` | |
| `delete_workergroup(id)` | `DELETE /api/v0/workergroups/{id}/` | |
| `cancel_copy(dst_id)` | `DELETE /api/v0/commands/copy_direct/` | Body: `{"dst_id":id}` |
| `cancel_sync(dst_id)` | `DELETE /api/v0/commands/rclone/` | Body: `{"dst_id":id}` |

## Slim fields

```python
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
```

## Out of scope

Machines (host-side, 10), Team (9), Network Volumes host-side (2), subaccounts/IP history/transfer credit (5). Add later if needed.
