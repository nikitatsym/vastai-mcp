# vastai-mcp

MCP server for Vast.ai GPU marketplace.

## Install

```json
{
  "mcpServers": {
    "vastai": {
      "command": "uvx",
      "args": ["--refresh", "--extra-index-url", "https://nikitatsym.github.io/vastai-mcp/simple", "vastai-mcp"],
      "env": {
        "VASTAI_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Groups

| Tool | Description |
|------|-------------|
| `vastai_read` | Search offers, list instances, read logs and billing (read-only) |
| `vastai_write` | Create instances, templates, endpoints (non-destructive) |
| `vastai_execute` | Reboot, disk commands, copy data |
| `vastai_delete` | Destroy instances, delete resources (destructive) |

Call any group with `operation="help"` to list available operations, or with
`operation="schema"` (add `params={"op": "<OpName>"}`) for their JSON Schema.

Vast.ai allows roughly 5 requests per 10 seconds per account. A 429 comes back as an
error carrying `retry_after`; the server never retries it silently, the caller waits.

`ExecuteCommand` is vast.ai's disk API rather than a remote shell: it takes only `ls`,
`rm` and `du`, and only while the instance is stopped.

## Development

Every gate runs through `dev.py`; CI calls the same commands. Each gate is a subprocess
of `uv run`, so `dev.py` itself needs nothing installed: a fresh clone starts with
`python dev.py hook`.

| Command | Runs |
|---------|------|
| `python dev.py hook` | point git at the tracked hook, once per clone |
| `uv run python dev.py lint` | ruff, mypy, tackbox |
| `uv run python dev.py test` | all tests, live ones included |
| `uv run python dev.py e2e` | sweep, then live tests only (`integration` marker) |
| `uv run python dev.py check` | lint + test |
| `uv run python dev.py precommit` | lint + tests without `integration` |
| `uv run python dev.py sweep` | destroy instances labeled `mcp-e2e-*` |

The pre-commit hook runs `dev.py check`, the same gate as CI and with no logic of its
own. That includes the live tests, so a commit rents a real GPU for a few minutes and
needs `VASTAI_API_KEY` in the environment; without the key the live tests fail rather
than skip. `precommit` is the same gate without them, for iterating by hand.
