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
| `vastai_read` | Search offers, list instances, get logs (read-only) |
| `vastai_write` | Create instances, templates, endpoints (non-destructive) |
| `vastai_execute` | Reboot, run commands, copy data |
| `vastai_delete` | Destroy instances, delete resources (destructive) |

Call any group with `operation="help"` to list available operations.

## Development

Every gate runs through `dev.py`; CI calls the same commands.

| Command | Runs |
|---------|------|
| `uv run python dev.py lint` | ruff, mypy, tackbox |
| `uv run python dev.py test` | all tests, live ones included |
| `uv run python dev.py e2e` | sweep, then live tests only (`integration` marker) |
| `uv run python dev.py check` | lint + test |
| `uv run python dev.py precommit` | lint + tests without `integration` |
| `uv run python dev.py sweep` | destroy instances labeled `mcp-e2e-*` |

Live tests rent real GPUs, so the hook runs `precommit` rather than `check`.
Enable the hook once per clone:

```bash
git config core.hooksPath .githooks
```
