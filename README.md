# vastai-mcp

MCP server for Vast.ai GPU marketplace.

## Install

```
uvx --extra-index-url https://nikitatsym.github.io/vastai-mcp/simple vastai-mcp
```

## Configure

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
