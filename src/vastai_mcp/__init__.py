from .server import mcp


def main() -> None:
    mcp.run(transport="stdio")
