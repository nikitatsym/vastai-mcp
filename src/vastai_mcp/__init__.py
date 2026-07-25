from .server import mcp


def main():
    mcp.run(transport="stdio")
