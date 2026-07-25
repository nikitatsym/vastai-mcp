from collections.abc import Callable
from typing import Any


class Group:
    """A named group of MCP tool operations exposed as a single meta-tool."""

    __slots__ = ("doc", "name")

    def __init__(self, name: str, doc: str):
        self.name = name
        self.doc = doc


ROOT = Group("root", "")


def _op(group: Group) -> Callable[[Any], Any]:
    """Mark a function as an MCP tool in the given group."""

    def decorator(fn: Any) -> Any:
        if not fn.__doc__:
            raise RuntimeError(f"Tool function {fn.__name__!r} has no docstring")
        fn._mcp_group = group
        return fn

    return decorator
