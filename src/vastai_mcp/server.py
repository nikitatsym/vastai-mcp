import inspect
import types
import typing
from collections.abc import Callable
from typing import Any

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ConfigDict, ValidationError, create_model
from pydantic.fields import FieldInfo

from . import tools as _tools_module
from .registry import ROOT

mcp = MCPServer("vastai")

# -- State (populated by _register_tools) --------------------

_group_ops: dict[str, dict[str, Any]] = {}  # {group_name: {PascalName: fn}}
_all_grouped: dict[str, str] = {}  # {PascalName: group_name}

_SCHEMA_HINT = "Call operation='schema', params={'op': 'OpName'} for the full JSON Schema."
_MAX_ECHOED_INPUT = 80


def _to_pascal(name: str) -> str:
    return "".join(w.capitalize() for w in name.split("_"))


def _is_union(origin: Any) -> bool:
    # Before 3.14 the origin of `str | None` is types.UnionType, not typing.Union.
    return origin is typing.Union or origin is types.UnionType


# -- Params model --------------------

def _build_params_model(fn: Any) -> type[BaseModel]:
    """Pydantic model mirroring the signature: types, defaults, Annotated descriptions.

    One declaration feeds three outputs: validation, help bullets, JSON Schema.
    """
    hints = typing.get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for name, param in inspect.signature(fn).parameters.items():
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (hints.get(name, Any), default)
    return create_model(
        f"{_to_pascal(fn.__name__)}Params",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _format_validation_error(error: ValidationError, op_name: str) -> str:
    lines = [f"Invalid params for {op_name}:"]
    for item in error.errors():
        loc = ".".join(str(x) for x in item["loc"]) or "<root>"
        got = repr(item.get("input"))
        if len(got) > _MAX_ECHOED_INPUT:
            got = got[: _MAX_ECHOED_INPUT - 3] + "..."
        lines.append(f"  - {loc}: {item['msg']} (got {got})")
    lines.append(f"Call operation='schema', params={{'op': {op_name!r}}} for the parameter spec.")
    return "\n".join(lines)


def _validated_call(fn: Any, params: dict[str, Any], op_name: str) -> Any:
    """Unknown keys, wrong types and missing required params all crash here, by field name."""
    model: type[BaseModel] = fn._params_model
    try:
        validated = model.model_validate(params)
    except ValidationError as e:
        raise ValueError(_format_validation_error(e, op_name)) from e
    # exclude_unset keeps the function's own defaults in charge of omitted params.
    return fn(**validated.model_dump(exclude_unset=True))


# -- Help --------------------

def _format_type(hint: Any) -> str:
    origin = typing.get_origin(hint)
    if origin is typing.Literal:
        vals = typing.get_args(hint)
        return "|".join(str(v) for v in vals)
    if _is_union(origin):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return _format_type(args[0])
        return "|".join(_format_type(a) for a in args)
    if hint is str:
        return "str"
    if hint is int:
        return "int"
    if hint is float:
        return "float"
    if hint is bool:
        return "bool"
    return str(hint)


def _format_param(name: str, field: FieldInfo) -> str:
    """`name: type` when required, `name?: type[=default]` when the caller may omit it."""
    type_str = _format_type(field.annotation)
    if field.is_required():
        return f"{name}: {type_str}"
    if field.default is None:
        return f"{name}?: {type_str}"
    return f"{name}?: {type_str}={field.default!r}"


def _build_help(group_name: str) -> str:
    """First docstring line is the op summary; the rest is indented under it.

    Per-param constraints come from Field(description=...) as bullets below the
    signature, so callers learn them from help, not from errors.
    """
    lines = []
    for pascal_name, fn in _group_ops[group_name].items():
        model: type[BaseModel] = fn._params_model
        parts = [_format_param(n, f) for n, f in model.model_fields.items()]
        doc = inspect.getdoc(fn) or ""
        head, _, body = doc.partition("\n\n")
        head = " ".join(head.split())
        lines.append(f"  {pascal_name}({', '.join(parts)}) - {head}")
        for body_line in body.rstrip().splitlines():
            lines.append(f"    {body_line}" if body_line else "")
        for name, field in model.model_fields.items():
            if field.description:
                lines.append(f"    {name}: {field.description}")
    count = len(_group_ops[group_name])
    return f"{count} operations available. {_SCHEMA_HINT}\n" + "\n".join(lines)


# -- Schema --------------------

def _build_schema(group_name: str, op_name: str | None) -> dict[str, Any]:
    ops = _group_ops[group_name]
    if op_name is None:
        return {
            "operations": sorted(ops),
            "hint": "Pass params={'op': '<OpName>'} for the JSON Schema of one operation.",
        }
    if op_name not in ops:
        raise ValueError(
            f"Unknown operation {op_name!r} in {group_name}. Available: {sorted(ops)}"
        )
    fn = ops[op_name]
    model: type[BaseModel] = fn._params_model
    schema: dict[str, Any] = model.model_json_schema()
    doc = inspect.getdoc(fn)
    if doc:
        schema["description"] = doc
    return schema


# -- Dispatch --------------------

def _dispatch(operation: str, group_name: str, params: dict[str, Any]) -> Any:
    if operation == "schema":
        return _build_schema(group_name, params.get("op"))
    ops = _group_ops[group_name]
    if operation not in ops:
        if operation in _all_grouped:
            correct = _all_grouped[operation]
            raise ValueError(
                f"{operation!r} belongs to {correct!r}, not {group_name!r}. "
                f"Call {correct}(operation={operation!r}, ...) instead."
            )
        raise ValueError(
            f"Unknown operation {operation!r} in {group_name}. "
            "Use operation='help' to list operations or operation='schema' for details."
        )
    return _validated_call(ops[operation], params, operation)


def _register_tools() -> None:
    groups: dict[str, tuple[Any, dict[str, Any]]] = {}
    # Any, not FunctionType: registration hangs attributes off the function object.
    fn: Any
    for name, fn in inspect.getmembers(_tools_module, inspect.isfunction):
        if not hasattr(fn, "_mcp_group"):
            continue
        fn._params_model = _build_params_model(fn)
        group = fn._mcp_group
        if group is ROOT:
            mcp.tool()(fn)
        else:
            if group.name not in groups:
                groups[group.name] = (group, {})
            groups[group.name][1][name] = fn

    for group_name, (group, fns) in groups.items():
        ops = {_to_pascal(n): fn for n, fn in fns.items()}
        _group_ops[group_name] = ops
        for pascal_name in ops:
            _all_grouped[pascal_name] = group_name

        def _make_tool(gname: str, gdoc: str) -> Callable[[str, dict[str, Any] | None], Any]:
            def tool_fn(operation: str, params: dict[str, Any] | None = None) -> Any:
                if operation == "help":
                    return _build_help(gname)
                return _dispatch(operation, gname, params or {})

            tool_fn.__name__ = gname
            tool_fn.__qualname__ = gname
            tool_fn.__doc__ = gdoc
            return tool_fn

        mcp.tool()(_make_tool(group_name, group.doc))


_register_tools()
