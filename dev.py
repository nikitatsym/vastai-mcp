"""Single entry point for repo gates: linters, tests, live-instance cleanup."""

import subprocess
import sys
from collections.abc import Callable

from vastai_mcp.tools import destroy_instance, list_instances

# Live e2e tests label every instance they rent with this prefix.
SWEEP_LABEL_PREFIX = "mcp-e2e-"

_PYTEST_NO_TESTS = 5


def _run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=False).returncode


def _pytest(*args: str) -> int:
    rc = _run([sys.executable, "-m", "pytest", *args])
    if rc == _PYTEST_NO_TESTS:
        print(f"dev.py: no tests collected for {list(args)}", file=sys.stderr)
        return 1
    return rc


def lint() -> int:
    """Every linter over the whole repo. tests/ is not a package, so mypy takes paths."""
    rcs = [
        _run([sys.executable, "-m", "ruff", "check", "."]),
        _run([sys.executable, "-m", "mypy", "src", "dev.py"]),
        _run(["uvx", "tackbox@latest", "lint", "."]),
    ]
    return next((rc for rc in rcs if rc != 0), 0)


def test() -> int:
    return _pytest()


def e2e() -> int:
    return _pytest("-m", "integration")


def check() -> int:
    return lint() or test()


def precommit() -> int:
    return lint() or _pytest("-m", "not integration")


def sweep() -> int:
    """Destroy every instance left behind by e2e runs."""
    instances = list_instances()["instances"]
    doomed = [
        i["id"] for i in instances if str(i.get("label") or "").startswith(SWEEP_LABEL_PREFIX)
    ]
    for instance_id in doomed:
        destroy_instance(instance_id)
        print(f"destroyed {instance_id}")
    print(f"sweep: {len(doomed)} instance(s) destroyed")
    return 0


COMMANDS: dict[str, Callable[[], int]] = {
    "lint": lint,
    "test": test,
    "e2e": e2e,
    "check": check,
    "precommit": precommit,
    "sweep": sweep,
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(f"usage: dev.py {{{'|'.join(COMMANDS)}}}", file=sys.stderr)
        return 2
    return COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    sys.exit(main())
