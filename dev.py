"""Single entry point for repo gates: linters, tests, live-instance cleanup."""

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Every gate is a subprocess of the project runner rather than of this interpreter: the
# entry point that installs the environment must not need it to already be there.
UV_PYTHON = ["uv", "run", "python"]

_PYTEST_NO_TESTS = 5


def _run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def _pytest(*args: str) -> int:
    rc = _run([*UV_PYTHON, "-m", "pytest", *args])
    if rc == _PYTEST_NO_TESTS:
        print(f"dev.py: no tests collected for {list(args)}", file=sys.stderr)
        return 1
    return rc


def lint() -> int:
    """Every linter over the whole repo. tests/ is not a package, so mypy takes paths."""
    rcs = [
        _run([*UV_PYTHON, "-m", "ruff", "check", "."]),
        _run([*UV_PYTHON, "-m", "mypy", "src", "scripts", "dev.py"]),
        _run(["uvx", "tackbox@latest", "lint", "."]),
    ]
    return next((rc for rc in rcs if rc != 0), 0)


def test() -> int:
    return _pytest()


def e2e() -> int:
    """Sweeps first: a rental left by a killed run keeps burning money through this one."""
    return sweep() or _pytest("-m", "integration")


def install_hook() -> int:
    """Point git at the repo's tracked pre-commit hook. Idempotent."""
    if (ROOT / ".githooks" / "pre-commit").exists():
        return subprocess.run(
            ["git", "config", "core.hooksPath", ".githooks"], cwd=ROOT, check=False
        ).returncode
    print("no tracked hook: expected .githooks/pre-commit", file=sys.stderr)
    return 1


def _hook_ready() -> bool:
    if (ROOT / ".git" / "hooks" / "pre-commit").exists():
        return True
    configured = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return bool(configured) and (ROOT / configured / "pre-commit").exists()


def _hook_hint() -> None:
    # Running the gate by hand is what an uninstalled hook looks like; CI does
    # not care, and precommit itself only ever runs once the hook exists.
    if not os.environ.get("CI") and not _hook_ready():
        print("hint: `python dev.py hook` installs the pre-commit gate", file=sys.stderr)


def check() -> int:
    _hook_hint()
    return lint() or test()


def precommit() -> int:
    return lint() or _pytest("-m", "not integration")


def sweep() -> int:
    """Destroy every instance left behind by e2e runs. Runs out of process because it is
    the only gate that needs the package importable."""
    return _run([*UV_PYTHON, "scripts/sweep.py"])


COMMANDS: dict[str, Callable[[], int]] = {
    "lint": lint,
    "test": test,
    "e2e": e2e,
    "check": check,
    "precommit": precommit,
    "sweep": sweep,
    "hook": install_hook,
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(f"usage: dev.py {{{'|'.join(COMMANDS)}}}", file=sys.stderr)
        return 2
    return COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    sys.exit(main())
