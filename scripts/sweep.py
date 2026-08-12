"""Destroy the instances live e2e runs leave behind.

A module of its own rather than a dev.py function: dev.py has to run on a bare
interpreter, and this is the one gate that needs the package installed.
"""

import sys

from vastai_mcp.tools import destroy_instance, list_instances

# Live e2e tests label every instance they rent with this prefix.
LABEL_PREFIX = "mcp-e2e-"


def sweep() -> int:
    instances = list_instances()["instances"]
    doomed = [
        inst["id"]
        for inst in instances
        if str(inst.get("label") or "").startswith(LABEL_PREFIX)
    ]
    for instance_id in doomed:
        destroy_instance(instance_id)
        print(f"destroyed {instance_id}")
    print(f"sweep: {len(doomed)} instance(s) destroyed")
    return 0


if __name__ == "__main__":
    sys.exit(sweep())
