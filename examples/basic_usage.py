"""Minimal example: measure instruction count for a Python solution.

Prereq: SandboxFusion-PerfCount server is running on localhost:8080.
See top-level README for launch instructions.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python examples/basic_usage.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client.sandbox_instr_count import SandboxClient


def main() -> None:
    cli = SandboxClient("http://localhost:8080")

    if not cli.ping():
        print("ERROR: sandbox unreachable at http://localhost:8080")
        print("Start it with: cd sandbox-server && make run-online WORKERS=64")
        sys.exit(1)

    # Two solutions for the same problem: sum of 1..n.
    # `slow`: O(n) Python loop. `fast`: O(1) closed form.
    slow = "n = int(input())\ns = 0\nfor i in range(n+1):\n    s += i\nprint(s)\n"
    fast = "n = int(input())\nprint(n*(n+1)//2)\n"

    tests = [
        {"stdin": "1000\n",    "expected": "500500\n",       "tag": "small"},
        {"stdin": "1000000\n", "expected": "500000500000\n", "tag": "large"},
    ]

    for label, src in [("slow O(n)", slow), ("fast O(1)", fast)]:
        result = cli.run(src, tests)
        print(f"\n=== {label} ===")
        print(f"  all_passed: {result.all_passed}")
        for c in result.per_case:
            instr = f"{c.instructions:>12,}" if c.instructions else "    (no perf)"
            print(f"  [{c.tag:>6}] passed={c.passed}  instructions={instr}  "
                  f"elapsed_ns={c.elapsed_ns:>12,}")


if __name__ == "__main__":
    main()
