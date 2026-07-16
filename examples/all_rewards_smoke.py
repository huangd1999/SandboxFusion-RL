"""Smoke test: all four rewards through the unified rewards/ facade.

Prereqs: SandboxFusion-RL server on localhost:8080; a mutant pool under
TESTGEN_POOL_ROOT for the suite_rewards half.

Run from the repo root:  python examples/all_rewards_smoke.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rewards import solution_rewards, suite_rewards


def demo_solution_rewards() -> None:
    print("== code-generation RL: correctness + efficiency ==")
    slow = "n = int(input())\ns = 0\nfor i in range(n+1):\n    s += i\nprint(s)\n"
    fast = "n = int(input())\nprint(n*(n+1)//2)\n"
    cases = [{"stdin": "1000000\n", "expected": "500000500000\n", "tag": "large"}]

    ref = solution_rewards(slow, cases)
    print(f"  reference (O(n) loop): correctness={ref['correctness']:.2f} "
          f"instructions={ref['instructions']}")
    cand = solution_rewards(fast, cases, reference_instructions=ref["instructions"])
    print(f"  candidate (O(1) form): correctness={cand['correctness']:.2f} "
          f"instructions={cand['instructions']} efficiency={cand['efficiency']}")


async def demo_suite_rewards() -> None:
    print("== test-generation RL: correctness + coverage + mutation ==")
    # Load one real ULT task so the prebuilt mutant pool exists.
    ult = Path("/home/nvidia/UnLeakedTestBench/datasets/ULTv3.jsonl")
    text = ult.read_text()
    records = (json.loads(text) if text.lstrip().startswith("[")
               else [json.loads(l) for l in text.splitlines() if l.strip()])
    rec = next((r for r in records
                if Path(f"/data/UnLeakedTestBench/pool/ULT/task_{r['task_id']}/cosmic-ray.sqlite").exists()),
               None)
    if rec is None:
        print("  SKIP: no task with a prebuilt ULT pool found")
        return

    fn = rec["func_name"]
    tests = [
        f"def test_smoke_type():\n    r = {fn}(*__SMOKE_ARGS__)\n",
    ]
    # A trivially-true test exercises the plumbing without knowing the spec;
    # real callers pass model-generated suites here instead.
    tests = [f"def test_runs():\n    assert callable({fn})\n"]
    res = await suite_rewards(rec["code"], tests, "ULT", rec["task_id"])
    keys = ["correctness", "statement_cov", "branch_cov", "mutation_score",
            "n_tests", "n_correct", "reason"]
    print(f"  task_id={rec['task_id']} func={fn}")
    print("  " + json.dumps({k: res.get(k) for k in keys if k in res or k == 'reason'},
                            default=str))
    per_test = res.get("per_test_mut_killed")
    print(f"  per_test_mut_killed present: {per_test is not None}")


if __name__ == "__main__":
    demo_solution_rewards()
    asyncio.run(demo_suite_rewards())
