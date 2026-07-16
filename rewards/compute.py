# coding: utf-8
"""Unified reward facade over the SandboxFusion-RL server.

suite_rewards()    - test-generation RL: correctness / coverage / mutation
solution_rewards() - code-generation RL: correctness / instruction-count efficiency

Both return plain dicts ready to feed a GRPO/PPO reward function.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))          # for client.sandbox_instr_count
sys.path.insert(0, str(_ROOT / "rewards"))

import sandbox_eval_lib as _lib  # noqa: E402  (correctness/coverage/mutation)

ENDPOINT = os.environ.get("SANDBOX_FUSION_ENDPOINT", "http://localhost:8080").rstrip("/")


async def suite_rewards(reference_code: str, test_snippets: list[str],
                        subset: str, task_id, mut_sample: int | None = None) -> dict:
    """Test-generation rewards for one rollout (a whole pytest suite).

    Requires a prebuilt cosmic-ray mutant pool under TESTGEN_POOL_ROOT
    (default /data/UnLeakedTestBench/pool) keyed by (subset, task_id).

    Returns keys: correctness, statement_cov, branch_cov, mutation_score,
    n_tests, n_correct, per_test_mut_killed, per_mutant_kills, ... (see
    sandbox_eval_lib._HARNESS for the full schema). Per-test first-kill
    counts are what token-level segment credit consumes.
    """
    return await _lib.evaluate_suite(reference_code, test_snippets,
                                     subset, task_id, mut_sample=mut_sample)


def solution_rewards(solution_code: str, io_cases: list[dict],
                     reference_instructions: int | None = None,
                     endpoint: str = ENDPOINT) -> dict:
    """Code-generation rewards for one candidate solution.

    io_cases: [{"stdin": str, "expected": str, "tag": str?}, ...]
    reference_instructions: retired-instruction count of the reference
        solution on the same cases (build once with tools/build_reference_table.py).

    Returns:
      correctness      fraction of cases passed
      instructions     summed retired instructions over passed cases (None if
                       the kernel refuses perf_event_open; reward code should
                       treat None as "efficiency signal unavailable")
      efficiency       log2(reference_instructions / instructions) when both
                       sides are available, else None. Positive = faster than
                       the reference; 0 = parity.
      per_case         list of {tag, passed, instructions, elapsed_ns}
    """
    from client.sandbox_instr_count import SandboxClient

    cli = SandboxClient(endpoint)
    res = cli.run(solution_code, io_cases)
    per_case = [{"tag": c.tag, "passed": c.passed,
                 "instructions": c.instructions, "elapsed_ns": c.elapsed_ns}
                for c in res.per_case]
    n = len(per_case)
    correctness = (sum(1 for c in per_case if c["passed"]) / n) if n else 0.0

    counts = [c["instructions"] for c in per_case if c["passed"] and c["instructions"]]
    instructions = sum(counts) if counts and res.all_passed else None

    efficiency = None
    if instructions and reference_instructions:
        efficiency = math.log2(reference_instructions / instructions)

    return {"correctness": correctness, "instructions": instructions,
            "efficiency": efficiency, "per_case": per_case}
