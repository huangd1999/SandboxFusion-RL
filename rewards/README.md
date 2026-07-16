# Unified reward layer

One import surface for every reward the sandbox can produce, aimed at
agentic-RL training loops (GRPO/PPO):

| Reward | Scenario | Entry point | Backend |
|---|---|---|---|
| correctness | both | `suite_rewards` / `solution_rewards` | pytest / stdin-stdout diff |
| statement + branch coverage | test-gen | `suite_rewards` | `pytest --cov` (per-test contexts) |
| mutation score (per-test first-kill) | test-gen | `suite_rewards` | prebuilt cosmic-ray pools, trinary kill matrix |
| efficiency (retired instructions) | code-gen | `solution_rewards` | `perf_event_open` inside bwrap (see `client/`) |

```python
from rewards import suite_rewards, solution_rewards

# test-generation RL: the model writes a pytest suite
res = await suite_rewards(reference_code, test_snippets, "ULT", task_id)
# -> correctness, statement_cov, branch_cov, mutation_score,
#    per_test_mut_killed (feeds token-level segment credit), ...

# code-generation RL: the model writes a solution
res = solution_rewards(candidate_code, io_cases, reference_instructions=ref_n)
# -> correctness, instructions, efficiency = log2(ref/model), per_case
```

Environment knobs (all optional):

- `SANDBOX_FUSION_ENDPOINT` (default `http://localhost:8080`)
- `TESTGEN_POOL_ROOT` (default `/data/UnLeakedTestBench/pool`) — mutant pools
- `TESTGEN_SANDBOX_CONCURRENCY`, `TESTGEN_MUT_SAMPLE`, `TESTGEN_EXEC_BUDGET`

Smoke test (server must be running): `python examples/all_rewards_smoke.py`

Notes:
- `efficiency` needs `kernel.perf_event_paranoid <= 1` (or the bwrap sudo
  path from the top-level README); the client degrades to `instructions=None`
  instead of failing, so reward code must handle the None case.
- Mutation pools are built offline once per task (see the Parsimony repo's
  pool builder); at train time each rollout reuses the pool, keeping reward
  latency ~1 s instead of ~110 s.
