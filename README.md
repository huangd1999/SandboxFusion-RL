# SandboxFusion-RL

**A code-execution sandbox built for agentic RL on code: one server, four
reward signals (correctness, coverage, mutation score, efficiency), each with
the fine-grained attribution that token-level credit assignment needs.**

Fork of [bytedance/SandboxFusion](https://github.com/bytedance/SandboxFusion)
(Apache-2.0). Existing code sandboxes answer one question: *did the program
pass?* That is not enough to train a policy. RL training loops need reward
signals that are (i) **dense** (per-test, per-case, not one scalar per
rollout), (ii) **fast** (a reward call sits inside the training step; 100 s
per rollout is unusable), and (iii) **robust** (one LLM-generated memory bomb
must not take down the host). SandboxFusion-RL adds all three on top of a
production sandbox.

## Reward menu

| Reward | RL scenario | Granularity | Backend |
|---|---|---|---|
| Correctness | code-gen + test-gen | per test / per case | pytest; stdin-stdout diff |
| Statement + branch coverage | test-gen | per test (`--cov-context=test`) | `pytest-cov` |
| Mutation score | test-gen | per test, first-kill attribution | prebuilt `cosmic-ray` pools |
| Efficiency (retired instructions) | code-gen | per case, deterministic | `perf_event_open` PMU counter |

All four are reachable from one import:

```python
from rewards import suite_rewards, solution_rewards
```

## Architecture

```
 GRPO / PPO trainer (verl, tinker, TRL, ...)
        │  K rollouts per task
        ▼
 rewards/  (client facade)
   ├─ suite_rewards()     test-generation RL
   └─ solution_rewards()  code-generation RL
        │  HTTP /run_code
        ▼
 SandboxFusion-RL server (uvicorn, N workers)
   ├─ bwrap isolation (mount/pid/net namespaces)
   ├─ 8 GB RLIMIT_AS per execution
   ├─ compile cache (language, source-hash)
   └─ runners: pytest+cov / cosmic-ray exec / perf-counter driver
```

The server is stateless; reward-specific assets live outside it:

- **mutation**: a per-task mutant pool, built once offline, shipped to the
  sandbox inline with each request (no server-side state, ~1 s per rollout
  instead of ~110 s for `cosmic-ray init` from scratch);
- **efficiency**: a per-task reference instruction count, measured once
  offline into a JSON table.

## Install

```bash
# 1. Server dependencies (Python 3.10+)
pip install poetry && poetry install

# 2. Sandbox runtime for user code (pytest, coverage, cosmic-ray, numpy, ...)
#    Any env visible to the server works; see runtime/ for the reference setup.

# 3. bwrap isolation (recommended; used by the efficiency runner)
sudo apt install bubblewrap
echo "$(whoami) ALL=(ALL) NOPASSWD: /usr/bin/bwrap" | sudo tee /etc/sudoers.d/sandbox-bwrap

# 4. Hardware instruction counting (efficiency reward only)
sudo sysctl kernel.perf_event_paranoid=1
```

## Launch

```bash
make run-online WORKERS=128   # uvicorn on 0.0.0.0:8080
```

Smoke-test every reward end to end (needs a mutant pool for the test-gen
half; see below):

```bash
python examples/all_rewards_smoke.py
```

## Computing each reward in an RL loop

### 1. Code-generation RL: correctness + efficiency

The policy writes a *solution*; the reward is "does it pass, and how few
instructions does it retire". Retired-instruction count comes from the CPU's
performance-monitoring unit via `perf_event_open`: for the same code on the
same input it is deterministic across runs (drift < 0.1%), unlike wall-clock
time, which drifts ±20% under the 64-way reward-worker concurrency typical of
RL training. A noisy denominator makes a noisy reward gradient; instruction
count removes that noise at the source.

Step 1, once per dataset: measure the reference solutions.

```bash
python tools/build_reference_table.py --dataset my_tasks.jsonl --out refs.json
```

Step 2, inside the training loop:

```python
from rewards import solution_rewards

def reward_fn(task, rollout_code):
    r = solution_rewards(
        rollout_code,
        io_cases=task["cases"],                        # [{"stdin","expected"},...]
        reference_instructions=refs[task["id"]],
    )
    if r["correctness"] < 1.0:
        return 0.0                                     # wrong code earns nothing
    if r["efficiency"] is None:                        # PMU unavailable on host
        return W_CORRECT
    return W_CORRECT + W_EFF * max(r["efficiency"], 0) # log2(ref/model) instructions
```

`efficiency = log2(reference_instructions / model_instructions)`: positive
means faster than the reference, 0 means parity. The client degrades to
`instructions=None` (never crashes the rollout) if the kernel refuses the
syscall.

### 2. Test-generation RL: correctness + coverage + mutation

The policy writes a *pytest suite* for a target function; the reward is "do
the tests pass on the reference, what do they cover, and how many seeded
faults (mutants) do they kill".

Step 1, once per dataset: enumerate the mutant pool for every task.
`cosmic-ray init` needs only the reference implementation, so the pool is
reusable for **any** candidate test suite, which is exactly what makes
mutation viable as a training-time reward:

```bash
python tools/pool_builder/build_pool.py --subset ULT --workers 32
# -> <POOL_ROOT>/ULT/task_<id>/{mod.py, cosmic-ray.toml, cosmic-ray.sqlite, mutants.json}
```

Step 2, inside the training loop (async, one call per rollout):

```python
from rewards import suite_rewards

async def reward_fn(task, rollout_tests):
    r = await suite_rewards(task["reference_code"], rollout_tests,
                            subset="ULT", task_id=task["id"])
    if not r["tests_ok"]:
        return 0.0
    return W_CORRECT * r["correctness"] + W_MUT * r["mutation_score"]
```

One sandbox call returns the whole bundle:

```
correctness        fraction of tests passing on the reference
statement_cov      union statement coverage of the correct tests
branch_cov         union branch coverage
mutation_score     fraction of sampled mutants killed
per_test_cov       {test_id: [covered lines]}          per-test attribution
per_test_branch    {test_id: [(from,to) branch arcs]}
per_test_mut_killed{test_id: first-kill count}         per-test attribution
per_mutant_kills   trinary (mutant x test) kill matrix
```

### 3. Token-level credit assignment (what the per-test fields are for)

Single-shot RL (one prompt, one whole suite, one scalar reward) cannot tell
the policy *which* test did the work. The per-test fields close that gap: map
each `def test_*` block to its token span (tokenizer offset mapping), convert
`per_test_mut_killed` into zero-mean per-token advantage offsets, and add
them to the scalar GRPO advantage. High-yield tests get positive credit,
redundant tests negative, inside the same trajectory. The Parsimony recipe
(segment credit + Pareto-gated conciseness bonus) is built entirely on these
two fields; `per_mutant_kills` likewise supports union-of-first-N evaluation
(mutation score of the first N tests only), the metric that exposes suite
bloat.

### 4. Sizing the loop

- Match client concurrency to server workers:
  `TESTGEN_SANDBOX_CONCURRENCY` (default 128) should not exceed uvicorn
  `WORKERS`, or requests queue past capacity, time out, and come back as
  spurious zero rewards. The failure mode is silent: rewards look valid but
  are deflated. When sharing the sandbox with another tenant (e.g. scoring
  during training), drop the client side to 8.
- Mutation cost scales with `TESTGEN_MUT_SAMPLE` (mutants per rollout,
  default 24). 128 is our offline-evaluation protocol; 24 is comfortable for
  training.
- Every execution is capped at 8 GB RLIMIT_AS, so one degenerate rollout
  cannot OOM the host or its co-tenants.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `SANDBOX_FUSION_ENDPOINT` | `http://localhost:8080` | server URL (client side) |
| `TESTGEN_POOL_ROOT` | `/data/UnLeakedTestBench/pool` | mutant pool root (builder + reward layer) |
| `TESTGEN_PYTHON_BIN` | `.../envs/testrl/bin` | env with cosmic-ray + pytest-cov (pool builder) |
| `TESTGEN_DATASETS_DIR` | `tools/pool_builder/datasets` | `<subset>.jsonl` task files (pool builder) |
| `TESTGEN_SANDBOX_CONCURRENCY` | `128` | max in-flight reward calls |
| `TESTGEN_MUT_SAMPLE` | `24` | mutants sampled per reward call |
| `TESTGEN_EXEC_BUDGET` | `60` | sandbox seconds per mutation run |
| `SANDBOX_LOG_LEVEL` | `DEBUG` | set `INFO`/`WARNING` in production |

## Upstream improvements retained from the parent fork

- `extra_args` support for passing command-line arguments to executed programs
- real `max_concurrency` control in `local.yaml`
- fix for stdin/stdout buffer overflow on large payloads

## Repository layout

```
sandbox/        server (upstream + bwrap isolation, RLIMIT_AS cap, compile cache)
rewards/        unified reward facade + pytest/cosmic-ray harness
client/         perf_event_open instruction-count client
tools/
  build_reference_table.py   reference instruction counts (efficiency)
  pool_builder/              per-task cosmic-ray mutant pools (mutation)
examples/       all_rewards_smoke.py, basic_usage.py
runtime/        reference sandbox runtime environments
```

## Attribution

Upstream: [bytedance/SandboxFusion](https://github.com/bytedance/SandboxFusion)
(Apache-2.0, LICENSE retained). This fork adds the RL reward layer
(`rewards/`, `client/`, `tools/`), bwrap isolation, per-execution memory
caps, and the compile cache. Instruction counting builds on Linux
`perf_event_open`; mutation testing on
[cosmic-ray](https://github.com/sixty-north/cosmic-ray).
