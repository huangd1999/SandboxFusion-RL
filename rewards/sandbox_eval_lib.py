# coding: utf-8
# Shared SandboxFusion evaluator for UnLeakedTestBench test-case rewards.
#
# evaluate_suite(reference_code, test_snippets, subset, task_id) submits ONE
# self-contained script to the sandbox that:
#   - assembles mod.py (reference) + test.py (the given test snippets),
#   - runs each test on the reference to find the CORRECT ones (per-test),
#   - deselects the incorrect tests, then measures statement/branch coverage,
#   - runs `cosmic-ray exec` against the prebuilt (subsampled) mutant pool
#     shipped inline as sqlite (NO re-init) and reads cr-report.
# Returns a dict: correctness, statement_cov, branch_cov, mutation_score, ...
#
# Used by both the single-shot reward (reward.py) and the incremental
# multi-turn tool (add_test_tool.py), so the scoring is identical everywhere.

import asyncio
import base64
import json
import os
import random
import re
import sqlite3

import aiohttp

SANDBOX_ENDPOINT = os.environ.get("SANDBOX_FUSION_ENDPOINT", "http://localhost:8080").rstrip("/")
POOL_ROOT = os.environ.get("TESTGEN_POOL_ROOT", "/data/UnLeakedTestBench/pool")
# Cap concurrent in-flight sandbox submissions process-wide. Match the
# SandboxFusion server's uvicorn --workers (128) so we never queue past its
# capacity (over-firing -> requests time out -> spurious reward 0). Used by the
# tinker recipe and verl's multi-turn add_test_tool (both call evaluate_suite);
# verl's single-shot reward.py has its own separate TESTGEN_CONCURRENCY cap.
SANDBOX_CONCURRENCY = int(os.environ.get("TESTGEN_SANDBOX_CONCURRENCY", "128"))
# Lazily create one semaphore per running event loop (a semaphore is bound to
# the loop it was created in; reward.py's asyncio.run() path uses a fresh loop
# per batch, so keying by loop avoids "bound to a different loop" errors).
_SEMS: dict = {}


def _get_sem() -> "asyncio.Semaphore":
    loop = asyncio.get_running_loop()
    sem = _SEMS.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(SANDBOX_CONCURRENCY)
        _SEMS[loop] = sem
    return sem
MUT_SAMPLE = int(os.environ.get("TESTGEN_MUT_SAMPLE", "24"))
MUT_PER = int(os.environ.get("TESTGEN_MUT_PER", "8"))
EXEC_BUDGET = int(os.environ.get("TESTGEN_EXEC_BUDGET", "60"))
COV_TIMEOUT = int(os.environ.get("TESTGEN_COV_TIMEOUT", "30"))

_PREAMBLE = (
    "import os, re, math, numpy, pandas, pytest, random, string, warnings, datetime\n"
    "import itertools, functools, collections, traceback\n"
    "import numpy as np\nimport pandas as pd\n"
    "from typing import (List, Dict, Any, Optional, Union, Tuple, Set, FrozenSet,\n"
    "    Sequence, Iterable, Generator, Callable)\n"
)

_EQ_HELPER = '''
def _eq(a, b, _rel=1e-6, _abs=1e-9):
    import math
    try:
        import numpy as _np
    except Exception:
        _np = None
    import collections.abc as _abc
    if _np is not None and (isinstance(a, _np.ndarray) or isinstance(b, _np.ndarray)):
        try:
            return _np.allclose(_np.asarray(a, dtype=float), _np.asarray(b, dtype=float),
                                rtol=_rel, atol=_abs, equal_nan=True)
        except Exception:
            return _np.array_equal(_np.asarray(a), _np.asarray(b))
    if isinstance(a, float) or isinstance(b, float):
        try:
            if math.isnan(a) and math.isnan(b):
                return True
        except Exception:
            pass
        return math.isclose(a, b, rel_tol=_rel, abs_tol=_abs)
    if isinstance(a, (str, bytes)):
        return a == b
    if isinstance(a, _abc.Mapping) and isinstance(b, _abc.Mapping):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, _abc.Sequence) and isinstance(b, _abc.Sequence):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, (set, frozenset)) and isinstance(b, (set, frozenset)):
        return a == b
    return a == b
'''.lstrip()

# %%TOKENS%% are replaced with JSON strings (safe, no brace-escaping needed).
_HARNESS = r'''
import os, sys, re, json, base64, sqlite3, subprocess, signal, tempfile, shutil, shlex
WD = tempfile.mkdtemp(prefix="rw_"); os.chdir(WD)
open("_bench_preamble.py", "w").write(%%PRE%%)
open("mod.py", "w").write(%%MOD%%)
open("test.py", "w").write(%%TEST%%)
open("cr.sqlite", "wb").write(base64.b64decode(%%SQL%%))

KEEP = set(json.loads(%%KEEP%%))
db = sqlite3.connect("cr.sqlite")
allids = [r[0] for r in db.execute("select job_id from work_items")]
drop = [(i,) for i in allids if KEEP and i not in KEEP]
db.executemany("delete from work_items where job_id=?", drop)
db.executemany("delete from mutation_specs where job_id=?", drop)
db.commit()
n_mut = db.execute("select count(*) from work_items").fetchone()[0]
db.close()

R = {"tests_ok": False, "n_tests": 0, "n_correct": 0, "correctness": 0.0,
     "pass_ids": [], "fail_ids": [],
     "statement_cov": 0.0, "branch_cov": 0.0,
     "n_mutants": n_mut, "completed": 0, "killed": 0, "surviving": 0,
     "mutation_score": 0.0, "mut_timeout": False}

# 1. per-test correctness on the reference
pv = subprocess.run([sys.executable, "-m", "pytest", "test.py", "-v", "--tb=no",
    "-p", "no:cacheprovider", "--no-header"], capture_output=True, text=True)
ov = pv.stdout + pv.stderr
pass_ids, fail_ids = [], []
for ln in ov.splitlines():
    m = re.match(r"(test\.py::\S+)\s+(PASSED|FAILED|ERROR)", ln)
    if m:
        (pass_ids if m.group(2) == "PASSED" else fail_ids).append(m.group(1))
R["n_tests"] = len(pass_ids) + len(fail_ids)
R["n_correct"] = len(pass_ids)
R["pass_ids"], R["fail_ids"] = pass_ids, fail_ids
R["correctness"] = round(len(pass_ids) / R["n_tests"], 4) if R["n_tests"] else 0.0
R["tests_ok"] = len(pass_ids) > 0
DESELECT = []
for fid in fail_ids:
    DESELECT += ["--deselect", fid]
ds_str = " ".join(shlex.quote(x) for x in DESELECT)

if R["tests_ok"]:
    cov = "cov.json"
    open(".coveragerc", "w").write("[json]\nshow_contexts = True\n")
    subprocess.run([sys.executable, "-m", "pytest", "test.py", "-q", "--no-header",
        "-p", "no:cacheprovider", *DESELECT, "--cov=mod", "--cov-branch",
        "--cov-context=test", "--cov-config=.coveragerc",
        "--cov-report=json:" + cov, "--cov-report="], capture_output=True, text=True)
    if os.path.exists(cov):
        cov_data = json.load(open(cov))
        t = cov_data.get("totals", {})
        ns, nb = t.get("num_statements", 0), t.get("num_branches", 0)
        R["statement_cov"] = round((t.get("covered_lines", 0) / ns) if ns else 0.0, 4)
        R["branch_cov"] = round((t.get("covered_branches", 0) / nb) if nb else 1.0, 4)
        # Per-test covered line numbers (relative to mod.py) + per-test branches
        # (approx: branch (X,Y) attributed to test T iff T's context executed
        # both endpoints; a small overcount when a test reaches Y via a diff path,
        # but a valid upper bound for "first-N union" branch aggregation).
        per_test_lines = {tid: set() for tid in pass_ids}
        executed_branches_all = []
        for fname, fdata in cov_data.get("files", {}).items():
            if "mod" not in os.path.basename(fname): continue
            for line_str, ctxs in fdata.get("contexts", {}).items():
                try: lineno = int(line_str)
                except ValueError: continue
                for ctx in ctxs:
                    # ctx examples: "test.py::test_foo|run", "test.py::test_foo"
                    tid = ctx.split("|")[0]
                    if tid in per_test_lines:
                        per_test_lines[tid].add(lineno)
            executed_branches_all.extend([tuple(b) for b in fdata.get("executed_branches", [])])
        R["per_test_cov"] = {tid: sorted(lines) for tid, lines in per_test_lines.items()}
        per_test_branches = {tid: set() for tid in pass_ids}
        for fr, to in executed_branches_all:
            for tid, lines in per_test_lines.items():
                if fr in lines and to in lines:
                    per_test_branches[tid].add((fr, to))
        R["per_test_branch"] = {tid: sorted(list(b)) for tid, b in per_test_branches.items()}
        R["n_statements_total"] = ns
        R["n_branches_total"] = nb

    # Per-test mutation: wrapper records pass/fail per test per mutant invocation.
    # Values baked inline so wrapper has no env dependency.
    WRAPPER = os.path.join(WD, "_cr_wrapper.py")
    SIDE_OUT = os.path.join(WD, "per_test_mut.jsonl")
    # Short-circuit (-x): pytest stops on first failure; tests AFTER first killer
    # are not executed. Per-test status legend:
    #   1 = PASSED on mutant (didn't kill it)
    #   0 = FAILED/ERROR (killed it; usually the first such test in suite order)
    #   -1 = not run (an earlier test already killed; we infer membership in
    #        union without knowing this test's independent effect)
    open(WRAPPER, "w").write(
        "import sys, json, subprocess, re\n"
        "out_path = " + json.dumps(SIDE_OUT) + "\n"
        "tests = " + json.dumps(pass_ids) + "\n"
        "result = {tid: -1 for tid in tests}\n"
        "try:\n"
        "    pr = subprocess.run([sys.executable, '-m', 'pytest', *tests, '-v', '-x',\n"
        "        '-p', 'no:cacheprovider', '--tb=no', '--no-header'],\n"
        "        capture_output=True, text=True, timeout=20)\n"
        "    out = pr.stdout + pr.stderr\n"
        "except Exception:\n"
        "    out = ''\n"
        "any_failed = False\n"
        "for ln in out.splitlines():\n"
        "    m = re.match(r'(test\\.py::\\S+)\\s+(PASSED|FAILED|ERROR)', ln)\n"
        "    if m and m.group(1) in result:\n"
        "        ok = m.group(2) == 'PASSED'\n"
        "        result[m.group(1)] = 1 if ok else 0\n"
        "        if not ok: any_failed = True\n"
        "with open(out_path, 'a') as f:\n"
        "    f.write(json.dumps({'r': result}) + '\\n')\n"
        "sys.exit(1 if any_failed else 0)\n"
    )
    open("cosmic-ray.toml", "w").write(
        '[cosmic-ray]\nmodule-path = "mod.py"\ntimeout = %%MUTPER%%\n'
        'excluded-modules = []\n'
        'test-command = "%s %s"\n'
        '[cosmic-ray.distributor]\nname = "local"\n' % (sys.executable, WRAPPER))
    if n_mut > 0:
        CR = shutil.which("cosmic-ray") or "cosmic-ray"
        CRR = shutil.which("cr-report") or "cr-report"
        try:
            pr = subprocess.Popen([CR, "exec", "cosmic-ray.toml", "cr.sqlite"],
                start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                pr.wait(timeout=%%EXECB%%)
            except subprocess.TimeoutExpired:
                R["mut_timeout"] = True
                try: os.killpg(os.getpgid(pr.pid), signal.SIGKILL)
                except Exception: pass
                pr.wait(timeout=10)
        except Exception as e:
            R["mut_err"] = str(e)[:200]
        rep = subprocess.run([CRR, "cr.sqlite", "--show-pending"], capture_output=True, text=True)
        x = rep.stdout
        def g(rx):
            m = re.search(rx, x); return int(m.group(1)) if m else 0
        tot = g(r"total jobs:\s*(\d+)") or n_mut
        R["n_mutants"] = tot
        R["completed"] = g(r"complete:\s*(\d+)\s*\(")
        R["surviving"] = g(r"surviving mutants:\s*(\d+)\s*\(")
        R["killed"] = max(R["completed"] - R["surviving"], 0)
        R["mutation_score"] = round(R["killed"] / tot, 4) if tot else 0.0

        # Parse sidechannel: each line is one mutant's per-test outcome.
        per_test_kill = {tid: 0 for tid in pass_ids}
        per_mutant_kills = []  # list of {tid: bool, ...}
        if os.path.exists(SIDE_OUT):
            for ln in open(SIDE_OUT):
                try:
                    d = json.loads(ln)
                except Exception:
                    continue
                rmap = d.get("r", {})
                killers = [tid for tid, ok in rmap.items() if ok == 0 and tid in per_test_kill]
                for tid in killers:
                    per_test_kill[tid] += 1
                # 0=killed, 1=passed, -1=not_run (short-circuit). Preserve trinary state.
                per_mutant_kills.append({tid: rmap.get(tid, -1) for tid in pass_ids})
        R["per_test_mut_killed"] = per_test_kill
        R["per_mutant_kills"] = per_mutant_kills
        R["n_mutants_scored"] = len(per_mutant_kills)

shutil.rmtree(WD, ignore_errors=True)
print("@@E@@" + json.dumps(R))
'''

_POOL_CACHE = {}


def _load_pool(subset, task_id):
    key = (subset, str(task_id))
    if key in _POOL_CACHE:
        return _POOL_CACHE[key]
    d = os.path.join(POOL_ROOT, subset, f"task_{task_id}")
    spath, mpath = os.path.join(d, "cosmic-ray.sqlite"), os.path.join(d, "mutants.json")
    if not (os.path.exists(spath) and os.path.exists(mpath)):
        _POOL_CACHE[key] = (None, [])
        return _POOL_CACHE[key]
    with open(spath, "rb") as f:
        sb64 = base64.b64encode(f.read()).decode()
    job_ids = [m["job_id"] for m in json.load(open(mpath)).get("mutants", [])]
    _POOL_CACHE[key] = (sb64, job_ids)
    return _POOL_CACHE[key]


def extract_code_block(text):
    """Pull the last ```python ...``` block (or the raw text if none)."""
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return (m[-1] if m else text).strip()


def _rename_tests(code):
    """Make each `def test_*` unique so concatenated snippets don't shadow."""
    out, n = [], 0
    pat = re.compile(r"(\s*def\s+)(test_?)(.*)")
    for line in code.split("\n"):
        m = pat.match(line)
        if m:
            n += 1
            out.append(f"{m.group(1)}test_{n}_{m.group(3)}")
        else:
            out.append(line)
    return "\n".join(out)


def _build_mod(reference_code):
    return "from _bench_preamble import *\n\n\n" + reference_code.strip() + "\n"


def _build_test(test_snippets):
    body = "\n\n\n".join(s.strip("\n") for s in test_snippets if s.strip())
    return _rename_tests("from mod import *\n\n\n" + _EQ_HELPER + "\n\n" + body + "\n")


def _build_harness(reference_code, test_snippets, sqlite_b64, keep_ids):
    subs = {
        "%%PRE%%": json.dumps(_PREAMBLE),
        "%%MOD%%": json.dumps(_build_mod(reference_code)),
        "%%TEST%%": json.dumps(_build_test(test_snippets)),
        "%%SQL%%": json.dumps(sqlite_b64),
        "%%KEEP%%": json.dumps(json.dumps(keep_ids)),
        "%%MUTPER%%": str(MUT_PER),
        "%%EXECB%%": str(EXEC_BUDGET),
    }
    out = _HARNESS
    for k, v in subs.items():
        out = out.replace(k, v)
    return out


async def _sandbox_run(code, run_timeout, session=None):
    payload = {"code": code, "language": "python", "run_timeout": run_timeout,
               "compile_timeout": 10, "files": {}, "fetch_files": []}
    own = session is None
    if own:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=run_timeout + 30))
    try:
        async with session.post(f"{SANDBOX_ENDPOINT}/run_code", json=payload) as r:
            data = await r.json()
    finally:
        if own:
            await session.close()
    return (data.get("run_result") or {}).get("stdout", "") or ""


_EMPTY = {"tests_ok": False, "n_tests": 0, "n_correct": 0, "correctness": 0.0,
          "statement_cov": 0.0, "branch_cov": 0.0, "mutation_score": 0.0, "n_mutants": 0}


async def evaluate_suite(reference_code, test_snippets, subset, task_id,
                         mut_sample=None, session=None):
    """Evaluate a list of test snippets against the reference + mutant pool.

    Returns the raw sandbox result dict (see _HARNESS R), or an `{**_EMPTY,
    "reason": ...}` dict on failure. Coverage/mutation reflect the CORRECT subset.
    """
    if not test_snippets:
        return {**_EMPTY, "reason": "no_tests"}
    sqlite_b64, job_ids = _load_pool(subset, task_id)
    if sqlite_b64 is None:
        return {**_EMPTY, "reason": f"no_pool:{subset}/{task_id}"}

    k = MUT_SAMPLE if mut_sample is None else mut_sample
    if k and len(job_ids) > k:
        keep = random.Random(str(task_id)).sample(job_ids, k)
    else:
        keep = job_ids

    harness = _build_harness(reference_code, test_snippets, sqlite_b64, keep)
    try:
        async with _get_sem():  # cap concurrent sandbox submissions (server has 128 workers)
            stdout = await _sandbox_run(harness, EXEC_BUDGET + COV_TIMEOUT + 10, session=session)
    except Exception as e:
        return {**_EMPTY, "reason": f"sandbox_error:{type(e).__name__}"}
    line = next((l[5:] for l in stdout.splitlines() if l.startswith("@@E@@")), None)
    if not line:
        return {**_EMPTY, "reason": "no_output"}
    res = json.loads(line)
    res["reason"] = "ok"
    return res


def evaluate_suite_sync(reference_code, test_snippets, subset, task_id, mut_sample=None):
    return asyncio.run(evaluate_suite(reference_code, test_snippets, subset, task_id, mut_sample))
