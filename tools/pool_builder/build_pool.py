# coding: utf-8
# Stage 1 — build the task-level reusable mutant pool.
#
# For every task in a subset:
#   /data/UnLeakedTestBench/pool/<subset>/task_<id>/
#       _bench_preamble.py   mod.py   cosmic-ray.toml   cosmic-ray.sqlite
#   `cosmic-ray init` enumerates the full mutant set from mod.py alone
#   (no tests needed -> the pool is reusable for any candidate test set).
#   mutants.json exports the enumerated specs; pool_manifest.jsonl summarizes.
#
# Usage:
#   <testrl python> pipeline/build_pool.py --subset ULT_Lite
#   <testrl python> pipeline/build_pool.py --subset PLT --tasks /data/.../subsets/PLT_500.txt --workers 32

import os
import sys
import json
import shutil
import sqlite3
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from common import (
    load_dataset, build_mod_code, PREAMBLE, TOML_TEMPLATE,
)
from tqdm.contrib.concurrent import process_map


def _test_command() -> str:
    # exec/baseline use this; init does not, but cosmic-ray validates the toml.
    return f"{config.PYTHON} -m pytest -q -p no:cacheprovider test.py"


def export_mutants(sqlite_path: str) -> list:
    with sqlite3.connect(sqlite_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT job_id, operator_name, occurrence, "
            "start_pos_row, start_pos_col, end_pos_row, end_pos_col "
            "FROM mutation_specs"
        )
        return [
            {
                "job_id": r[0], "operator": r[1], "occurrence": r[2],
                "start_line": r[3], "start_col": r[4],
                "end_line": r[5], "end_col": r[6],
            }
            for r in cur.fetchall()
        ]


def build_one(args):
    subset, task_id, fut_code, timeout = args
    d = config.pool_dir(subset, task_id)
    result = {"subset": subset, "task_id": str(task_id), "task_dir": d,
              "status": "error", "n_mutants": 0, "error": None}
    try:
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

        with open(os.path.join(d, "_bench_preamble.py"), "w") as f:
            f.write(PREAMBLE)
        with open(os.path.join(d, "mod.py"), "w") as f:
            f.write(build_mod_code(fut_code))
        with open(os.path.join(d, "cosmic-ray.toml"), "w") as f:
            f.write(TOML_TEMPLATE.format(timeout=timeout,
                                         test_command=_test_command()))

        # Sanity: mod.py must at least be importable (syntax + preamble).
        chk = subprocess.run(
            [config.PYTHON, "-c", "import ast,io; ast.parse(open('mod.py').read())"],
            cwd=d, capture_output=True, text=True, timeout=30,
        )
        if chk.returncode != 0:
            result["error"] = "mod.py syntax error: " + chk.stderr.strip()[-300:]
            return result

        proc = subprocess.run(
            [config.COSMIC_RAY, "init", "cosmic-ray.toml", "cosmic-ray.sqlite"],
            cwd=d, capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            result["error"] = "cosmic-ray init failed: " + proc.stderr.strip()[-300:]
            return result

        mutants = export_mutants(os.path.join(d, "cosmic-ray.sqlite"))
        with open(os.path.join(d, "mutants.json"), "w") as f:
            json.dump({"subset": subset, "task_id": str(task_id),
                       "n_mutants": len(mutants), "mutants": mutants}, f)

        result["status"] = "ok"
        result["n_mutants"] = len(mutants)
        return result
    except subprocess.TimeoutExpired:
        result["error"] = "timeout"
        return result
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, choices=list(config.DATASET_FILES))
    ap.add_argument("--tasks", default=None,
                    help="optional file with one task_id per line (subset of the dataset)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=10,
                    help="cosmic-ray per-mutant test timeout (seconds)")
    args = ap.parse_args()

    data = load_dataset(config.DATASET_FILES[args.subset])
    by_id = {str(x["task_id"]): x for x in data}

    if args.tasks:
        with open(args.tasks) as f:
            ids = [ln.strip() for ln in f if ln.strip()]
    else:
        ids = list(by_id.keys())
    if args.limit:
        ids = ids[: args.limit]

    jobs = [(args.subset, tid, by_id[tid]["code"], args.timeout)
            for tid in ids if tid in by_id]
    print(f"[+] subset={args.subset} tasks={len(jobs)} workers={args.workers}")

    results = process_map(build_one, jobs, max_workers=args.workers,
                          chunksize=1, desc="[+] cosmic-ray init")

    manifest = os.path.join(config.POOL_ROOT, args.subset, "pool_manifest.jsonl")
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    ok = tot_mut = 0
    with open(manifest, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            if r["status"] == "ok":
                ok += 1
                tot_mut += r["n_mutants"]
    print(f"[+] ok={ok}/{len(jobs)}  total_mutants={tot_mut}  "
          f"avg={tot_mut / ok:.1f}" if ok else "[+] ok=0")
    print(f"[+] manifest -> {manifest}")


if __name__ == "__main__":
    main()
