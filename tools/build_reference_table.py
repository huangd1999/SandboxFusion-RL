"""Compute a per-problem reference-instruction table from a parquet dataset.

Reads `(problem_id, reference_solution, test_inputs, expected_outputs)` rows,
runs each reference once via PerfCount sandbox, and emits a JSON mapping
`problem_id -> ref_instructions` plus stats.

The output table feeds an efficiency reward like:
    score = log2(ref_instructions / generated_instructions)

so a slower-than-ref solution gets negative reward, faster gets positive.

Input parquet schema expected (one row per problem):
    - problem_id           : str
    - reference_solution   : str (Python source that passes all tests)
    - test_inputs          : list[str]
    - expected_outputs     : list[str]
    - time_limit           : float | None (optional, default 2.0)

Usage:
    python tools/build_reference_table.py \\
        --input  /path/to/dataset.parquet \\
        --output /path/to/reference_instructions.json \\
        --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq
from client.sandbox_instr_count import SandboxClient


def _measure_row(row: dict, cli: SandboxClient) -> tuple[str, dict]:
    problem_id = row["problem_id"]
    tests = [
        {"stdin": str(s), "expected": str(e)}
        for s, e in zip(row["test_inputs"], row["expected_outputs"])
    ]
    src = row.get("reference_solution") or ""
    if not src or not tests:
        return problem_id, {"ref_instructions": None, "note": "missing_input"}

    result = cli.run(src, tests)
    if not result.all_passed:
        return problem_id, {
            "ref_instructions": None,
            "all_passed": False,
            "note": "reference_failed_tests",
        }
    total = result.total_instructions
    if total is None or total <= 0:
        return problem_id, {
            "ref_instructions": None,
            "all_passed": True,
            "note": "no_perf_counter",
        }
    return problem_id, {
        "ref_instructions": int(total),
        "all_passed": True,
        "n_tests": len(tests),
    }


def main(input_path: str, output_path: str, workers: int, endpoint: str) -> None:
    cli = SandboxClient(endpoint=endpoint)
    if not cli.ping():
        print(f"ERROR: sandbox unreachable at {endpoint}", file=sys.stderr)
        sys.exit(1)

    # Resume support: skip rows already in output.
    out: dict[str, dict] = {}
    if os.path.exists(output_path):
        try:
            out = json.loads(Path(output_path).read_text())
            print(f"Resume: loaded {len(out)} existing entries", flush=True)
        except (json.JSONDecodeError, OSError):
            out = {}

    pf = pq.ParquetFile(input_path)
    print(f"Input: {input_path} ({pf.metadata.num_rows} rows)", flush=True)

    work: list[dict] = []
    for batch in pf.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            if row["problem_id"] not in out:
                work.append(row)

    print(f"To process: {len(work)} (skipping {pf.metadata.num_rows - len(work)} cached)", flush=True)

    t0 = time.time()
    done = 0
    SAVE_EVERY = 200

    def _atomic_save() -> None:
        tmp = output_path + ".tmp"
        Path(tmp).write_text(json.dumps(out, indent=2))
        os.replace(tmp, output_path)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_measure_row, r, cli) for r in work]
        try:
            for f in as_completed(futs):
                pid, res = f.result()
                out[pid] = res
                done += 1
                if done % 50 == 0 or done == len(work):
                    rate = done / max(1e-9, time.time() - t0)
                    eta = (len(work) - done) / rate if rate > 0 else 0
                    with_ref = sum(1 for v in out.values() if v.get("ref_instructions"))
                    print(f"  {done}/{len(work)}  rate={rate:.2f}/s  "
                          f"eta={eta:.0f}s  with_ref={with_ref}", flush=True)
                if done % SAVE_EVERY == 0:
                    _atomic_save()
        finally:
            _atomic_save()

    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="parquet with problem_id/reference_solution/tests")
    p.add_argument("--output", required=True, help="output JSON path")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--endpoint", default=os.environ.get("SANDBOX_FUSION_ENDPOINT", "http://localhost:8080"))
    args = p.parse_args()
    main(args.input, args.output, args.workers, args.endpoint)
