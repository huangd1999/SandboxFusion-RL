# Copyright 2026 SandboxFusion-PerfCount authors
# SPDX-License-Identifier: Apache-2.0
"""Slim Python client for the PerfCount fork of SandboxFusion.

Submits a Python source + list of stdin/expected test cases, runs the user
code inside the sandbox under a `perf_event_open` hardware-instruction
counter, and returns (per-case correctness, elapsed_ns, retired-instruction
count, all-passed flag).

Why a hardware counter instead of wall-clock? Wall-clock is noisy under
shared load (other GPUs, sandbox concurrency, NIC interrupts). The
`PERF_COUNT_HW_INSTRUCTIONS` event is deterministic: same code on same
input emits the same number of retired instructions every run. This makes
it useful as a relative "efficiency" signal for RLHF / code-eval pipelines.

Usage:

    from client.sandbox_instr_count import SandboxClient

    cli = SandboxClient("http://localhost:8080")
    result = cli.run(
        source="n = int(input())\\nprint(sum(range(n+1)))",
        test_cases=[{"stdin": "10\\n", "expected": "55\\n"}],
    )
    print(result.all_passed, result.per_case[0].instructions)

Sandbox prerequisites (see top-level README):
  * SandboxFusion server running on `endpoint` with `isolation: bwrap`.
  * Host kernel allows perf_event_open (perf_event_paranoid <= 1 if
    instruction counts are required; the driver returns ``None`` for
    `instructions` instead of failing when the syscall is denied).
  * Each subprocess runs under an 8 GB RLIMIT_AS by default (configurable
    in the sandbox patch); legitimate competitive-programming solutions
    fit comfortably, runaway memory-bomb LLM output is bounded.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import requests


DEFAULT_ENDPOINT = os.environ.get("SANDBOX_FUSION_ENDPOINT", "http://localhost:8080").rstrip("/")


# --------------------------------------------------------------------------- #
# In-sandbox driver
# --------------------------------------------------------------------------- #
# This is a single Python script that the sandbox runs once per /run_code
# request. It receives the user source wrapped as `_solve()` and the test
# list, then for each test:
#   1. Resets the perf counter
#   2. Calls `_solve()` with stdin redirected
#   3. Reads back the instruction count + elapsed_ns + stdout
#   4. Compares stdout to expected to decide `passed`
# A JSON blob between `__PERFCOUNT_RESULT__` and `__END__` markers carries
# the aggregated outcome on stderr.

_DRIVER_PREFIX = """\
# SandboxFusion-PerfCount driver. Do not edit.
import sys, io, json, time, ctypes, ctypes.util, os, struct, fcntl

_TESTS = __PERFCOUNT_TESTS__  # filled in by the client

class _PerfInstr:
    _PERF_EVENT_OPEN = 298  # x86_64 syscall
    _PERF_TYPE_HW = 0
    _PERF_HW_INSTRUCTIONS = 1
    _PERF_FLAG_FD_CLOEXEC = 1 << 3
    _IOC_RESET = 0x2403
    _IOC_ENABLE = 0x2400
    _IOC_DISABLE = 0x2401

    class _Attr(ctypes.Structure):
        _fields_ = [
            ('type', ctypes.c_uint32),
            ('size', ctypes.c_uint32),
            ('config', ctypes.c_uint64),
            ('_pad', ctypes.c_uint64 * 16),
        ]

    def __init__(self):
        self.fd = -1
        try:
            libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
            a = self._Attr()
            a.type = self._PERF_TYPE_HW
            a.size = ctypes.sizeof(a)
            a.config = self._PERF_HW_INSTRUCTIONS
            fd = libc.syscall(self._PERF_EVENT_OPEN, ctypes.byref(a),
                              0, -1, -1, self._PERF_FLAG_FD_CLOEXEC)
            if fd >= 0:
                self.fd = fd
        except Exception:
            self.fd = -1

    def measure(self, fn):
        if self.fd < 0:
            return None, fn()
        try:
            fcntl.ioctl(self.fd, self._IOC_RESET, 0)
            fcntl.ioctl(self.fd, self._IOC_ENABLE, 0)
            r = fn()
            fcntl.ioctl(self.fd, self._IOC_DISABLE, 0)
            return struct.unpack('Q', os.read(self.fd, 8))[0], r
        except Exception:
            return None, fn()

    def close(self):
        if self.fd >= 0:
            try: os.close(self.fd)
            except Exception: pass
            self.fd = -1

_PERF = _PerfInstr()

def _capture_run(case):
    import sys as _sys
    out_buf = io.StringIO()
    saved_in, saved_out = _sys.stdin, _sys.stdout
    _sys.stdin = io.StringIO(case["stdin"])
    _sys.stdout = out_buf
    err = [None]
    def _run():
        try:
            _solve()
        except SystemExit:
            pass
        except Exception as e:
            err[0] = "%s: %s" % (type(e).__name__, e)
    t0 = time.perf_counter_ns()
    instr, _ = _PERF.measure(_run)
    elapsed = time.perf_counter_ns() - t0
    _sys.stdin, _sys.stdout = saved_in, saved_out
    return out_buf.getvalue(), elapsed, instr, err[0]

# --- user source wrapped as _solve() ---
"""

_DRIVER_SUFFIX = """

_results = []
for _c in _TESTS:
    _stdout, _ns, _instr, _err = _capture_run(_c)
    _passed = (_err is None) and (_stdout.strip() == _c["expected"].strip())
    _results.append({
        "stdin": _c["stdin"][:200],
        "expected": _c["expected"][:200],
        "stdout": _stdout[:1000],
        "elapsed_ns": _ns,
        "instructions": _instr,
        "error": _err,
        "passed": _passed,
        "tag": _c.get("tag", ""),
    })
_PERF.close()

sys.stderr.write("__PERFCOUNT_RESULT__" + json.dumps({
    "results": _results,
    "all_passed": all(r["passed"] for r in _results),
    "any_run": len(_results) > 0,
}) + "__END__\\n")
"""


def _build_driver(user_source: str, test_cases: list[dict]) -> str:
    indented = "\n".join("    " + line if line else "" for line in user_source.split("\n"))
    wrapper = "def _solve():\n" + indented + "\n"
    prefix_filled = _DRIVER_PREFIX.replace(
        "__PERFCOUNT_TESTS__", json.dumps(test_cases)
    )
    return prefix_filled + wrapper + _DRIVER_SUFFIX


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class CaseResult:
    passed: bool
    elapsed_ns: int
    instructions: Optional[int]
    stdout: str
    error: Optional[str]
    tag: str = ""


@dataclass(slots=True)
class RunResult:
    all_passed: bool
    per_case: list[CaseResult] = field(default_factory=list)
    raw_stderr: str = ""
    sandbox_status: str = ""

    @property
    def total_instructions(self) -> Optional[int]:
        vals = [c.instructions for c in self.per_case if c.instructions]
        return sum(vals) if vals else None


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
@dataclass
class SandboxClient:
    """Posts /run_code to a PerfCount-patched SandboxFusion server."""

    endpoint: str = DEFAULT_ENDPOINT
    language: str = "python"
    run_timeout_s: int = 30
    request_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        # `slots=True` would forbid adding `_session` here; intentionally
        # using a non-slotted dataclass so the lazy session lives on `self`.
        self._session = requests.Session()

    # ------------------------------------------------------------------ #
    def ping(self) -> bool:
        try:
            r = self._session.get(f"{self.endpoint}/v1/ping", timeout=2.0)
            return r.status_code == 200 and "pong" in r.text
        except (requests.RequestException, OSError):
            return False

    def run(self, source: str, test_cases: list[dict]) -> RunResult:
        """Execute `source` under perf-counter on each (stdin, expected) test.

        Args:
            source: Python source (without test harness). The client wraps
                it in `def _solve():` and feeds tests via stdin.
            test_cases: list of `{"stdin": str, "expected": str, "tag": str?}`.

        Returns:
            `RunResult` with per-case correctness + instruction count.
        """
        driver = _build_driver(source, test_cases)
        payload = {
            "code": driver,
            "language": self.language,
            "compile_timeout": 10,
            "run_timeout": self.run_timeout_s,
        }
        try:
            resp = self._session.post(
                f"{self.endpoint}/run_code",
                json=payload,
                timeout=self.request_timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            return RunResult(all_passed=False, sandbox_status=f"unreachable: {exc!r}")

        run_result = data.get("run_result") or {}
        stderr = run_result.get("stderr", "") or ""
        status = data.get("status", "")

        # Extract the marker block.
        marker = "__PERFCOUNT_RESULT__"
        end = "__END__"
        i = stderr.find(marker)
        j = stderr.rfind(end)
        if i < 0 or j < 0 or j <= i:
            return RunResult(all_passed=False, raw_stderr=stderr, sandbox_status=status)

        try:
            blob = json.loads(stderr[i + len(marker):j])
        except json.JSONDecodeError:
            return RunResult(all_passed=False, raw_stderr=stderr, sandbox_status=status)

        cases = [
            CaseResult(
                passed=bool(c["passed"]),
                elapsed_ns=int(c["elapsed_ns"]),
                instructions=int(c["instructions"]) if c.get("instructions") else None,
                stdout=c.get("stdout", ""),
                error=c.get("error"),
                tag=c.get("tag", ""),
            )
            for c in blob.get("results", [])
        ]
        return RunResult(
            all_passed=bool(blob.get("all_passed")),
            per_case=cases,
            raw_stderr="",
            sandbox_status=status,
        )
