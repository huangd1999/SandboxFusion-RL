# coding: utf-8
# Shared helpers: dataset loading, module/test assembly, diff->code.
#
# Design note (task-level reusable mutant pool):
#   For every task we build a self-contained working dir:
#       _bench_preamble.py   common imports (NOT mutated, NOT counted much)
#       mod.py               `from _bench_preamble import *` + the FUT
#       cosmic-ray.toml      module-path = mod.py, test-command = pytest test.py
#       cosmic-ray.sqlite    enumerated mutant pool (from `cosmic-ray init`)
#   `cosmic-ray init` only parses mod.py, so the pool is independent of any
#   particular test set. Evaluation copies this dir, drops in a candidate
#   test.py, and runs `cosmic-ray exec` against the copy.

import re
import json

# Imports the benchmark functions commonly rely on without declaring them.
# Kept in a separate (un-mutated) preamble module so it barely affects the
# coverage denominator of the function under test.
PREAMBLE = """\
import os
import re
import math
import numpy
import pandas
import pytest
import random
import string
import warnings
import datetime
import itertools
import functools
import collections
import traceback
import numpy as np
import pandas as pd
from typing import (
    List, Dict, Any, Optional, Union, Tuple, Set, FrozenSet,
    Sequence, Iterable, Generator, Callable,
)
"""

TOML_TEMPLATE = """\
[cosmic-ray]
module-path = "mod.py"
timeout = {timeout}
excluded-modules = []
test-command = "{test_command}"

[cosmic-ray.distributor]
name = "local"
"""


def load_dataset(path: str) -> list:
    """PLT/ULT/ULT_Lite files are JSON arrays (despite the .jsonl name)."""
    with open(path, "r") as fh:
        head = fh.read(64).lstrip()
        fh.seek(0)
        if head.startswith("["):
            return json.load(fh)
        return [json.loads(line) for line in fh if line.strip()]


def split_call_lines(ti: str) -> list:
    """Split a test_input blob into top-level call statements, keeping
    multi-line bracketed args together and dropping comment-only lines."""
    lines, buf, depth = [], [], 0
    for raw in (ti or "").split("\n"):
        s = raw.rstrip()
        if not buf and (not s.strip() or s.strip().startswith("#")):
            continue
        buf.append(s)
        depth += s.count("(") + s.count("[") + s.count("{")
        depth -= s.count(")") + s.count("]") + s.count("}")
        if depth <= 0:
            stmt = "\n".join(buf).strip()
            if stmt:
                lines.append(stmt)
            buf, depth = [], 0
    if buf:
        stmt = "\n".join(buf).strip()
        if stmt:
            lines.append(stmt)
    return lines


def strip_inline_comment(expr: str) -> str:
    """Drop a trailing ` # ...` comment from a single-line call expr (only
    when it is not inside a string/bracket). Conservative: skip multi-line."""
    if "\n" in expr or "#" not in expr:
        return expr.strip()
    out, i, depth = [], 0, 0
    quote = None
    s = expr
    while i < len(s):
        c = s[i]
        if quote:
            out.append(c)
            if c == "\\" and i + 1 < len(s):
                out.append(s[i + 1]); i += 2; continue
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c; out.append(c)
        elif c in "([{":
            depth += 1; out.append(c)
        elif c in ")]}":
            depth -= 1; out.append(c)
        elif c == "#" and depth == 0:
            break
        else:
            out.append(c)
        i += 1
    return "".join(out).strip()


# Tolerant comparator embedded into generated oracle test files: handles
# float approx (incl. nested), numpy/pandas, sets, generators->list.
ORACLE_EQ_HELPER = '''
def _eq(a, b, _rel=1e-6, _abs=1e-9):
    import math
    try:
        import numpy as _np
    except Exception:
        _np = None
    import collections.abc as _abc
    if _np is not None and (isinstance(a, _np.ndarray) or isinstance(b, _np.ndarray)):
        try:
            return _np.allclose(_np.asarray(a, dtype=float),
                                _np.asarray(b, dtype=float),
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
        return (a.keys() == b.keys()
                and all(_eq(a[k], b[k]) for k in a))
    if (isinstance(a, _abc.Sequence) and isinstance(b, _abc.Sequence)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, (set, frozenset)) and isinstance(b, (set, frozenset)):
        return a == b
    return a == b
'''.lstrip()


def build_mod_code(fut_code: str) -> str:
    """mod.py = preamble import + the function under test."""
    return "from _bench_preamble import *\n\n\n" + fut_code.strip() + "\n"


def rename_test_functions(test_code: str) -> str:
    """Make every `def test_*` unique so concatenated candidate tests don't
    shadow one another."""
    counter = 0
    out = []
    pat = re.compile(r"(\s*def\s+)(test_)(.*)")
    for line in test_code.split("\n"):
        m = pat.match(line)
        if m:
            counter += 1
            out.append(f"{m.group(1)}test_{counter}_{m.group(3)}")
        else:
            out.append(line)
    return "\n".join(out)


def normalize_tests(tests) -> str:
    """Accept candidate tests as:
        - a list of full `def test_...():` strings, or
        - a list of bare `assert ...` / expression lines, or
        - a single string blob
    and return a runnable test.py body (without the `from mod import *`).
    """
    if isinstance(tests, str):
        tests = [tests]

    blocks = []
    for i, t in enumerate(tests):
        t = t.strip("\n")
        if not t.strip():
            continue
        if re.search(r"^\s*(def |class |@|import |from )", t, re.MULTILINE):
            # already real code (test fn, helper, fixture) -> keep verbatim
            blocks.append(t)
        else:
            # wrap bare asserts / call expressions into a test function
            body = "\n".join("    " + ln if ln.strip() else ln
                             for ln in t.split("\n"))
            blocks.append(f"def test_case_{i}():\n{body}")
    return "\n\n\n".join(blocks)


def build_test_code(tests) -> str:
    body = normalize_tests(tests)
    code = "from mod import *\n\n\n" + body + "\n"
    return rename_test_functions(code)


# --- diff -> mutated source (ported from Ray/generate_mutation_details.py) ----
def get_mutation_code_from_diff(original_code: str, diff: str) -> str:
    if not diff or diff == "No diff":
        return original_code
    original_lines = original_code.splitlines()
    diff_lines = diff.strip().split("\n")
    changes = []
    i = 0
    while i < len(diff_lines):
        line = diff_lines[i]
        m = re.match(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@", line)
        if m:
            old_start, old_count = int(m.group(1)), int(m.group(2))
            new_start, new_count = int(m.group(3)), int(m.group(4))
            i += 1
            old_lines, new_lines = [], []
            while i < len(diff_lines):
                line = diff_lines[i]
                if line.startswith("@@"):
                    break
                if line.startswith("---") or line.startswith("+++"):
                    i += 1
                    continue
                if line.startswith("-"):
                    old_lines.append(line[1:])
                elif line.startswith("+"):
                    new_lines.append(line[1:])
                elif line.startswith(" "):
                    old_lines.append(line[1:])
                    new_lines.append(line[1:])
                i += 1
            changes.append({
                "old_start": old_start, "old_count": old_count,
                "new_lines": new_lines,
            })
        else:
            i += 1
    result = original_lines.copy()
    for ch in reversed(changes):
        s = ch["old_start"] - 1
        del result[s:s + ch["old_count"]]
        result[s:s] = ch["new_lines"]
    return "\n".join(result)
